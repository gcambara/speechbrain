"""This lobe contains SpeechBrain's implementation of wav2vec2.0

Reference: https://arxiv.org/abs/2006.11477
Reference: https://arxiv.org/abs/1904.05862

Authors
 * Guillermo Cambara 2022
"""

import collections
from einops import rearrange, repeat
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from typing import Optional, Union
from speechbrain.nnet.CNN import Conv1d, ConvTranspose1d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.normalization import GroupNorm, LayerNorm
from speechbrain.lobes.features import Fbank
from speechbrain.lobes.models.transformer.Transformer import TransformerEncoderLayer

class CAPE1d(nn.Module):
    def __init__(self, d_model: int, max_global_shift: float = 0.0, max_local_shift: float = 0.0,
                 max_global_scaling: float = 1.0, normalize: bool = False, freq_scale: float = 1.0,
                 batch_first: bool = False):
        super().__init__()

        assert max_global_shift >= 0, f"""Max global shift is {max_global_shift},
        but should be >= 0."""
        assert max_local_shift >= 0, f"""Max local shift is {max_local_shift},
        but should be >= 0."""
        assert max_global_scaling >= 1, f"""Global scaling is {max_global_scaling},
        but should be >= 1."""

        self.max_global_shift = max_global_shift
        self.max_local_shift = max_local_shift
        self.max_global_scaling = max_global_scaling
        self.normalize = normalize
        self.freq_scale = freq_scale
        self.batch_first = batch_first

        freq = freq_scale * torch.exp(-2.0 * torch.floor(torch.arange(d_model) / 2)
                                      * (math.log(1e4) / d_model))
        self.register_buffer('freq', freq)

        _sin2cos_phase_shift = torch.pi / 2.0
        cos_shifts = _sin2cos_phase_shift * (torch.arange(d_model) % 2)
        self.register_buffer('cos_shifts', cos_shifts)

        self.register_buffer('content_scale', Tensor([math.sqrt(d_model)]))

    def forward(self, x: Tensor, x_lengths: Optional[Tensor] = None,
                positions_delta: Optional[Union[int, Tensor]] = None) -> Tensor:
        return (x * self.content_scale) + self.compute_pos_emb(x, x_lengths, positions_delta)

    def compute_pos_emb(self, x: Tensor, x_lengths: Optional[Tensor] = None,
                        positions_delta: Optional[Union[int, Tensor]] = None) -> Tensor:
        if self.batch_first:
            batch_size, n_tokens, _ = x.shape # b, t, c
        else:
            n_tokens, batch_size, _ = x.shape # t, b, c

        positions = repeat(torch.arange(n_tokens),
                           't -> new_axis t', new_axis=batch_size).to(x)

        if positions_delta is None:
            positions_delta = 1
        else:
            if torch.is_tensor(positions_delta) and len(positions_delta.shape) == 1:
                positions_delta = rearrange(positions_delta, 'b -> b 1')
            positions *= positions_delta

        if x_lengths is not None:
            padding_mask = positions > x_lengths[:, None]
            positions[padding_mask] = float('nan')

        if self.normalize:
            positions -= torch.nanmean(positions, axis=1, keepdim=True)
        
        positions = self.augment_positions(positions, positions_delta)

        positions = rearrange(positions, 'b t -> b t 1')
        product = positions * self.freq.to(x)

        pos_emb = torch.sin(product + self.cos_shifts.to(x))

        if not self.batch_first:
            pos_emb = rearrange(pos_emb, 'b t c -> t b c')

        pos_emb = torch.nan_to_num(pos_emb, nan=0)

        return pos_emb

    def augment_positions(self, positions: Tensor,
                          positions_delta: Optional[Union[int, Tensor]] = None):
        if self.training:
            batch_size, n_tokens = positions.shape

            if self.max_global_shift:
                delta = torch.FloatTensor(batch_size, 1).uniform_(-self.max_global_shift,
                                                                  self.max_global_shift)
                delta = delta.to(positions.device)
            else:
                delta = 0

            if self.max_local_shift:
                epsilon = self.max_local_shift
                delta_local = torch.FloatTensor(batch_size, n_tokens)
                delta_local = delta_local.uniform_(-epsilon,
                                                   epsilon)
                delta_local = delta_local.to(positions.device)
                if positions_delta is not None:
                    if torch.is_tensor(positions_delta) and len(positions_delta.shape) == 1:
                        positions_delta = rearrange(positions_delta, 'b -> b 1')
                    delta_local *= positions_delta
            else:
                delta_local = 0

            if self.max_global_scaling > 1.0:
                log_lambdas = torch.FloatTensor(batch_size, 1)
                log_lambdas = log_lambdas.uniform_(-math.log(self.max_global_scaling),
                                                   math.log(self.max_global_scaling))
                log_lambdas = log_lambdas.to(positions.device)
            else:
                log_lambdas = torch.zeros(1).to(positions.device)

            positions = (positions + delta + delta_local) * torch.exp(log_lambdas)

        return positions

    def set_content_scale(self, content_scale: float):
        self.content_scale = Tensor([content_scale])

class CAPE2d(nn.Module):
    def __init__(self, d_model: int, max_global_shift: float = 0.0, max_local_shift: float = 0.0,
                 max_global_scaling: float = 1.0, batch_first: bool = False):
        super().__init__()

        assert max_global_shift >= 0, f"""Max global shift is {max_global_shift},
        but should be >= 0."""
        assert max_local_shift >= 0, f"""Max local shift is {max_local_shift},
        but should be >= 0."""
        assert max_global_scaling >= 1, f"""Global scaling is {max_global_scaling},
        but should be >= 1."""
        assert d_model % 2 == 0, f"""The number of channels should be even,
                                     but it is odd! # channels = {d_model}."""

        self.max_global_shift = max_global_shift
        self.max_local_shift = max_local_shift
        self.max_global_scaling = max_global_scaling
        self.batch_first = batch_first

        half_channels = d_model // 2
        rho = 10 ** torch.linspace(0, 1, half_channels)
        w_x = rho * torch.cos(torch.arange(half_channels))
        w_y = rho * torch.sin(torch.arange(half_channels))
        self.register_buffer('w_x', w_x)
        self.register_buffer('w_y', w_y)

        self.register_buffer('content_scale', Tensor([math.sqrt(d_model)]))

    def forward(self, patches: Tensor) -> Tensor:
        return (patches * self.content_scale) + self.compute_pos_emb(patches)

    def compute_pos_emb(self, patches: Tensor) -> Tensor:
        if self.batch_first:
            batch_size, patches_x, patches_y, _ = patches.shape # b, x, y, c
        else:
            patches_x, patches_y, batch_size, _ = patches.shape # x, y, b, c

        x = torch.zeros([batch_size, patches_x, patches_y])
        y = torch.zeros([batch_size, patches_x, patches_y])
        x += torch.linspace(-1, 1, patches_x)[None, :, None]
        y += torch.linspace(-1, 1, patches_y)[None, None, :]

        x, y = self.augment_positions(x, y)

        phase = torch.pi * (self.w_x * x[:, :, :, None]
                            + self.w_y * y[:, :, :, None])
        pos_emb = torch.cat([torch.cos(phase), torch.sin(phase)], axis=-1)

        if not self.batch_first:
            pos_emb = rearrange(pos_emb, 'b x y c -> x y b c')

        return pos_emb

    def augment_positions(self, x: Tensor, y: Tensor):
        if self.training:
            batch_size, _, _ = x.shape

            if self.max_global_shift:
                x += (torch.FloatTensor(batch_size, 1, 1).uniform_(-self.max_global_shift,
                                                                   self.max_global_shift)
                     ).to(x.device)
                y += (torch.FloatTensor(batch_size, 1, 1).uniform_(-self.max_global_shift,
                                                                   self.max_global_shift)
                     ).to(y.device)

            if self.max_local_shift:
                diff_x = x[0, -1, 0] - x[0, -2, 0]
                diff_y = y[0, 0, -1] - y[0, 0, -2]
                epsilon_x = diff_x*self.max_local_shift
                epsilon_y = diff_y*self.max_local_shift
                x += torch.FloatTensor(x.shape).uniform_(-epsilon_x,
                                                         epsilon_x).to(x.device)
                y += torch.FloatTensor(y.shape).uniform_(-epsilon_y,
                                                         epsilon_y).to(y.device)

            if self.max_global_scaling > 1.0:
                log_l = math.log(self.max_global_scaling)
                lambdas = (torch.exp(torch.FloatTensor(batch_size, 1, 1).uniform_(-log_l,
                                                                                  log_l))
                          ).to(x.device)
                x *= lambdas
                y *= lambdas

        return x, y

    def set_content_scale(self, content_scale: float):
        self.content_scale = Tensor([content_scale])

class FbankFeaturizer(nn.Module):
    """Filterbank Featurizer
    """

    def __init__(self,
                 n_mels=80,
                 hop_length=10,
                 ):

        super().__init__()
        self.hop_length = hop_length
        self.featurizer = Fbank(n_mels=n_mels, hop_length=hop_length)

    def forward(self, x):
        x = x.squeeze(-1)
        return self.featurizer(x)

class FeatureProjector(nn.Module):
    """ Feature projector
    """

    def __init__(self,
                 n_neurons=512,
                 input_size=768,
                 dropout=0.1,
                 ):
        super().__init__()
        self.linear = Linear(n_neurons=n_neurons, input_size=input_size)
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        k = math.sqrt(1 / input_size)
        nn.init.uniform_(self.linear.w.weight, a=-k, b=k)
        nn.init.uniform_(self.linear.w.bias, a=-k, b=k)

    def forward(self, x):
        x = self.linear(x)
        if self.dropout:
            x = self.dropout(x)
        return x

class FeatureToPatchProjector(nn.Module):
    """ Feature to patch projector
    """

    def __init__(self,
                 patch_sizes,
                 input_size=512,
                 dropout=0.0,
                 ):
        super().__init__()
        self.projectors = self.build_projector(input_size, patch_sizes)

    def forward(self, x, patch_info):
        patches = []
        for i, projector in enumerate(self.projectors):
            offset_start = patch_info[i]['offset']#self.offsets[i]
            if i == len(self.projectors) - 1:
                offset_end = None
            else:
                offset_end = patch_info[i + 1]['offset']
            patch = projector(x[:, offset_start:offset_end, :])
            patches.append(patch)

        return patches

    def build_projector(self, input_size, patch_sizes):
        projectors = nn.ModuleList()
        #offsets = []
        for patch_size in patch_sizes:
            #offset = info['offset']
            #offsets.append(offset)

            time_size, channel_size = patch_size
            projector = Linear(n_neurons=time_size*channel_size, input_size=input_size)
            projectors.append(projector)
        
        return projectors

class ContextExtractorBase(nn.Module):
    """Default context extractor, inspired in wav2vec2's
    """

    def __init__(self,
                 d_ffn=[3072] * 12,
                 nhead=[8] * 12,
                 d_model=[768] * 12,
                 dropout=[0.1] * 12,
                 activation=[nn.GELU] * 12,
                 normalize_before=[True] * 12,
                 layer_drop=0.05
                 ):

        super().__init__()
        assert len(d_ffn) == len(nhead) == len(d_model) == len(dropout) == len(activation) == len(normalize_before), "Error! Check that input lists in the constructor have the same length."

        layers = collections.OrderedDict()
        for i in range(len(d_ffn)):
            layers[f'trn_layer_{i}'] = TransformerEncoderLayer(d_ffn=d_ffn[i],
                                                               nhead=nhead[i],
                                                               d_model=d_model[i],
                                                               dropout=dropout[i],
                                                               activation=activation[i],
                                                               normalize_before=normalize_before[i])

        self.context_extractor = nn.Sequential(layers)
        self.layer_drop = layer_drop
        if normalize_before:
            self.norm = LayerNorm(input_size=d_model[-1])

        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x, output_hidden_states=False):
        hidden_states = []
        for layer in self.context_extractor:
            layer_drop_prob = np.random.random()
            if not self.training or (layer_drop_prob > self.layer_drop):
                x = layer(x)[0]
                if output_hidden_states:
                    hidden_states.append(x)

        if self.norm:
            x = self.norm(x)

        return x, hidden_states

class FeatureMasker(nn.Module):
    def __init__(self,
                 mask_dim=512,
                 mask_prob=0.065,
                 mask_len=2,
                 len_sorting='random'):
        super().__init__()
        self.mask_prob = mask_prob
        self.mask_len = mask_len
        self.len_sorting = len_sorting

        if self.mask_prob > 0:
            self.mask_emb = nn.Parameter(torch.FloatTensor(mask_dim).uniform_())

    def forward(self, x, mask_indices, not_mask_indices):
        ''' Appends the mask tokens at the correct indices. '''
        mask_indices = mask_indices.to(x.device)
        not_mask_indices = not_mask_indices.to(x.device)
        
        masked_embeddings = self.mask_emb.repeat(x.shape[0], len(mask_indices), 1)

        x_ = torch.cat([x[:, :, :], masked_embeddings], dim=1)

        ids_restore = torch.cat([not_mask_indices, mask_indices], dim=-1)
        ids_restore = torch.argsort(ids_restore, dim=0)
        ids_restore = ids_restore.unsqueeze(-1).repeat(x.shape[0], 1, x.shape[2])
        x = torch.gather(x_, dim=1, index=ids_restore)

        return x

    def get_mask(self, input_shape, wav_lens=None, force_masking=True):
        ''' The same mask is applied to every sample in the batch 
            Wav lens indicates the percentage of unpadded samples within
            each wav in the batch, can be used to ignore padded samples.
            We assume right padding.'''

        batch_size, timesteps = input_shape[0], input_shape[1]

        mask_indices = []
        while len(mask_indices) == 0: # if force_masking is set, loop until a mask is generated
            if wav_lens is not None:
                minimum_len = self.get_minimum_len(wav_lens)
                minimum_len = int(minimum_len * timesteps)
                max_min_diff = timesteps - minimum_len
                mask = torch.rand(minimum_len - self.mask_len)
            else:
                mask = torch.rand(timesteps - self.mask_len)
                max_min_diff = 0

            mask = torch.where(mask < self.mask_prob, True, False)
            mask = torch.cat((mask, torch.zeros((self.mask_len + max_min_diff),
                              dtype=bool)), dim=0)

            mask_indices = mask.nonzero()
            mask_indices = mask_indices.squeeze(-1)
            not_mask_indices = (mask == False).nonzero()
            not_mask_indices = not_mask_indices.squeeze(-1)

            if not force_masking:
                break

        for timestep in mask_indices:
            mask[timestep:timestep + self.mask_len] = True

        mask_indices = mask.nonzero()
        mask_indices = mask_indices.squeeze(-1)
        not_mask_indices = (mask == False).nonzero()
        not_mask_indices = not_mask_indices.squeeze(-1)

        mask = mask.unsqueeze(0)
        mask = mask.expand(batch_size, timesteps)

        return mask, mask_indices, not_mask_indices

    def get_mask_from_bigger_patch(self, input_shape, patch_info, wav_lens=None,
                                   force_masking=True):
        bigger_patch_id = None
        curr_patch_size = 0
        for patch_id, info in patch_info.items():
            patch_size = info['patch_size']
            n_patches = patch_size[0]*patch_size[1]
            if n_patches > curr_patch_size:
                bigger_patch_id = patch_id

        offset = patch_info[bigger_patch_id]['offset']
        time_length = patch_info[bigger_patch_id]['time_length']
        time_size = patch_info[bigger_patch_id]['patch_size'][0]

        batch_size, timesteps, channels_size = input_shape
        input_shape = torch.empty(batch_size, time_length, channels_size).shape

        _, mask_indices, not_mask_indices = self.get_mask(input_shape, wav_lens=wav_lens, force_masking=force_masking)
        
        masked_to_time_frame = (mask_indices + 1)*time_size

        mask = []
        # Extrapolate to other patch images indices
        for patch_id, info in patch_info.items():
            patch_time_size = info['patch_size'][0]
            patch_time_length = info['time_length']
            patch_frames = torch.arange(start=1, end=patch_time_length + 1)
            patch_frames *= patch_time_size

            patch_mask = patch_frames < 0
            for upper_frame in masked_to_time_frame:
                lower_frame = upper_frame - time_size
                upper_mask = patch_frames <= upper_frame
                lower_mask = lower_frame < patch_frames
                patch_mask += upper_mask * lower_mask
            mask.append(patch_mask)

        mask = torch.cat(mask)
        mask_indices = mask.nonzero()
        mask_indices = mask_indices.squeeze(-1)
        not_mask_indices = (mask == False).nonzero()
        not_mask_indices = not_mask_indices.squeeze(-1)

        mask = mask.unsqueeze(0)
        mask = mask.expand(batch_size, timesteps)

        return mask, mask_indices, not_mask_indices

    def get_mask_indices_per_patch(self, mask_indices, patch_info):
        mask_indices_per_patch = []
        curr_patch_id = 0
        curr_max_frame = (patch_info[curr_patch_id]['offset'] + 
                          patch_info[curr_patch_id]['time_length'])
        curr_mask_indices = []
        for mask_index in mask_indices:
            if mask_index < curr_max_frame:
                curr_mask_indices.append(int(mask_index))
            else:
                mask_indices_per_patch.append((curr_patch_id, curr_mask_indices))
                curr_mask_indices = [int(mask_index)]
                curr_patch_id += 1
                curr_max_frame = (patch_info[curr_patch_id]['offset'] + 
                                  patch_info[curr_patch_id]['time_length'])

        mask_indices_per_patch.append((curr_patch_id, curr_mask_indices))
        return mask_indices_per_patch

    def get_masks_per_patch(self, mask, patch_info):
        masks_per_patch = []
        for i, info in patch_info.items():
            offset_start = info['offset']
            if i == len(patch_info) - 1:
                offset_end = None
            else:
                offset_end = patch_info[i + 1]['offset']
            
            patch_mask = mask[:, offset_start:offset_end]
            masks_per_patch.append(patch_mask)

        return masks_per_patch

    def get_masked_features(self, x, mask_indices):
        ''' Gets features that are masked '''
        return x[:, mask_indices, :] # expects & returns B, T, C

    def get_minimum_len(self, wav_lens):
        if self.len_sorting == 'random':
            minimum_len = min(wav_lens)
        elif self.len_sorting == 'ascending':
            minimum_len = wav_lens[0]
        elif self.len_sorting == 'descending':
            minimum_len = wav_lens[-1]
        else:
            raise NotImplementedError
        return minimum_len

    def upsample_mask_indices(self, mask_indices, upsample_factor):
        upsampled_mask_indices = []
        for mask_index in mask_indices:
            for i in range(upsample_factor):
                new_mask_index = int(mask_index*upsample_factor) + i
                upsampled_mask_indices.append(new_mask_index)

        upsampled_mask_indices = torch.LongTensor(upsampled_mask_indices)
        return upsampled_mask_indices

class DecoderBase(ContextExtractorBase):
    """
    Default decoder
    """

    def __init__(self):
        super().__init__(d_ffn=[2048] * 8,
                         nhead=[8] * 8,
                         d_model=[512] * 8,
                         dropout=[0.1] * 8,
                         activation=[nn.GELU] * 8,
                         normalize_before=[True] * 8,
                         layer_drop=0.0)

class UpsamplerConv1dBase(nn.Module):
    ''' Basic Upsampler, it just doubles every sample by default '''
    def __init__(self,
                 in_channels=512,
                 out_channels=1280,
                 kernel_size=2,
                 stride=2,
                 padding=0,
                 output_padding=0,
                 groups=1,
                 bias=True,
                 dilation=1):
        super().__init__()
        self.upsampler = ConvTranspose1d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         dilation=dilation,
                                         padding=padding,
                                         output_padding=output_padding,
                                         groups=groups,
                                         bias=bias)

    def forward(self, x):
        return self.upsampler(x)

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="mean")

    def forward(self, preds, targets, masks=None):
        '''
        We can reconstruct different views of the same image at the
        same time. Just pass in the preds, targets and masks for each,
        and the average will be computed. For instance:
        [16x80, 32x80, 64x80]

        If a list with the masks is passed, selection of masked features
        is done within this forward method.
        Otherwise, we expect the masked features to be already passed in.
        '''
        patch_losses = []
        avg_loss = 0.0
        for i, pred in enumerate(preds):
            target = targets[i]

            if masks:
                mask = masks[i]
                pred = pred[mask]
                target = target[mask]

            patch_loss = self.mse_loss(pred, target)
            avg_loss += patch_loss / len(preds)
            patch_losses.append(patch_loss)
        
        return avg_loss, patch_losses

class PatcherLayer(nn.Module):
    ''' Patcher layer. 
        patch_size is a Tuple with T, C dimensions
        patch_stride is a Tuple with T, C dimensions
    '''
    def __init__(self, patch_size=(16, 80), patch_stride=(16, 80), 
                 embedding_dim=768, padding=True, projector=True):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        if projector:
            self.patch_and_projection = nn.Conv2d(1, embedding_dim,
                                                kernel_size=self.patch_size, 
                                                stride=self.patch_stride)
        else:
            self.patch_and_projection = None
        self.padding = padding

    def forward(self, x):
        ''' Input  = (B, 1, C, T) 
            Output = (B, C', T') 
        '''
        ''' Input  = (B, T, C) 
            Output = (B, T', C') 
        '''
        assert len(x.shape) == 3, f"Error! Expected input feature dimensions are 3, but got {len(x.shape)}. Input feature shape = {x.shape}"
        padding = None
        if self.patch_and_projection:
            x = x.unsqueeze(1) # B, 1, T, C
            if self.padding:
                x, padding = self.pad(x, pad_patch_size=self.patch_size)

            x = self.patch_and_projection(x).flatten(2) # B, C', T'
            x = x.transpose(1, 2) # B, T', C'

        return x, padding

    def pad(self, x, pad_patch_size):
        _, _, time_size, channel_size = x.shape
        patch_time_size, patch_channel_size = pad_patch_size

        time_remainder = time_size % patch_time_size 
        channel_remainder = channel_size % patch_channel_size

        if time_remainder != 0:
            time_pad = patch_time_size - time_remainder
        else:
            time_pad = 0
        
        if channel_remainder != 0:
            channel_pad = patch_channel_size - channel_remainder
        else:
            channel_pad = 0

        padding = (0, channel_pad, time_pad, 0)

        x = nn.ZeroPad2d(padding=padding)(x)
        return x, padding

    def get_flat_patches(self, x, pad_patch_size):
        ''' Input = (B, T, C)
            Output = (B, T', C')
        '''

        assert len(x.shape) == 3, f"Error! Expected input feature dimensions are 3, but got {len(x.shape)}. Input feature shape = {x.shape}"

        x = x.unsqueeze(1)
        if self.padding:
            x, _ = self.pad(x, pad_patch_size=pad_patch_size)

        return F.unfold(x, kernel_size=self.patch_size, dilation=1, padding=0, stride=self.patch_stride).transpose(1, 2)

    def get_time_size(self):
        return self.patch_size[0]

    def get_channel_size(self):
        return self.patch_size[1]

    def get_time_stride(self):
        return self.patch_stride[0]

    def get_channel_stride(self):
        return self.patch_stride[1]

    def get_patch_size(self):
        return self.patch_size

    def get_patch_stride(self):
        return self.patch_stride

class PatchAndPos(nn.Module):
    ''' Patcher and positional embedding
    '''
    def __init__(self, patch_sizes=[[16, 80]], patch_strides=[[16, 80]], embedding_dim=768, 
                 padding=True, feat_stride=10.0,
                 positional_embedding=CAPE1d(d_model=768, normalize=True, 
                                             freq_scale=10.0, batch_first=True)):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.patch_strides = patch_strides
        self.embedding_dim = embedding_dim
        self.feat_stride = feat_stride * 0.001 # feat stride input is in ms, convert it to seconds
        self.positional_embedding = positional_embedding
        self.patcher = self.build_patcher(self.patch_sizes, self.patch_strides, 
                                          self.embedding_dim, padding=padding)

    def forward(self, x):
        ''' Input  = (B, T, C) 
            Output = (B, T', C')'''

        patches = []
        patch_info = {}
        patch_offset = 0
        for i, patcher in enumerate(self.patcher):
            patch, padding = patcher(x)

            if self.positional_embedding:
                positions_delta = patcher.get_time_stride() * self.feat_stride
                patch = self.positional_embedding(patch, positions_delta=positions_delta)

            patches.append(patch)
            _, padding_channel, padding_time, _ = padding

            patch_info[i] = {'time_length': patch.size(1), 'patch_size': patcher.get_patch_size(),
                             'patch_stride': patcher.get_patch_stride(), 'offset': patch_offset,
                             'padding_channel': padding_channel, 'padding_time': padding_time}
            
            patch_offset += patch.size(1)

        patches = torch.cat(patches, dim=1)

        return patches, patch_info

    def build_patcher(self, patch_sizes, patch_strides, embedding_dim, padding):
        assert len(patch_sizes) == len(patch_strides), f"Error! The number of input patch sizes and patch strides should be the same. # patch sizes = {len(patch_sizes)}, patch strides = {len(patch_strides)}"

        layers = collections.OrderedDict()
        for i in range(len(patch_sizes)):
            patch_size, patch_stride = patch_sizes[i], patch_strides[i]
            layers[f"patch_layer_{i}"] = PatcherLayer(patch_size, patch_stride, embedding_dim, 
                                                      padding)

        return nn.Sequential(layers)

    def get_flat_patches(self, x):
        flat_patches = []
        for i, patcher in enumerate(self.patcher):
            flat_patch = patcher.get_flat_patches(x)
            flat_patches.append(flat_patch)
        return flat_patches


class Patchies(nn.Module):
    """This lobe is patchies implementation.
    """

    def __init__(self,
                 featurizer=FbankFeaturizer(),
                 patcher=PatchAndPos(),
                 target_patcher=PatcherLayer(projector=False),
                 feat_masker=FeatureMasker(),
                 contextualizer=ContextExtractorBase(),
                 feat_projector=FeatureProjector(),
                 decoder_pos_emb=CAPE1d(d_model=512, normalize=True, 
                                        freq_scale=10.0*16.0, batch_first=True),
                 decoder=DecoderBase(),
                 upsampler=None,
                 feat_to_patch_projector=FeatureToPatchProjector([[16, 80]]),
                 loss=ReconstructionLoss(),
                ):
        super().__init__()
        self.featurizer = featurizer
        self.patcher = patcher
        self.target_patcher = target_patcher
        self.feat_masker = feat_masker
        self.contextualizer = contextualizer
        self.feat_projector = feat_projector
        self.decoder_pos_emb = decoder_pos_emb
        self.decoder = decoder
        self.upsampler = upsampler
        self.feat_to_patch_projector = feat_to_patch_projector
        self.loss = loss

        self.featurizer_hop_length = featurizer.hop_length / 1000.0 # in seconds

    def forward(self, wav, wav_lens=None, apply_mask=False,
                normalize_wav=True, output_norm=False, stage='inference'):
        """Takes an input waveform and returns its corresponding patchies encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """
        if normalize_wav:
            wav = F.layer_norm(wav, normalized_shape=wav.shape[1:])

        feat = self.featurizer(wav)

        if stage == 'train':
            target_patches = self.target_patcher.get_flat_patches(feat, 
                                                                  self.patcher.patch_sizes[-1])
        else:
            target_patches = None
        feat, patch_info = self.patcher(feat)

        if stage == 'train':
            if apply_mask:
                mask, mask_indices, not_mask_indices = self.feat_masker.get_mask(feat.shape, wav_lens=wav_lens)
                unmasked_feat = self.feat_masker.get_masked_features(feat, not_mask_indices)
                masked_feat = self.feat_masker.get_masked_features(feat, mask_indices)
            else:
                unmasked_feat = feat
                mask_indices = None
                not_mask_indices = torch.arange(feat.size(1))

            unmasked_feat, hidden_states = self.contextualizer(unmasked_feat)

            unmasked_feat = self.feat_projector(unmasked_feat)

            if apply_mask:
                feat = self.feat_masker(unmasked_feat, mask_indices, not_mask_indices)
            else:
                feat = unmasked_feat

            if self.decoder_pos_emb:
                positions_delta = (self.featurizer_hop_length * 
                                self.patcher.patch_sizes[-1][0]) # get patch size T. watch out for multires patch!
                feat = self.decoder_pos_emb(feat, positions_delta=positions_delta)

            feat, _ = self.decoder(feat)

            if self.upsampler:
                feat = self.upsampler(feat)
            elif self.feat_to_patch_projector:
                feat = self.feat_to_patch_projector(feat, patch_info)[0] #gcambara: we are working with single patching, for now
        elif stage == 'inference':
            feat, hidden_states = self.contextualizer(feat)
            mask_indices, not_mask_indices = None, None
        else:
            raise NotImplementedError

        if output_norm:
            feat = F.layer_norm(feat, feat.shape)

        return {'feat': feat, 'mask_indices': mask_indices, 'not_mask_indices': not_mask_indices,
                'target_patches': target_patches}

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
