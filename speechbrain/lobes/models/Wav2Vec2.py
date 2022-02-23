"""This lobe contains SpeechBrain's implementation of wav2vec2.0

Reference: https://arxiv.org/abs/2006.11477
Reference: https://arxiv.org/abs/1904.05862

Authors
 * Guillermo Cambara 2022
 * Rudolf Braun      2022
 * Titouan Parcollet 2022
 * Sarthak Yadav     2022
"""

import collections
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.normalization import GroupNorm, LayerNorm
from speechbrain.nnet.quantizers import GumbelVectorQuantizer
from speechbrain.lobes.models.transformer.Transformer import TransformerEncoderLayer

class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None

class W2V2LatentExtractor(nn.Module):
    """wav2vec2.0 default latent extractor
    """

    def __init__(self,
                 in_channels=[1, 512, 512, 512, 512, 512, 512],
                 out_channels=[512] * 7,
                 kernel_size=[10, 3, 3, 3, 3, 2, 2],
                 stride=[5, 2, 2, 2, 2, 2, 2],
                 bias=[False] * 7,
                 norms=[GroupNorm(num_groups=512, input_size=512, affine=True)] + [None] * 6,
                 acts=[nn.GELU()] * 7,
                 init='kaiming',
                 ):

        super().__init__()
        assert len(in_channels) == len(out_channels) == len(kernel_size) == len(stride) == len(norms) == len(acts), "Error! Check that input lists in the constructor have the same length."

        self.kernel_size = kernel_size
        self.stride = stride

        blocks = collections.OrderedDict()
        for i in range(len(in_channels)):
            conv_block = collections.OrderedDict()

            conv = Conv1d(out_channels=out_channels[i],
                          kernel_size=kernel_size[i],
                          in_channels=in_channels[i],
                          stride=stride[i],
                          bias=bias[i])
            if init == 'kaiming':
                nn.init.kaiming_normal_(conv.conv.weight)
            conv_block[f'conv1d_{i}'] = conv

            if norms[i]:
                conv_block[f'norm_{i}'] = norms[i]

            conv_block[f'activation_{i}'] = acts[i]

            blocks[f'conv_block_{i}'] = nn.Sequential(conv_block)

        self.latent_extractor = nn.Sequential(blocks)

    def forward(self, x):
        return self.latent_extractor(x)

    def get_output_lengths(self, input_lengths: torch.LongTensor):
        def _conv_out_length(input_length, kernel_size, stride):
            return torch.round((input_length - kernel_size) / stride + 1)

        for kernel_size, stride in zip(self.kernel_size, self.stride):
            input_lengths = _conv_out_length(
                input_lengths, kernel_size, stride
            )
        return input_lengths.to(torch.long)

class W2V2LatentProjector(nn.Module):
    """wav2vec2.0 default latent projector
    """

    def __init__(self,
                 n_neurons=768,
                 input_size=512,
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

class W2V2ContextExtractorBase(nn.Module):
    """wav2vec2.0 default context extractor, inits as BASE
    """

    def __init__(self,
                 d_ffn=[3072] * 12,
                 nhead=[8] * 12,
                 d_model=[768] * 12,
                 dropout=[0.1] * 12,
                 activation=[nn.GELU] * 12,
                 normalize_before=[False] * 12,
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

        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x, attention_mask=None, output_hidden_states=False):
        hidden_states = []
        for layer in self.context_extractor:
            layer_drop_prob = np.random.random()
            if not self.training or (layer_drop_prob > self.layer_drop):
                x = layer(x, src_key_padding_mask=attention_mask)[0]
                if output_hidden_states:
                    hidden_states.append(x)
        return x, hidden_states

class W2V2ContextExtractorLarge(nn.Module):
    def __init__(self):
        self.context_extractor = W2V2ContextExtractorBase(d_ffn=[4096] * 24,
                                                          nhead=[16] * 24,
                                                          d_model=[1024] * 24,
                                                          dropout=[0.1] * 24,
                                                          activation=[nn.GELU()] * 24,
                                                          normalize_before=[False] * 24,
                                                          layer_drop=0.0
                                                          )

    def forward(self, x, attention_mask=None, output_hidden_states=False):
        hidden_states = []
        for layer in self.context_extractor:
            layer_drop_prob = np.random.random()
            if not self.training or (layer_drop_prob > self.layer_drop):
                x = layer(x, src_key_padding_mask=attention_mask)[0]
                if output_hidden_states:
                    hidden_states.append(x)
        return x, hidden_states

class W2V2PositionalEncoding(nn.Module):
    def __init__(self,
                 in_channels=768,
                 kernel_size=128,
                 stride=1,
                 padding=128 // 2,
                 groups=16,
                 act=nn.GELU()
                 ):
        super().__init__()
        self.positional_encoding = nn.Conv1d(in_channels=in_channels,
                                             out_channels=in_channels,
                                             kernel_size=kernel_size,
                                             stride=stride,
                                             padding=padding,
                                             groups=groups
                                            )

        std = math.sqrt(4 / (kernel_size * in_channels))
        nn.init.normal_(self.positional_encoding.weight, mean=0, std=std)
        nn.init.constant_(self.positional_encoding.bias, 0)
        self.positional_encoding = nn.utils.weight_norm(self.positional_encoding, 
                                                        name="weight", 
                                                        dim=2)
        self.act = act
        self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.positional_encoding(x)

        if self.remove > 0:
            x = x[:, :, : - self.remove]

        x = self.act(x)
        x = x.transpose(1, 2)

        return x

class W2V2FeatureMasker(nn.Module):
    def __init__(self,
                 mask_dim=768,
                 mask_prob=0.065,
                 mask_len=10,
                 len_sorting='random'):
        super().__init__()
        self.mask_prob = mask_prob
        self.mask_len = mask_len
        self.len_sorting = len_sorting

        if self.mask_prob > 0:
            self.mask_emb = nn.Parameter(torch.FloatTensor(mask_dim).uniform_())

    def forward(self, x, mask):
        x[mask] = self.mask_emb
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

            if not force_masking:
                break

        for timestep in mask_indices:
            mask[timestep:timestep + self.mask_len] = True

        mask_indices = mask.nonzero()
        mask_indices = mask_indices.squeeze(-1)

        mask = mask.unsqueeze(0)
        mask = mask.expand(batch_size, timesteps)

        return mask, mask_indices

    def get_unmasked_features(self, x, mask_indices):
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

class W2V2Quantizer(nn.Module):
    def __init__(self,
                 dim=512,
                 num_vars=320,
                 temp=(2, 0.5, 0.999995),
                 groups=2,
                 combine_groups=False,
                 vq_dim=256,
                 time_first=True,
                 activation=nn.GELU(),
                 weight_proj_depth=1,
                 weight_proj_factor=3,
                 input_dropout=0.1
                 ):
        super().__init__()
        self.quantizer = GumbelVectorQuantizer(dim=dim,
                                               num_vars=num_vars,
                                               temp=temp,
                                               groups=groups,
                                               combine_groups=combine_groups,
                                               vq_dim=vq_dim,
                                               time_first=time_first,
                                               activation=activation,
                                               weight_proj_depth=weight_proj_depth,
                                               weight_proj_factor=weight_proj_factor,
                                              )

        if input_dropout > 0.0:
            self.dropout = nn.Dropout(input_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        if self.dropout:
            x = self.dropout(x)
        return self.quantizer(x, produce_targets=False)

class W2V2Loss(nn.Module):
    def __init__(self,
                 contrastive_loss=nn.CrossEntropyLoss(reduction='sum'),
                 contrastive_weight=1.0,
                 diversity_weight=0.1,
                 latent_l2_weight=10.0,
                 similarity=nn.CosineSimilarity(dim=-1),
                 temp=0.1):
        super().__init__()

        self.contrastive_loss = contrastive_loss
        self.contrastive_weight = contrastive_weight
        self.diversity_weight = diversity_weight
        self.latent_l2_weight = latent_l2_weight
        self.similarity = similarity
        self.temp = temp

    def forward(self, feat, pos_target, neg_target, num_vars,
                prob_perplexity, latent_l2):
        loss = 0.0
        if self.contrastive_weight:
            logits = self.compute_logits(feat, pos_target, neg_target)
            target = logits.new_zeros(logits.size(0), dtype=torch.long)
            contrastive_loss = self.contrastive_loss(logits, target)
            loss += self.contrastive_weight * contrastive_loss
        else:
            logits, contrastive_loss = None, None

        if self.diversity_weight:
            diversity_loss = (num_vars - prob_perplexity) / num_vars
            loss += self.diversity_weight * diversity_loss
        else:
            diversity_loss = None

        if self.latent_l2_weight:
            loss += self.latent_l2_weight * latent_l2
        else:
            latent_l2 = None

        return {'loss': loss, 'contrastive_loss': contrastive_loss,
                'diversity_loss': diversity_loss,
                'latent_l2_loss': latent_l2,
                'logits': logits, 'target': target}

    def compute_logits(self, feat, pos_target, neg_target):
        pos_target = pos_target.unsqueeze(0)
        target = torch.cat([pos_target, neg_target], dim=0)
        logits = self.similarity(feat, target) / self.temp # (distr + 1, bsz, masked_feats)

        neg_is_pos = (pos_target == neg_target).all(-1)
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")

        logits = logits.transpose(0, 2).reshape(-1, logits.size(0)) # (bsz x masked_feats, distr + 1)

        return logits

class Wav2Vec2(nn.Module):
    """This lobe is a wav2vec2.0 implementation.
    The idea is that, by default, this is initialized to the
    original wav2vec2 implementation. Everything from wav2vec2
    is self-contained here. Still, modifications can be applied
    with HyperPyYAML.
    """

    def __init__(self,
                 latent_extractor=W2V2LatentExtractor(),
                 latent_projector=W2V2LatentProjector(),
                 latent_norm=LayerNorm(input_size=512),
                 positional_encoding=W2V2PositionalEncoding(),
                 context_extractor=W2V2ContextExtractorBase(),
                 final_projector=Linear(n_neurons=256, input_size=768),
                 target_projector=Linear(n_neurons=256, input_size=256),
                 vector_quantizer=W2V2Quantizer(),
                 feat_masker=W2V2FeatureMasker(),
                 loss=W2V2Loss(),
                ):
        super().__init__()
        self.latent_extractor = latent_extractor
        self.latent_projector = latent_projector
        self.latent_norm = latent_norm
        self.positional_encoding = positional_encoding
        self.context_extractor = context_extractor
        self.final_projector = final_projector
        self.target_projector = target_projector
        self.vector_quantizer = vector_quantizer
        self.feat_masker = feat_masker
        self.loss = loss

    def forward(self, wav, wav_lens=None, apply_mask=False,
                normalize_wav=True, output_norm=False,
                return_latent=False,
                output_hidden_states=False,
                penalize_latent=True, 
                do_final_projection=False,
                latent_grad_weight=1.0):
        """Takes an input waveform and returns its corresponding wav2vec2.0 encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """
        if normalize_wav:
            wav = F.layer_norm(wav, normalized_shape=wav.shape[1:])

        feat = self.latent_extractor(wav)

        if latent_grad_weight != 1.0:
            feat = GradMultiply.apply(feat, latent_grad_weight)

        if penalize_latent:
            latent_l2 = feat.float().pow(2).mean()
        else:
            latent_l2 = None

        feat = self.latent_norm(feat)

        if apply_mask:
            mask, mask_indices = self.feat_masker.get_mask(feat.shape, wav_lens)
            unmasked_feats = self.feat_masker.get_unmasked_features(feat, mask_indices)

            if self.vector_quantizer:
                quant = self.vector_quantizer(unmasked_feats)

                if self.target_projector:
                    quant['x'] = self.target_projector(quant['x'])
                cont_target = None
            else:
                if self.target_projector:
                    cont_target = self.target_projector(unmasked_feats)
                quant = None
        else:
            mask_indices, quant, cont_target = None, None, None

        if self.latent_projector:
            feat = self.latent_projector(feat)

        if return_latent:
            latent = feat.clone()
        else:
            latent = None

        if apply_mask:
            feat = self.feat_masker(feat, mask)

        if self.positional_encoding:
            feat += self.positional_encoding(feat)

        if wav_lens is not None:
            sample_wav_lens = torch.round(wav_lens * wav.size(1))
            output_lens = self.latent_extractor.get_output_lengths(sample_wav_lens)
            attention_mask = self.make_padding_mask(feat.shape, output_lens)
        else:
            attention_mask = None

        feat, hidden_states = self.context_extractor(feat, attention_mask=attention_mask, output_hidden_states=output_hidden_states)

        if self.final_projector and do_final_projection:
            feat = self.final_projector(feat)

        if output_norm:
            feat = F.layer_norm(feat, feat.shape)

        return {'feat': feat, 'latent': latent, 'quant': quant,
                'latent_l2': latent_l2, 'hidden_states': hidden_states,
                'cont_target': cont_target, 'mask_indices': mask_indices}

    def sample_negatives(self, y, num, padding_count=None, num_negatives=100, cross_sample_negatives=0):
        if num_negatives == 0 and cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        y = y.view(-1, fsz)  # BTC => (BxT)C

        # FIXME: what happens if padding_count is specified?
        cross_high = tsz * bsz
        high = tsz - (padding_count or 0)
        with torch.no_grad():
            assert high > 1, f"{bsz,tsz,fsz}"

            if num_negatives > 0:
                tszs = (
                    torch.arange(num)
                    .unsqueeze(-1)
                    .expand(-1, num_negatives)
                    .flatten()
                )

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, num_negatives * num)
                )
                neg_idxs[neg_idxs >= tszs] += 1

            if cross_sample_negatives > 0:
                tszs = (
                    torch.arange(num) # gcambara: this can be buffered and reused
                    .unsqueeze(-1)
                    .expand(-1, cross_sample_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, cross_sample_negatives * num),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if num_negatives > 0:
            for i in range(1, bsz):
                neg_idxs[i] += i * high
        else:
            neg_idxs = cross_neg_idxs

        if cross_sample_negatives > 0 and num_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(
            bsz, num, num_negatives + cross_sample_negatives, fsz
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC

        return negs, neg_idxs

    def make_padding_mask(self, input_shape, abs_lens):
        """ False when to be used, following speechbrain convention """
        padding_mask = torch.arange(input_shape[1])[None, :].to(abs_lens) >= abs_lens[:, None]
        return padding_mask

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
