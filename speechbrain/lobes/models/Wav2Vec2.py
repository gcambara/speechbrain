"""This lobe contains SpeechBrain's implementation of wav2vec2.0

Reference: https://arxiv.org/abs/2006.11477
Reference: https://arxiv.org/abs/1904.05862

Authors
 * Guillermo Cambara 2021
"""

import collections
import math
import torch
import torch.nn.functional as F
from torch import nn
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.normalization import LayerNorm
from speechbrain.lobes.models.transformer.Transformer import TransformerEncoderLayer

class W2V2LatentExtractor(nn.Module):
    """wav2vec2.0 default latent extractor
    """

    def __init__(self,
                 in_channels=[1, 512, 512, 512, 512, 512, 512],
                 out_channels=[512] * 7,
                 kernel_size=[10, 3, 3, 3, 3, 2, 2],
                 stride=[5, 2, 2, 2, 2, 2, 2],
                 norms=[LayerNorm(input_size=512)] * 7,
                 acts=[nn.GELU()] * 7
                 ):

        super().__init__()
        assert len(in_channels) == len(out_channels) == len(kernel_size) == len(stride) == len(norms) == len(acts), "Error! Check that input lists in the constructor have the same length."

        blocks = collections.OrderedDict()
        for i in range(len(in_channels)):
            conv_block = collections.OrderedDict()
            conv_block[f'conv1d_{i}'] = Conv1d(out_channels=out_channels[i], kernel_size=kernel_size[i], in_channels=in_channels[i], stride=stride[i])
            conv_block[f'layernorm_{i}'] = norms[i]
            conv_block[f'activation_{i}'] = acts[i]

            blocks[f'conv_block_{i}'] = nn.Sequential(conv_block)

        self.latent_extractor = nn.Sequential(blocks)

    def forward(self, x):
        return self.latent_extractor(x)

class W2V2ContextExtractorBase(nn.Module):
    """wav2vec2.0 default context extractor, inits as BASE
    """

    def __init__(self,
                 d_ffn=[3072] * 12,
                 nhead=[8] * 12,
                 d_model=[768] * 12,
                 dropout=[0.1] * 12,
                 activation=[nn.GELU] * 12,
                 normalize_before=[False] * 12
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

    def forward(self, x):
        for layer in self.context_extractor:
            x = layer(x)[0]
        return x

class W2V2ContextExtractorLarge(nn.Module):
    def __init__(self):
        self.context_extractor = W2V2ContextExtractorBase(d_ffn=[4096] * 24,
                                                          nhead=[16] * 24,
                                                          d_model=[1024] * 24,
                                                          dropout=[0.1] * 24,
                                                          activation=[nn.GELU()] * 24,
                                                          normalize_before=[False] * 24
                                                          )

    def forward(self, x):
        for layer in self.context_extractor:
            x = layer(x)[0]
        return x

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
                 mask_len=10):
        super().__init__()
        self.mask_prob = mask_prob
        self.mask_len = mask_len

        if self.mask_prob > 0:
            self.mask_emb = nn.Parameter(torch.FloatTensor(mask_dim).uniform_())

    def forward(self, x, mask):
        x[mask] = self.mask_emb
        return x

    def get_mask(self, input_shape):
        ''' The same mask is applied to every sample in the batch '''

        batch_size, timesteps = input_shape[0], input_shape[1]

        mask = torch.rand(timesteps - self.mask_len)
        mask = torch.where(mask < self.mask_prob, True, False)
        mask = torch.cat((mask, torch.zeros((self.mask_len), dtype=bool)), dim=0)
        mask_indices = mask.nonzero()
        mask_indices = mask_indices.squeeze(-1)

        for timestep in mask_indices:
            mask[timestep:timestep + self.mask_len] = True

        mask_indices = mask.nonzero()
        mask_indices = mask_indices.squeeze(-1)

        mask = mask.unsqueeze(0)
        mask = mask.expand(batch_size, timesteps)

        return mask, mask_indices

    def get_unmasked_features(self, x, mask_indices):
        return x[:, mask_indices, :] # expects & returns B, T, C

class Wav2Vec2(nn.Module):
    """This lobe is a wav2vec2.0 implementation.
    The idea is that, by default, this is initialized to the
    original wav2vec2 implementation. Everything from wav2vec2
    is self-contained here. Still, modifications can be applied
    with HyperPyYAML.
    """

    def __init__(self,
                 latent_extractor=W2V2LatentExtractor(),
                 latent_projector=Linear(n_neurons=768, input_size=512),
                 positional_encoding=W2V2PositionalEncoding(),
                 context_extractor=W2V2ContextExtractorBase(),
                 vector_quantizer=None,
                 feat_masker=W2V2FeatureMasker(),
                 loss_terms=None
                ):
        super().__init__()
        self.latent_extractor = latent_extractor
        self.latent_projector = latent_projector
        self.positional_encoding = positional_encoding
        self.context_extractor = context_extractor
        self.vector_quantizer = vector_quantizer
        self.feat_masker = feat_masker # any function that returns a feat map masked, with the indices
                                       # a feat masker has the mask vector (learnable or not), and the
                                       # policy to select the indices to be masked
        self.loss_terms = loss_terms # a dict with loss functions to be summed up, plus their weights

    def forward(self, wav, apply_mask=False, return_latent=True):
        """Takes an input waveform and returns its corresponding wav2vec2.0 encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """

        feat = self.latent_extractor(wav)
        feat = self.latent_projector(feat)

        if return_latent:
            latent = feat.clone()
        else:
            latent = None

        if apply_mask:
            mask, mask_indices = self.feat_masker.get_mask(feat.shape) # get indices
            target = self.feat_masker.get_unmasked_features(feat, mask_indices)

            if self.vector_quantizer:
                target = self.vector_quantizer(target)

            feat = self.feat_masker(feat, mask) # mask
        else:
            mask_indices, target = None, None

        if self.positional_encoding:
            feat += self.positional_encoding(feat)
        feat = self.context_extractor(feat)

        return {'feat': feat, 'latent': latent, 'target': target, 'mask_indices': mask_indices}

    def compute_losses(self, pred, target):
        losses = {'total_loss': 0.0}

        for loss_name, loss_tuple in self.loss_terms.items():
            loss_weight, loss_function = loss_tuple
            loss = loss_function(pred, target)

            losses[loss_name] = loss
            losses['total_loss'] += loss_weight * loss

        return losses
