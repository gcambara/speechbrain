# CommonVoice SSL with SpeechBrain's wav2vec2.0
This folder contains scripts necessary to run a wav2vec2.0 SSL experiment with the CommonVoice dataset: [CommonVoice Homepage](https://commonvoice.mozilla.org/)

# How to run
python train.py hparams/{hparam_file}.py

# Data preparation
It is important to note that CommonVoice initially offers mp3 audio files at 42Hz. Hence, audio files are downsampled on the fly within the dataio function of the training script.

# Usage

The implementation is done to be used as a standalone module outside of SpeechBrain environment, so
anyone can just import the module and play around with it to do inference:

```python
import torch
from speechbrain.lobes.models.Wav2Vec2 import Wav2Vec2

# Inits to wav2vec2.0 BASE
wav2vec2 = Wav2Vec2()

# Forward
x = torch.randn(1, 16000, 1)
out = wav2vec2(x)
feat = out['feat']
# feat.shape = torch.Size([1, 51, 256])
```

Or to train with it:

```python
# Forward
out = wav2vec2(x, apply_mask=True)

# Unpack needed variables to compute
# contrastive and diversity loss
feat = out['feat']
quant = out['quant']
mask_indices = out['mask_indices']
quant_feat = quant['x']
num_vars = quant['num_vars']
prob_perplexity = quant['prob_perplexity']

# Get the mask features and append distractors
feat_masked = feat[:, mask_indices, :]
feat_masked, quant_feat, target = wav2vec2.arrange_distractors(feat_masked,
                                                               quant_feat,
                                                               max_distractors=100)

# Compute the loss
loss_dict = self.modules.wav2vec2.loss(feat_masked, quant_feat, target, num_vars, prob_perplexity)

# Retrieve the loss terms
loss = loss_dict['loss']
contrastive_loss = loss_dict['contrastive_loss']
diversity_loss = loss_dict['diversity_loss']
```

Furthermore, all the modules are easily tweakable at SpeechBrain's YAML configuration:
```
wav2vec2_latent_extractor: !new:speechbrain.lobes.models.Wav2Vec2.W2V2LatentExtractor
    in_channels: [1, 32, 64, 128, 256, 512, 512]
wav2vec2_context_extractor: !new:speechbrain.lobes.models.Wav2Vec2.W2V2ContextExtractorBase
    nhead: [8, 8, 8, 8, 16, 16, 16]
wav2vec2_positional_encoding: !new:speechbrain.lobes.models.Wav2Vec2.W2V2PositionalEncoding
wav2vec2_feature_masker: !new:speechbrain.lobes.models.Wav2Vec2.W2V2FeatureMasker
wav2vec2_vector_quantizer: !new:speechbrain.lobes.models.Wav2Vec2.W2V2Quantizer
wav2vec2_loss: !new:speechbrain.lobes.models.Wav2Vec2.W2V2Loss
    contrastive_weight: 0.1
    diversity_weight: 100

wav2vec2: !new:speechbrain.lobes.models.Wav2Vec2.Wav2Vec2
    latent_extractor:    !ref <wav2vec2_latent_extractor>
    context_extractor:   !ref <wav2vec2_context_extractor>
    positional_encoding: !ref <wav2vec2_positional_encoding>
    vector_quantizer:    !ref <wav2vec2_vector_quantizer>
    feat_masker:         !ref <wav2vec2_feature_masker>
    loss:                !ref <wav2vec2_loss>

```