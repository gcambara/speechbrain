import torch
from speechbrain.lobes.models.patchies import Patchies, PatcherLayer

#model = Patchies()

#wav = torch.randn(4, 16000, 1)

#feat = model(wav)['feat']
#print(wav.shape)
#print(feat.shape)

fbank = torch.randn(4, 100, 80)
patcher_layer = PatcherLayer(patch_size=(16, 16), patch_stride=(16, 16), embedding_dim=768)

patched = patcher_layer(fbank)

patcher_layer = PatcherLayer(patch_size=(16, 80), patch_stride=(16, 80), embedding_dim=768)

patched = patcher_layer(fbank)
print(patched.shape)

patchesitos = patcher_layer.get_flat_patches(fbank)
print(patchesitos.shape)