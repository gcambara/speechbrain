import torch
from speechbrain.lobes.models.patchies import CAPE1d, ContextExtractorBase, PatchAndPos, Patchies, PatcherLayer

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
#print(patched.shape)

patchesitos = patcher_layer.get_flat_patches(fbank)
#print(patchesitos.shape)

pos_emb = CAPE1d(d_model=768, max_global_shift=0.0, 
                 max_local_shift=0.0, max_global_scaling=1.0, 
                 normalize=True, freq_scale=10.0, batch_first=True)

positions_delta = 0.16 # 16 * 10 ms of stride
patched_pos = pos_emb(patched, positions_delta=positions_delta)

print("PATCH AND POS TEST")
patch_sizes = [[16, 80], [32, 80]]
patch_strides = [[16, 1], [32, 1]]
embedding_dim = 768
feat_stride = 0.01
patch_and_pos = PatchAndPos(patch_sizes=patch_sizes, patch_strides=patch_strides,
                            embedding_dim=embedding_dim)

fbank = torch.randn(4, 100, 80)
patches, patch_info = patch_and_pos(fbank)
print(patches.shape)
print(patch_info)

contextualizer = ContextExtractorBase()
y, hidden_states = contextualizer(patches)
print(y.shape)
print(hidden_states)

y, hidden_states = contextualizer(patches, True)
print(y.shape)
print(len(hidden_states))
