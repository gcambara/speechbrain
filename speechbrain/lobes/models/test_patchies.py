import torch
from speechbrain.lobes.models.patchies import CAPE1d, ContextExtractorBase, DecoderBase, FeatureMasker, FeatureProjector, PatchAndPos, Patchies, PatcherLayer

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
print(f"Patches shape = {patches.shape}")
print(f"Patches info = {patch_info}")

print("CONTEXTUALIZER WO HIDDEN STATES TEST")
contextualizer = ContextExtractorBase()
y, hidden_states = contextualizer(patches)
print(f"Contextualizer out shape = {y.shape}")
print(f"Hidden states (none) = {hidden_states}")

print("CONTEXTUALIZER W HIDDEN STATES TEST")
y, hidden_states = contextualizer(patches, True)
print(f"Contextualizer out shape = {y.shape}")
print(f"# Hidden states = {len(hidden_states)}")

print("MASKER TEST")
print("NORMAL CASE (RANDOM MASKING ACROSS ALL PATCH RESOLUTIONS)")
feat_masker = FeatureMasker(mask_dim=768, mask_prob=0.065, mask_len=2, len_sorting='random')
mask, mask_indices = feat_masker.get_mask(patches.shape, wav_lens=None)
print(f"Mask shape = {mask.shape}")
print("Mask:")
print(mask)
print(f"Mask indices = {mask_indices}")
unmasked_patches = feat_masker.get_unmasked_features(patches, mask_indices)
print(f"Patches shape = {patches.shape}")
print(f"Unmasked patches shape = {unmasked_patches.shape}")

print(f"Unmasked patches = {patches[0, mask_indices]}")
patches = feat_masker(patches, mask)
print(f"Masked patches = {patches[0, mask_indices]}")

print("BIG TO SMALL CASE")
print(patch_info)
mask, mask_indices = feat_masker.get_mask_from_bigger_patch(patches.shape, patch_info)
print(mask_indices)
mask_indices_per_patch = feat_masker.get_mask_indices_per_patch(mask_indices, patch_info)

print("FEATURE PROJECTOR TEST")
feat_projector = FeatureProjector()
print(f"Encoder out shape = {y.shape}")
y_proj = feat_projector(y)
print(f"Encoder out projected shape = {y_proj.shape}")

print("DECODER TEST")
decoder = DecoderBase()
y_decoded, _ = decoder(y_proj)
print(f"Decoder out shape = {y_decoded.shape}")
