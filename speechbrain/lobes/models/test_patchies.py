import torch
from speechbrain.lobes.models.patchies import CAPE1d, ContextExtractorBase, DecoderBase, FeatureMasker, FeatureProjector, FeatureToPatchProjector, PatchAndPos, Patchies, PatcherLayer, ReconstructionLoss, UpsamplerConv1dBase

#model = Patchies()

#wav = torch.randn(4, 16000, 1)

#feat = model(wav)['feat']
#print(wav.shape)
#print(feat.shape)

def test_multipatch():
    fbank = torch.randn(4, 100, 80)
    patcher_layer = PatcherLayer(patch_size=(16, 16), patch_stride=(16, 16), embedding_dim=768)

    patched, _ = patcher_layer(fbank)

    patcher_layer = PatcherLayer(patch_size=(16, 80), patch_stride=(16, 80), embedding_dim=768)

    patched, _ = patcher_layer(fbank)
    #print(patched.shape)

    patchesitos = patcher_layer.get_flat_patches(fbank, pad_patch_size=patcher_layer.patch_size)
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
    feat_stride = 10.0
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
    mask, mask_indices, _ = feat_masker.get_mask(patches.shape, wav_lens=None)
    print(f"Mask shape = {mask.shape}")
    print("Mask:")
    print(mask)
    print(f"Mask indices = {mask_indices}")
    unmasked_patches = feat_masker.get_masked_features(patches, not_mask_indices)
    print(f"Patches shape = {patches.shape}")
    print(f"Unmasked patches shape = {unmasked_patches.shape}")

    print(f"Unmasked patches = {patches[0, mask_indices]}")
    patches = feat_masker(patches, mask)
    print(f"Masked patches = {patches[0, mask_indices]}")

    print("BIG TO SMALL CASE")
    print(patch_info)
    mask, mask_indices, _ = feat_masker.get_mask_from_bigger_patch(patches.shape, patch_info)
    print(f"Mask indices = {mask_indices}")
    mask_indices_per_patch = feat_masker.get_mask_indices_per_patch(mask_indices, patch_info)
    print(f"Mask indices per patch = {mask_indices_per_patch}")
    masks_per_patch = feat_masker.get_masks_per_patch(mask, patch_info)
    print(f"Masks per patch = {masks_per_patch}")

    print("FEATURE PROJECTOR TEST")
    feat_projector = FeatureProjector()
    print(f"Encoder out shape = {y.shape}")
    y_proj = feat_projector(y)
    print(f"Encoder out projected shape = {y_proj.shape}")

    print("DECODER TEST")
    decoder = DecoderBase()
    y_decoded, _ = decoder(y_proj)
    print(f"Decoder out shape = {y_decoded.shape}")

    print("FEATURE TO PATCH PROJECTOR TEST")
    feat_to_patch = FeatureToPatchProjector(patch_info)
    print(f"Projectors = {feat_to_patch.projectors}")
    print(f"Offsets = {feat_to_patch.offsets}")
    decoded_patches = feat_to_patch(y_decoded)
    print(f"Decoded patches len = {len(decoded_patches)}")
    print(f"Decoded patches shapes = {decoded_patches[0].shape} & {decoded_patches[1].shape}")

    print("FLAT PATCHES FROM PATCHANDPOS TEST")
    patched_fbank = patch_and_pos.get_flat_patches(fbank, pad_patch_size=patch_and_pos.patcher_layer.patch_size)
    print(f"Patched fbank shapes = {patched_fbank[0].shape} & {patched_fbank[1].shape}")

    print("RECONSTRUCTION LOSS TEST")
    reconstruction_loss = ReconstructionLoss()
    loss, patch_losses = reconstruction_loss(patched_fbank, decoded_patches, masks_per_patch)
    print(loss)
    print(patch_losses)

def test_crosspatch():
    print("FBANK TEST")
    fbank = torch.randn(4, 100, 80)
    print(f"FBANK shape = {fbank.shape}")
    print("\n")

    print("PATCH GROUND TRUTH TEST")
    print(f"Patches {16}x{80}")
    patcher_layer = PatcherLayer(patch_size=(16, 80), patch_stride=(16, 80), embedding_dim=768)
    patch_target = patcher_layer.get_flat_patches(fbank, pad_patch_size=patcher_layer.patch_size)
    print(f"Patch target shape = {patch_target.shape}")

    print("PATCH GROUND TRUTH WITH PADDING TO COVER BIGGER PATCH TEST")
    print(f"Patches {16}x{80}")
    patcher_layer = PatcherLayer(patch_size=(16, 80), patch_stride=(16, 80), embedding_dim=768)
    patch_target = patcher_layer.get_flat_patches(fbank, pad_patch_size=(32, 80))
    print(f"Patch target shape = {patch_target.shape}")

    print("PATCH AND POS TEST")
    print(f"Patches {32}x{80}")
    patch_sizes = [[32, 80]]
    patch_strides = [[32, 1]]
    embedding_dim = 768
    feat_stride = 10.0
    patch_and_pos = PatchAndPos(patch_sizes=patch_sizes, patch_strides=patch_strides,
                                embedding_dim=embedding_dim)
    patches, patch_info = patch_and_pos(fbank)
    print(f"Patches shape = {patches.shape}")
    print(f"Patches info = {patch_info}")
    print("\n")

    print("MASKER TEST")
    print("NORMAL CASE (RANDOM MASKING)")
    feat_masker = FeatureMasker(mask_dim=512, mask_prob=0.065, mask_len=2, len_sorting='random')
    mask, mask_indices, not_mask_indices = feat_masker.get_mask(patches.shape, wav_lens=None)
    print(f"Mask shape = {mask.shape}")
    print("Mask:")
    print(mask)
    print(f"Mask indices = {mask_indices}")
    print(f"Not Mask indices = {not_mask_indices}")
    unmasked_patches = feat_masker.get_masked_features(patches, not_mask_indices)
    print(f"Patches shape = {patches.shape}")
    print(f"Unmasked patches shape = {unmasked_patches.shape}")
    print(f"Unmasked patches from function = {unmasked_patches}")

    masked_patches = feat_masker.get_masked_features(patches, mask_indices)
    print(f"Patches shape = {patches.shape}")
    print(f"Masked patches shape = {masked_patches.shape}")
    print(f"Masked patches from function = {masked_patches}")

    masked_patches = feat_masker.get_masked_features(patches, mask_indices)
    unmasked_patches = feat_masker.get_masked_features(patches, not_mask_indices)
    print(f"Masked patches = {masked_patches}")
    print(f"Unmasked patches = {unmasked_patches}")
    print(f"Masked patches shape = {masked_patches.shape}")
    print(f"Unmasked patches shape = {unmasked_patches.shape}")

    print("CONTEXTUALIZER WO HIDDEN STATES TEST")
    contextualizer = ContextExtractorBase()
    feat, hidden_states = contextualizer(unmasked_patches)
    print(f"Contextualizer out shape = {feat.shape}")
    print(f"Hidden states (none) = {hidden_states}")

    print("FEATURE PROJECTOR TEST")
    feat_projector = FeatureProjector()
    print(f"Encoder out shape = {feat.shape}")
    feat = feat_projector(feat)
    print(f"Encoder out projected shape = {feat.shape}")

    print("APPEND MASKED FEATURES")
    feat = feat_masker(feat, mask_indices, not_mask_indices)
    print(f"Feat w masked embs shape = {feat.shape}")

    print("DECODER ADD POSITIONAL EMBEDDING")
    freq_scale = 10.0 * 32.0 # the stride is not 10 ms anymore, but 10 ms * patch size
    decoder_pos_emb = CAPE1d(d_model=512, max_global_shift=0.0, 
                             max_local_shift=0.0, max_global_scaling=1.0, 
                             normalize=True, freq_scale=freq_scale, batch_first=True)
    positions_delta = 0.01 * 32 # 32 * 10 ms of stride
    print(f"Feat decoder without pos emb = {feat}")
    feat = decoder_pos_emb(feat, positions_delta=positions_delta)
    print(f"Feat decoder with pos emb = {feat}")

    print("DECODER TEST")
    decoder = DecoderBase()
    feat, _ = decoder(feat)
    print(f"Decoder out shape = {feat.shape}")

    saved_feat = feat

    print("UPSAMPLE IF NEEDED (YES IN THIS CASE)")
    print(f"Patch target shape = {patch_target.shape}")
    print(f"Decoder not-upsampled shape = {feat.shape}")
    upsampler = UpsamplerConv1dBase()
    feat = upsampler(feat)
    print(f"Decoder upsampled shape = {feat.shape}")

    print("RECONSTRUCTION LOSS TEST")
    print(f"Not upsampled mask indices = {mask_indices}")
    print(mask_indices)
    upsampled_mask_indices = feat_masker.upsample_mask_indices(mask_indices, 2)
    print(f"Upsampled mask indices = {upsampled_mask_indices}")
    pred_masked = feat_masker.get_masked_features(feat, upsampled_mask_indices)
    print(f"Pred masked shape = {pred_masked.shape}")
    target_masked = feat_masker.get_masked_features(patch_target, upsampled_mask_indices)
    print(f"Target masked shape = {target_masked.shape}")
    reconstruction_loss = ReconstructionLoss()
    loss, patch_losses = reconstruction_loss([pred_masked], [target_masked], masks=None)
    print(loss)
    print(patch_losses)

    print("FEATURE TO PATCH PROJECTOR TEST")
    feat_to_patch = FeatureToPatchProjector(patch_sizes)
    print(f"Projectors = {feat_to_patch.projectors}")
    decoded_patches = feat_to_patch(saved_feat, patch_info)
    #print(f"Offsets = {feat_to_patch.offsets}")
    print(f"Decoded patches len = {len(decoded_patches)}")
    print(f"Decoded patches shapes = {decoded_patches[0].shape}")

    # print("FEATURE TO PATCH PROJECTOR TEST")
    # feat_to_patch = FeatureToPatchProjector(patch_info)
    # #print(f"Projectors = {feat_to_patch.projectors}")
    # #print(f"Offsets = {feat_to_patch.offsets}")
    # feat_patches = feat_to_patch(feat)
    # print(f"Decoded patches len = {len(feat_patches)}")
    # print(f"Decoded patches shapes = {feat_patches[0].shape}")






#test_multipatch()
test_crosspatch()