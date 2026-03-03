from models.experimental.swin_l.tt import TtSwinLBackbone, load_backbone_weights, compute_attn_masks

# Load weights from any mmdet checkpoint with Swin-L backbone
params = load_backbone_weights(checkpoint_path, device)
attn_masks = compute_attn_masks(input_h, input_w, patch_size=4, window_size=12, device=device)

model = TtSwinLBackbone(device, params, attn_masks=attn_masks)
features = model(input_nchw)  # Returns list of 4 NCHW feature maps
