import torch
from src.hub.backbones import _clean_backbone_key

# 1. Load ViT-Base 384 model architecture locally
encoder, predictor = torch.hub.load(
    ".", "vjepa2_1_vit_base_384", source="local", pretrained=False
)

# 2. Load weights from local checkpoint
checkpoint = torch.load("checkpoints/vjepa2_1_vitb_dist_vitG_384.pt", map_location="cpu")
encoder.load_state_dict(_clean_backbone_key(checkpoint["ema_encoder"]), strict=True)
predictor.load_state_dict(_clean_backbone_key(checkpoint["predictor"]), strict=True)

encoder.eval()
print("✓ ViT-Base model loaded successfully!\n")

# -------------------------------------------------------------
# Test Run: Video Input
# -------------------------------------------------------------
# Input shape: (Batch, Channels, Time/Frames, Height, Width)
# Note:
#   - H, W = 384, 384
#   - Patch size = 16x16 -> (384/16) * (384/16) = 24 * 24 = 576 spatial patches
#   - Tubelet size = 2 (each token spans 2 consecutive frames) -> 16 / 2 = 8 temporal steps
#   - Total tokens = 8 temporal steps * 576 spatial patches = 4608 tokens
B, C, T, H, W = 1, 3, 16, 384, 384
video_input = torch.randn(B, C, T, H, W)

print(f"--- Video Input ---")
print(f"Input Shape: {video_input.shape} -> (Batch={B}, Channels={C}, Frames={T}, Height={H}, Width={W})")

with torch.no_grad():
    video_output = encoder(video_input)

print(f"Output Shape: {video_output.shape} -> (Batch={video_output.shape[0]}, Tokens={video_output.shape[1]}, EmbedDim={video_output.shape[2]})")
# Global video embedding via average pooling over tokens:
video_embedding = video_output.mean(dim=1)
print(f"Global Video Embedding (Pooled): {video_embedding.shape}\n")

# -------------------------------------------------------------
# Test Run: Single Image Input
# -------------------------------------------------------------
# For a single image, T = 1: (Batch, Channels, 1, Height, Width)
# Total tokens = 1 * 576 = 576 tokens
image_input = torch.randn(1, 3, 1, 384, 384)

print(f"--- Image Input ---")
print(f"Input Shape: {image_input.shape} -> (Batch=1, Channels=3, Frames=1, Height=384, Width=384)")

with torch.no_grad():
    image_output = encoder(image_input)

print(f"Output Shape: {image_output.shape} -> (Batch={image_output.shape[0]}, Tokens={image_output.shape[1]}, EmbedDim={image_output.shape[2]})")
image_embedding = image_output.mean(dim=1)
print(f"Global Image Embedding (Pooled): {image_embedding.shape}")
