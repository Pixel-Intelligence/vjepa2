import torch
import torch.nn as nn
import torch.nn.functional as F

class VJEPAPerFrameDETR(nn.Module):
    def __init__(
        self,
        vjepa_encoder: nn.Module,
        num_classes: int = 80,
        num_queries: int = 100,
        d_model: int = 256,
        nhead: int = 8,
        num_decoder_layers: int = 4,
    ):
        super().__init__()
        # 1. Frozen V-JEPA backbone
        self.vjepa = vjepa_encoder
        for param in self.vjepa.parameters():
            param.requires_grad = False
        self.vjepa.eval()

        # 2. Channel projection: 768 -> 256
        self.proj = nn.Conv3d(768, d_model, kernel_size=1)

        # 3. Object queries for DETR
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        # 4. Standard DETR Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # 5. Prediction Heads
        self.class_head = nn.Linear(d_model, num_classes + 1)  # +1 for "no object"
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4),
            nn.Sigmoid()  # Normalized coordinates [cx, cy, w, h] in [0, 1]
        )

    def forward(self, video: torch.Tensor):
        """
        video: (B, 3, T, 384, 384) where T = 16
        """
        B, C, T, H, W = video.shape
        H_p, W_p = H // 16, W // 16       # 24, 24
        T_tube = T // 2                   # 8 tubelet steps

        # Step 1: Extract spatiotemporal tokens from frozen V-JEPA
        with torch.no_grad():
            tokens = self.vjepa(video)    # (B, 4608, 768)

        # Step 2: Reshape to 5D grid (B, C=768, T_tube=8, H=24, W=24)
        grid = tokens.view(B, T_tube, H_p, W_p, 768).permute(0, 4, 1, 2, 3)

        # Step 3: 1x1x1 Conv projection (768 -> 256)
        feat_256 = self.proj(grid)        # (B, 256, 8, 24, 24)

        # Step 4: Temporal upscaling via trilinear interpolation (8 -> 16 frames)
        upscaled = F.interpolate(
            feat_256, size=(T, H_p, W_p), mode="trilinear", align_corners=False
        )                                 # (B, 256, 16, 24, 24)

        # Step 5: Frame-batching: fold B and T together for the DETR decoder
        # (B, 256, 16, 24, 24) -> (B * 16, 576, 256)
        memory = upscaled.permute(0, 2, 3, 4, 1).reshape(B * T, H_p * W_p, -1)

        # Step 6: Expand object queries across all (B * T) frames
        queries = self.query_embed.weight.unsqueeze(0).repeat(B * T, 1, 1)  # (B * T, 100, 256)

        # Step 7: DETR Decoder forward pass
        hs = self.decoder(tgt=queries, memory=memory)                       # (B * T, 100, 256)

        # Step 8: Compute outputs and reshape back to (B, T, 100, ...)
        pred_logits = self.class_head(hs).view(B, T, self.num_queries, -1)  # (B, 16, 100, num_classes + 1)
        pred_boxes = self.box_head(hs).view(B, T, self.num_queries, 4)     # (B, 16, 100, 4)

        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes
        }


if __name__ == "__main__":
    import os
    print("=" * 60)
    print("Testing VJEPAPerFrameDETR Model Architecture")
    print("=" * 60)

    # 1. Instantiate or load V-JEPA encoder
    ckpt_path = "checkpoints/vjepa2_1_vitb_dist_vitG_384.pt"
    if os.path.exists(ckpt_path):
        print(f"Found checkpoint at {ckpt_path}. Loading real V-JEPA 2.1 ViT-B backbone...")
        from src.hub.backbones import _clean_backbone_key
        encoder, _ = torch.hub.load(".", "vjepa2_1_vit_base_384", source="local", pretrained=False)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        encoder_key = "ema_encoder" if "ema_encoder" in checkpoint else "target_encoder"
        encoder.load_state_dict(_clean_backbone_key(checkpoint[encoder_key]), strict=True)
        encoder.eval()
    else:
        print(f"Checkpoint not found at {ckpt_path}. Using a mock encoder for shape testing...")
        class MockVJEPA(nn.Module):
            def forward(self, x):
                # x: (B, 3, 16, 384, 384) -> (B, 4608, 768)
                B = x.shape[0]
                return torch.randn(B, 4608, 768, device=x.device)
        encoder = MockVJEPA()

    # 2. Build the detector
    detector = VJEPAPerFrameDETR(
        vjepa_encoder=encoder,
        num_classes=80,
        num_queries=100,
        d_model=256,
        nhead=8,
        num_decoder_layers=4,
    )
    detector.eval()

    # 3. Create dummy input video batch (Batch=1, Channels=3, Frames=16, H=384, W=384)
    B, C, T, H, W = 1, 3, 16, 384, 384
    dummy_video = torch.randn(B, C, T, H, W)
    print(f"\nInput video shape: {dummy_video.shape} (B={B}, C={C}, T={T}, H={H}, W={W})")

    with torch.no_grad():
        outputs = detector(dummy_video)

    pred_logits = outputs["pred_logits"]
    pred_boxes = outputs["pred_boxes"]

    print("\nForward pass successful!")
    print(f"Output 'pred_logits' shape: {pred_logits.shape} -> (Batch={B}, Frames={T}, Queries=100, Classes=81)")
    print(f"Output 'pred_boxes'  shape: {pred_boxes.shape} -> (Batch={B}, Frames={T}, Queries=100, [cx,cy,w,h]=4)")
    print("=" * 60)

