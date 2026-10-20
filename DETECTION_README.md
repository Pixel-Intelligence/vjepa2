# V-JEPA 2 / 2.1 Per-Frame DETR Video Object Detector

This document describes **[`vjepa_detr_detector.py`](file:///c:/Users/shale/OneDrive/Documents/GitHub/vjepa2/vjepa_detr_detector.py)**, its architectural pipeline, how to run it, and the dataset format expected for training and inference.

---

## 1. Overview & Architecture: What is Going On?

The model implemented in [`vjepa_detr_detector.py`](file:///c:/Users/shale/OneDrive/Documents/GitHub/vjepa2/vjepa_detr_detector.py) is `VJEPAPerFrameDETR`. It combines a **frozen self-supervised V-JEPA 2 / 2.1 video foundation model** with a **DETR (DEtection TRansformer) decoder** to perform per-frame object detection across 16-frame video clips.

```
Input Video (B, 3, 16, 384, 384)
            │
            ▼
┌───────────────────────────────────────┐
│ Frozen V-JEPA 2.1 Backbone (ViT-B)    │  Patch: 16x16, Tubelet: 2 frames
└───────────────────────────────────────┘
            │
            ▼ Tokens (B, 4608, 768)
┌───────────────────────────────────────┐
│ Reshape to 5D Grid                    │  (B, 768, T_tube=8, H=24, W=24)
└───────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────┐
│ 3D Conv Projection (1x1x1)            │  Channels: 768 ──> 256 (d_model)
└───────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────┐
│ Trilinear Temporal Upscaling          │  T_tube=8 ──> T=16 native frames
└───────────────────────────────────────┘
            │
            ▼ Frame-batching: (B * 16, 576, 256)
┌───────────────────────────────────────┐
│ DETR Transformer Decoder              │  ◄── 100 Learnable Object Queries
└───────────────────────────────────────┘
            │
      ┌─────┴────────────────┐
      ▼                      ▼
┌──────────────┐      ┌──────────────┐
│  Class Head  │      │   Box Head   │
│ (num_cls + 1)│      │  (3-layer    │
│              │      │     MLP)     │
└──────────────┘      └──────────────┘
      │                      │
      ▼                      ▼
 pred_logits             pred_boxes
(B, 16, 100, 81)       (B, 16, 100, 4)
```

### Detailed Tensor Transformations & Mechanics

1. **Input Video**:
   - Shape: `(B, 3, T=16, H=384, W=384)`.
   - $B$: Batch size, $C=3$: RGB channels, $T=16$: Number of frames, $H=W=384$: Spatial resolution.

2. **Frozen V-JEPA Encoder**:
   - The weights of the V-JEPA backbone are frozen (`param.requires_grad = False`, in `eval()` mode).
   - **Patch Size**: $16 \times 16 \implies H_p = 384 / 16 = 24$, $W_p = 384 / 16 = 24$ (576 spatial patches per temporal step).
   - **Tubelet Size**: $2$ frames $\implies T_{tube} = 16 / 2 = 8$ temporal steps.
   - **Output Tokens**: $8 \times 24 \times 24 = 4608$ tokens of feature dimension $D = 768$ (for ViT-Base).
   - Tensor shape: `(B, 4608, 768)`.

3. **5D Spatiotemporal Feature Grid**:
   - The token sequence is unflattened and permuted into volumetric video feature coordinates:
     ```python
     grid = tokens.view(B, 8, 24, 24, 768).permute(0, 4, 1, 2, 3)
     # Shape: (B, 768, 8, 24, 24) -> (Batch, Channel, Tubelets, Height, Width)
     ```

4. **Channel Projection (`nn.Conv3d`)**:
   - Projects feature dimension from $768$ down to $d_{model} = 256$ using a $1 \times 1 \times 1$ 3D convolution:
     ```python
     feat_256 = self.proj(grid)  # (B, 256, 8, 24, 24)
     ```

5. **Temporal Upsampling (Trilinear Interpolation)**:
   - Because each V-JEPA token represents a $2$-frame tubelet, the temporal dimension is $8$, whereas the original video has $16$ frames.
   - Trilinear interpolation smoothly upsamples the temporal axis from $8 \to 16$ without introducing heavy parameters:
     ```python
     upscaled = F.interpolate(feat_256, size=(16, 24, 24), mode="trilinear", align_corners=False)
     # Shape: (B, 256, 16, 24, 24)
     ```

6. **Frame-Batching for DETR**:
   - The batch dimension ($B$) and temporal frame dimension ($T=16$) are folded together:
     ```python
     memory = upscaled.permute(0, 2, 3, 4, 1).reshape(B * T, 576, 256)
     ```
   - Each frame is treated as an independent spatial memory of $24 \times 24 = 576$ tokens for the Transformer Decoder, retaining spatiotemporally conditioned contextual representations from V-JEPA.

7. **Object Queries & Transformer Decoder**:
   - Learnable query embeddings: `(100, 256)`.
   - Repeated across all $B \times T$ frames: `(B * T, 100, 256)`.
   - Standard 4-layer Transformer Decoder performs multi-head cross-attention ($8$ heads) between the $100$ object queries and the $576$ frame memory tokens.

8. **Prediction Heads**:
   - **Class Head**: `Linear(256, num_classes + 1)`. Predicts class logits for each query plus a background/no-object class index (`num_classes`).
   - **Box Head**: 3-layer MLP (`Linear(256, 256)` -> `ReLU` -> `Linear(256, 256)` -> `ReLU` -> `Linear(256, 4)` -> `Sigmoid`).
   - Outputs normalized coordinates `[cx, cy, w, h]` in the range $[0.0, 1.0]$.
   - Results are reshaped back to per-video, per-frame tensors:
     - `pred_logits`: `(B, 16, 100, num_classes + 1)`
     - `pred_boxes`: `(B, 16, 100, 4)`

---

## 2. How to Run

### Step 1: Activate Environment & Dependencies

Ensure you are using the `vjepa2-312` conda environment:

```powershell
conda activate vjepa2-312
```

Required packages: `torch`, `torchvision`, `timm`, `einops`, `scipy`, `pillow`.

### Step 2: Run Standalone Smoke Test

[`vjepa_detr_detector.py`](file:///c:/Users/shale/OneDrive/Documents/GitHub/vjepa2/vjepa_detr_detector.py) includes a built-in verification execution block that tests the forward pass.

Execute:

```powershell
python vjepa_detr_detector.py
```

**Expected Console Output**:
```text
============================================================
Testing VJEPAPerFrameDETR Model Architecture
============================================================
Found checkpoint at checkpoints/vjepa2_1_vitb_dist_vitG_384.pt. Loading real V-JEPA 2.1 ViT-B backbone...

Input video shape: torch.Size([1, 3, 16, 384, 384]) (B=1, C=3, T=16, H=384, W=384)

Forward pass successful!
Output 'pred_logits' shape: torch.Size([1, 16, 100, 81]) -> (Batch=1, Frames=16, Queries=100, Classes=81)
Output 'pred_boxes'  shape: torch.Size([1, 16, 100, 4]) -> (Batch=1, Frames=16, Queries=100, [cx,cy,w,h]=4)
============================================================
```

*(Note: If the checkpoint is absent, the script automatically uses an internal mock encoder to test and verify tensor shapes).*

---

### Step 3: Using in Custom Python Code

To use the detector model in your own scripts:

```python
import torch
from vjepa_detr_detector import VJEPAPerFrameDETR
from src.hub.backbones import _clean_backbone_key

# 1. Load V-JEPA backbone
encoder, _ = torch.hub.load(".", "vjepa2_1_vit_base_384", source="local", pretrained=False)
ckpt = torch.load("checkpoints/vjepa2_1_vitb_dist_vitG_384.pt", map_location="cpu")
encoder.load_state_dict(_clean_backbone_key(ckpt["ema_encoder"]), strict=True)

# 2. Instantiate Detector
model = VJEPAPerFrameDETR(
    vjepa_encoder=encoder,
    num_classes=80,     # Number of foreground classes
    num_queries=100,    # Max object queries per frame
    d_model=256,
    nhead=8,
    num_decoder_layers=4,
)
model.eval()

# 3. Input video batch (Batch=1, Channels=3, Frames=16, Height=384, Width=384)
video = torch.randn(1, 3, 16, 384, 384)

with torch.no_grad():
    outputs = model(video)

logits = outputs["pred_logits"]  # (1, 16, 100, 81)
boxes = outputs["pred_boxes"]    # (1, 16, 100, 4) in [cx, cy, w, h] normalized format
```

---

### Step 4: Training with `train.py`

Full end-to-end training (Hungarian Matcher, L1 + GIoU + Cross-Entropy loss, AdamW optimizer) is provided in [`train.py`](file:///c:/Users/shale/OneDrive/Documents/GitHub/vjepa2/train.py).

To start training:

```powershell
python train.py
```

---

## 3. Expected Dataset Structure

The model expects video data sliced into **16-frame clips**, with corresponding **per-frame YOLO-format label text files**.

### Directory Structure

```text
data/
└── train/
    ├── clip_0001/
    │   ├── images/
    │   │   ├── 0000.jpg
    │   │   ├── 0001.jpg
    │   │   ├── 0002.jpg
    │   │   │   ...
    │   │   └── 0015.jpg      <-- Exactly 16 images (or .png)
    │   └── labels/
    │       ├── 0000.txt
    │       ├── 0001.txt
    │       │   ...
    │       └── 0015.txt      <-- Corresponding label per frame
    ├── clip_0002/
    │   ├── images/
    │   └── labels/
    └── ...
```

### Label File Format (YOLO Normalized Coordinates)

Each text file corresponds to one frame image with the same base name (e.g., `0000.txt` corresponds to `0000.jpg`).

Each row represents one detected object:
```text
<class_id> <cx> <cy> <w> <h>
```

- **`class_id`**: 0-indexed integer from `0` to `num_classes - 1` (e.g. `0` to `79` for 80 classes).
- **`cx`**: Center X coordinate of the bounding box, normalized by image width (`0.0` to `1.0`).
- **`cy`**: Center Y coordinate of the bounding box, normalized by image height (`0.0` to `1.0`).
- **`w`**: Bounding box width, normalized by image width (`0.0` to `1.0`).
- **`h`**: Bounding box height, normalized by image height (`0.0` to `1.0`).

#### Example `0000.txt`:
```text
0 0.485000 0.352000 0.120000 0.240000
2 0.720000 0.610000 0.080000 0.150000
```

#### Frames with No Objects:
- If a frame contains no objects, the `.txt` file can simply be **empty** (0 bytes), or omitted. The loader interprets this as 0 target objects.

---

### Frame & Tensor Specifications

| Attribute | Expected Value | Description |
| :--- | :--- | :--- |
| **Clip Length** | **16 frames** | Clips longer than 16 are sliced to the first 16; clips shorter are padded by repeating the last frame. |
| **Spatial Resolution** | **$384 \times 384$** | Resized during loading. |
| **Channels** | **3 (RGB)** | Standard RGB images (JPEG/PNG). |
| **Normalization** | **ImageNet Standard** | Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]`. |
| **Batch Tensor Shape** | `(B, 3, 16, 384, 384)` | Channels-first format expected by V-JEPA `(B, C, T, H, W)`. |

---

## 4. Converting Existing Datasets

If your data is currently stored as `.mp4` videos or COCO/MOT annotations:

1. **Extract 16-frame clips**:
   ```bash
   ffmpeg -i source_video.mp4 -vf "select='not(mod(n\,2))',scale=384:384" -vframes 16 clip_0001/images/%04d.jpg
   ```
2. **Convert Annotations to YOLO Format**:
   If starting from pixel coordinates `[x_min, y_min, x_max, y_max]` with frame dimensions $W$ and $H$:
   $$\text{cx} = \frac{x_{min} + x_{max}}{2 \cdot W}, \quad \text{cy} = \frac{y_{min} + y_{max}}{2 \cdot H}$$
   $$\text{w} = \frac{x_{max} - x_{min}}{W}, \quad \text{h} = \frac{y_{max} - y_{min}}{H}$$
   Save these lines into `labels/%04d.txt`.

---

## 5. Loss Formulation & Training Details

In [`train.py`](file:///c:/Users/shale/OneDrive/Documents/GitHub/vjepa2/train.py):
- **Hungarian Matcher**: Per-frame bipartite matching between the 100 queries and ground-truth boxes using `scipy.optimize.linear_sum_assignment`.
- **Cost / Loss Function**:
  $$\mathcal{L}_{total} = \lambda_{ce} \mathcal{L}_{ce} + \lambda_{L1} \mathcal{L}_{L1} + \lambda_{giou} \mathcal{L}_{giou}$$
  Default weights: $\lambda_{ce} = 1.0$, $\lambda_{L1} = 5.0$, $\lambda_{giou} = 2.0$.
  The background class ("no object") is downweighted by an EOS coefficient of $0.1$ to handle class imbalance.
