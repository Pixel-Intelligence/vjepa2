import os
from pathlib import Path
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import generalized_box_iou, box_convert
from scipy.optimize import linear_sum_assignment
from PIL import Image
from torchvision import transforms

from vjepa_detr_detector import VJEPAPerFrameDETR
from src.hub.backbones import _clean_backbone_key


# ==============================================================================
# 1. Dataset: Clips with Frame Images & YOLO Text Annotations
# ==============================================================================
# Directory Structure:
# data_root/
#   clip_001/
#     images/
#       0000.jpg, 0001.jpg, ...
#     labels/
#       0000.txt, 0001.txt, ...
#
# Format in YOLO .txt:
# <class_id> <cx> <cy> <w> <h>  (all normalized between 0.0 and 1.0)
# ==============================================================================

class VideoYOLODataset(Dataset):
    def __init__(
        self,
        data_root: str,
        num_frames: int = 16,
        img_size: int = 384,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.num_frames = num_frames
        self.img_size = img_size

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Discover all clip folders
        self.clips = [p for p in self.data_root.iterdir() if p.is_dir() and (p / "images").exists()]
        if not self.clips:
            print(f"[Warning] No clip directories found under {data_root}.")

    def __len__(self):
        return len(self.clips)

    def _load_yolo_labels(self, label_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        if not label_path.exists():
            return torch.zeros((0,), dtype=torch.int64), torch.zeros((0, 4), dtype=torch.float32)

        classes, boxes = [], []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    classes.append(cls_id)
                    boxes.append([cx, cy, w, h])

        if not classes:
            return torch.zeros((0,), dtype=torch.int64), torch.zeros((0, 4), dtype=torch.float32)

        return torch.tensor(classes, dtype=torch.int64), torch.tensor(boxes, dtype=torch.float32)

    def __getitem__(self, idx):
        clip_dir = self.clips[idx]
        image_files = sorted(list((clip_dir / "images").glob("*.jpg")) + list((clip_dir / "images").glob("*.png")))
        label_dir = clip_dir / "labels"

        # Ensure we take exactly num_frames
        if len(image_files) >= self.num_frames:
            sampled_files = image_files[:self.num_frames]
        else:
            # Repeat last frame if clip is shorter
            sampled_files = image_files + [image_files[-1]] * (self.num_frames - len(image_files))

        frame_tensors = []
        frame_targets = []

        for img_path in sampled_files:
            # Read Image
            img = Image.open(img_path).convert("RGB")
            frame_tensors.append(self.transform(img))

            # Read Corresponding Label
            label_path = label_dir / f"{img_path.stem}.txt"
            classes, boxes = self._load_yolo_labels(label_path)
            frame_targets.append({"labels": classes, "boxes": boxes})

        # Stack into (Channels, Num_Frames, Height, Width) -> V-JEPA format
        video_tensor = torch.stack(frame_tensors, dim=1)

        return video_tensor, frame_targets


def collate_fn(batch):
    videos = torch.stack([item[0] for item in batch], dim=0)  # (B, 3, T, H, W)
    targets = [item[1] for item in batch]                     # list of lists of dicts
    return videos, targets


# ==============================================================================
# 2. Hungarian Matcher & DETR Set Criterion (Loss)
# ==============================================================================

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, pred_logits: torch.Tensor, pred_boxes: torch.Tensor, targets: List[Dict]):
        """
        pred_logits: (N_frames, num_queries, num_classes + 1)
        pred_boxes:  (N_frames, num_queries, 4) in [cx, cy, w, h] format
        targets: list of dicts, len = N_frames, each with 'labels' and 'boxes'
        """
        N_frames, num_queries = pred_logits.shape[:2]
        indices = []

        for i in range(N_frames):
            out_prob = pred_logits[i].softmax(-1)
            out_bbox = pred_boxes[i]

            tgt_ids = targets[i]["labels"]
            tgt_bbox = targets[i]["boxes"]

            if len(tgt_ids) == 0:
                indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                continue

            # Class cost: -prob for ground truth class
            cost_class = -out_prob[:, tgt_ids]

            # L1 box cost
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # GIoU cost
            out_xyxy = box_convert(out_bbox, in_fmt="cxcywh", out_fmt="xyxy")
            tgt_xyxy = box_convert(tgt_bbox, in_fmt="cxcywh", out_fmt="xyxy")
            cost_giou = -generalized_box_iou(out_xyxy, tgt_xyxy)

            # Final Cost Matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.cpu()

            src_idx, tgt_idx = linear_sum_assignment(C.numpy())
            indices.append((torch.as_tensor(src_idx, dtype=torch.int64), torch.as_tensor(tgt_idx, dtype=torch.int64)))

        return indices


class DETRSetCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: Dict[str, float],
        eos_coef: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict

        # Downweight background ("no object") class
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict]):
        # Flatten (B, T, ...) -> (B * T, ...)
        B, T = outputs["pred_logits"].shape[:2]
        pred_logits = outputs["pred_logits"].flatten(0, 1)
        pred_boxes = outputs["pred_boxes"].flatten(0, 1)

        # Flatten list of targets (B * T dicts)
        flat_targets = [tgt for batch_item in targets for tgt in batch_item]

        # Match predictions to ground truth
        indices = self.matcher(pred_logits, pred_boxes, flat_targets)

        # 1. Classification Loss
        target_classes = torch.full(
            pred_logits.shape[:2], self.num_classes, dtype=torch.int64, device=pred_logits.device
        )
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(tgt_idx) > 0:
                target_classes[i, src_idx] = flat_targets[i]["labels"][tgt_idx].to(pred_logits.device)

        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, weight=self.empty_weight)

        # 2. Bounding Box Losses (L1 & GIoU)
        num_boxes = sum(len(tgt["labels"]) for tgt in flat_targets)
        num_boxes = max(num_boxes, 1)

        src_boxes = []
        tgt_boxes = []
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                src_boxes.append(pred_boxes[i, src_idx])
                tgt_boxes.append(flat_targets[i]["boxes"][tgt_idx].to(pred_boxes.device))

        if src_boxes:
            src_boxes = torch.cat(src_boxes, dim=0)
            tgt_boxes = torch.cat(tgt_boxes, dim=0)

            loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="sum") / num_boxes

            src_xyxy = box_convert(src_boxes, in_fmt="cxcywh", out_fmt="xyxy")
            tgt_xyxy = box_convert(tgt_boxes, in_fmt="cxcywh", out_fmt="xyxy")
            loss_giou = (1.0 - torch.diag(generalized_box_iou(src_xyxy, tgt_xyxy))).sum() / num_boxes
        else:
            loss_bbox = pred_boxes.sum() * 0.0
            loss_giou = pred_boxes.sum() * 0.0

        losses = {
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
        }

        total_loss = sum(losses[k] * self.weight_dict[k] for k in losses.keys())
        return total_loss, losses


# ==============================================================================
# 3. Model Builder & Training Routine
# ==============================================================================

def build_model(
    vjepa_checkpoint_path: str = "checkpoints/vjepa2_1_vitb_dist_vitG_384.pt",
    model_name: str = "vjepa2_1_vit_base_384",
    num_classes: int = 80,
    num_queries: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"Loading V-JEPA backbone ({model_name}) from {vjepa_checkpoint_path}...")
    encoder, predictor = torch.hub.load(".", model_name, source="local", pretrained=False)

    ckpt = torch.load(vjepa_checkpoint_path, map_location="cpu", weights_only=False)
    encoder_key = "ema_encoder" if "ema_encoder" in ckpt else "target_encoder"
    encoder.load_state_dict(_clean_backbone_key(ckpt[encoder_key]), strict=True)

    model = VJEPAPerFrameDETR(
        vjepa_encoder=encoder,
        num_classes=num_classes,
        num_queries=num_queries,
        d_model=256,
        nhead=8,
        num_decoder_layers=4,
    )
    return model.to(device)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_epoch_loss = 0.0

    for step, (videos, targets) in enumerate(dataloader):
        videos = videos.to(device)  # (B, 3, 16, 384, 384)

        optimizer.zero_grad()
        outputs = model(videos)

        loss, loss_dict = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_epoch_loss += loss.item()

        if step % 5 == 0:
            print(
                f"Step [{step}/{len(dataloader)}] "
                f"Total Loss: {loss.item():.4f} "
                f"(CE: {loss_dict['loss_ce'].item():.4f}, "
                f"L1: {loss_dict['loss_bbox'].item():.4f}, "
                f"GIoU: {loss_dict['loss_giou'].item():.4f})"
            )

    return total_epoch_loss / max(len(dataloader), 1)


# ==============================================================================
# 4. Main Entry Point
# ==============================================================================

def main():
    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    VJEPA_CKPT = "checkpoints/vjepa2_1_vitb_dist_vitG_384.pt"
    DATA_ROOT = "data/train"  # Put your clips here
    NUM_CLASSES = 80          # E.g., COCO classes or your custom number of classes
    BATCH_SIZE = 2
    NUM_EPOCHS = 10
    LR = 1e-4

    print(f"Using device: {DEVICE}")

    # Build Model
    model = build_model(
        vjepa_checkpoint_path=VJEPA_CKPT,
        model_name="vjepa2_1_vit_base_384",
        num_classes=NUM_CLASSES,
        num_queries=100,
        device=DEVICE,
    )

    # Optimizer: Only train the non-frozen parameters (Conv3D projection + DETR decoder)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=1e-4)

    # Criterion & Matcher
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    criterion = DETRSetCriterion(num_classes=NUM_CLASSES, matcher=matcher, weight_dict=weight_dict)

    # Dataset & Dataloader
    if os.path.exists(DATA_ROOT) and any(Path(DATA_ROOT).iterdir()):
        dataset = VideoYOLODataset(data_root=DATA_ROOT, num_frames=16, img_size=384)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

        print("\nStarting Training...")
        for epoch in range(1, NUM_EPOCHS + 1):
            print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")
            avg_loss = train_one_epoch(model, dataloader, optimizer, criterion, DEVICE)
            print(f"Epoch {epoch} Complete. Average Loss: {avg_loss:.4f}")

            # Save checkpoint
            os.makedirs("checkpoints_detr", exist_ok=True)
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer": optimizer.state_dict()},
                f"checkpoints_detr/vjepa_detr_epoch_{epoch}.pt",
            )
    else:
        print(f"\n[Info] Data directory '{DATA_ROOT}' not found or empty.")
        print("To train, create folders under data/train/ like:")
        print("  data/train/clip_001/images/ (0000.jpg, 0001.jpg, ...)")
        print("  data/train/clip_001/labels/ (0000.txt, 0001.txt, ...)")


if __name__ == "__main__":
    main()
