"""
Phase 2 — Topic 4: Segmentation — U-Net & Mask R-CNN
========================================================
Repository : deep-learning-mastery/phase-2-cnns/04-segmentation-unet-maskrcnn/
File       : implementation.py

Sections:
  A │ Synthetic segmentation datasets (semantic masks + per-instance masks)
  B │ U-Net — FULLY from scratch (encoder-decoder with skip connections)
  C │ Dice Loss & combined BCE+Dice loss — verified properties
  D │ Train U-Net on semantic segmentation
  E │ Mask R-CNN via torchvision — custom tiny backbone, trained from scratch
  F │ Evaluation — mIoU (U-Net), mask precision/recall (Mask R-CNN)
  G │ Visualization dashboard
"""

import os, time, warnings
import numpy as np
import matplotlib, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = "results"; os.makedirs(RESULTS, exist_ok=True)
np.random.seed(SEED); torch.manual_seed(SEED)
print(f"[CONFIG] Device: {DEVICE}  |  PyTorch: {torch.__version__}")

IMG_SIZE = 64
CLASS_NAMES = ["circle", "square", "triangle"]   # instance classes (0-indexed)
NUM_CLASSES = len(CLASS_NAMES)                     # for instance seg (Mask R-CNN)
NUM_SEM_CLASSES = NUM_CLASSES + 1                  # +1 background, for U-Net


# ═════════════════════════════════════════════════════════════════════════════
# SECTION A — SYNTHETIC SEGMENTATION DATASETS
# ═════════════════════════════════════════════════════════════════════════════

def _shape_mask_points(cls, cx, cy, r):
    """Return polygon points (or ellipse bbox) for a given shape class."""
    if cls == 0:    # circle
        return ("ellipse", [cx-r, cy-r, cx+r, cy+r])
    elif cls == 1:  # square
        return ("rectangle", [cx-r, cy-r, cx+r, cy+r])
    else:           # triangle
        return ("polygon", [(cx, cy-r), (cx-r, cy+r), (cx+r, cy+r)])


def generate_segmentation_sample(rng, img_size=IMG_SIZE, max_objects=2):
    """
    Generate one image with 1-2 shapes, returning:
      - image (3,H,W) float32
      - semantic_mask (H,W) int64  [0=bg, 1=circle, 2=square, 3=triangle]
      - instance_masks: list of (H,W) uint8 binary masks, one per object
      - boxes: list of [x1,y1,x2,y2]
      - labels: list of class ids (0-indexed, matching CLASS_NAMES)
    """
    img = Image.new("RGB", (img_size, img_size), color=(10, 10, 10))
    draw = ImageDraw.Draw(img)
    semantic_mask = Image.new("L", (img_size, img_size), color=0)

    n_objects = rng.integers(1, max_objects + 1)
    boxes, labels, instance_masks = [], [], []
    placed = []

    attempts = 0
    while len(boxes) < n_objects and attempts < 30:
        attempts += 1
        r = int(rng.integers(7, 11))
        cx = int(rng.integers(r + 2, img_size - r - 2))
        cy = int(rng.integers(r + 2, img_size - r - 2))

        too_close = any(abs(cx - pcx) < (r + pr + 4) and abs(cy - pcy) < (r + pr + 4)
                        for pcx, pcy, pr in placed)
        if too_close:
            continue

        cls = int(rng.integers(0, NUM_CLASSES))
        color = tuple(int(c) for c in rng.integers(110, 256, size=3))

        kind, geom = _shape_mask_points(cls, cx, cy, r)
        if kind == "ellipse":
            draw.ellipse(geom, fill=color)
        elif kind == "rectangle":
            draw.rectangle(geom, fill=color)
        else:
            draw.polygon(geom, fill=color)

        # Draw this instance's binary mask on its OWN clean canvas
        inst_img = Image.new("L", (img_size, img_size), color=0)
        inst_draw = ImageDraw.Draw(inst_img)
        if kind == "ellipse":
            inst_draw.ellipse(geom, fill=1)
        elif kind == "rectangle":
            inst_draw.rectangle(geom, fill=1)
        else:
            inst_draw.polygon(geom, fill=1)
        instance_masks.append(np.array(inst_img, dtype=np.uint8))

        # Draw onto the cumulative semantic mask (class id + 1, since 0=background)
        sem_draw = ImageDraw.Draw(semantic_mask)
        if kind == "ellipse":
            sem_draw.ellipse(geom, fill=cls + 1)
        elif kind == "rectangle":
            sem_draw.rectangle(geom, fill=cls + 1)
        else:
            sem_draw.polygon(geom, fill=cls + 1)

        boxes.append([cx - r, cy - r, cx + r, cy + r])
        labels.append(cls)
        placed.append((cx, cy, r))

    arr = np.array(img, dtype=np.float32) / 255.0
    noise = rng.normal(0, 0.03, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 1).transpose(2, 0, 1)

    sem_mask_arr = np.array(semantic_mask, dtype=np.int64)

    return arr, sem_mask_arr, instance_masks, boxes, labels


class SemanticSegDataset(Dataset):
    """For U-Net: returns (image, semantic_mask)."""
    def __init__(self, n_images, seed=SEED, max_objects=2):
        rng = np.random.default_rng(seed)
        self.images, self.masks = [], []
        for _ in range(n_images):
            img, sem_mask, _, _, _ = generate_segmentation_sample(rng, max_objects=max_objects)
            self.images.append(img)
            self.masks.append(sem_mask)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return (torch.tensor(self.images[idx], dtype=torch.float32),
                torch.tensor(self.masks[idx], dtype=torch.int64))


class InstanceSegDataset(Dataset):
    """For Mask R-CNN: returns (image, boxes, labels, masks)."""
    def __init__(self, n_images, seed=SEED, max_objects=2):
        rng = np.random.default_rng(seed)
        self.images, self.boxes, self.labels, self.masks = [], [], [], []
        for _ in range(n_images):
            img, _, inst_masks, boxes, labels = generate_segmentation_sample(rng, max_objects=max_objects)
            self.images.append(img)
            self.boxes.append(boxes)
            self.labels.append(labels)
            self.masks.append(inst_masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx], dtype=torch.float32)
        boxes = torch.tensor(self.boxes[idx], dtype=torch.float32)
        labels = torch.tensor(self.labels[idx], dtype=torch.int64)
        masks = torch.tensor(np.stack(self.masks[idx]), dtype=torch.uint8)
        return img, boxes, labels, masks


def instance_collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    boxes  = [b[1] for b in batch]
    labels = [b[2] for b in batch]
    masks  = [b[3] for b in batch]
    return images, boxes, labels, masks


def build_segmentation_data(n_train=350, n_val=70, batch_size=16, max_objects=2):
    print("\n" + "="*65)
    print("SECTION A — Synthetic Segmentation Datasets")
    print("="*65)

    sem_train = SemanticSegDataset(n_train, seed=SEED, max_objects=max_objects)
    sem_val   = SemanticSegDataset(n_val,   seed=SEED+1, max_objects=max_objects)
    sem_train_loader = DataLoader(sem_train, batch_size=batch_size, shuffle=True)
    sem_val_loader   = DataLoader(sem_val,   batch_size=batch_size, shuffle=False)

    inst_train = InstanceSegDataset(n_train, seed=SEED, max_objects=max_objects)
    inst_val   = InstanceSegDataset(n_val,   seed=SEED+1, max_objects=max_objects)
    inst_train_loader = DataLoader(inst_train, batch_size=batch_size, shuffle=True,
                                   collate_fn=instance_collate_fn)
    inst_val_loader   = DataLoader(inst_val,   batch_size=batch_size, shuffle=False,
                                   collate_fn=instance_collate_fn)

    print(f"\n  Classes: {CLASS_NAMES}  (+ background for semantic task)")
    print(f"  Train images: {n_train}  |  Val images: {n_val}  |  Image size: {IMG_SIZE}x{IMG_SIZE}")

    return (sem_train_loader, sem_val_loader, sem_val,
            inst_train_loader, inst_val_loader, inst_val)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION B — U-NET FROM SCRATCH
# ═════════════════════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):
    """Two 3x3 convs with BatchNorm+ReLU — the basic U-Net building block."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """
    Full U-Net: 3 encoder downsampling stages + bottleneck + 3 decoder
    upsampling stages with skip connections (channel concatenation).

    Input:  (batch, 3, 64, 64)
    Output: (batch, num_classes, 64, 64)  — raw logits per pixel
    """
    def __init__(self, num_classes=NUM_SEM_CLASSES):
        super().__init__()
        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = DoubleConv(3, 16)     # 64x64
        self.enc2 = DoubleConv(16, 32)    # 32x32 (after pool)
        self.enc3 = DoubleConv(32, 64)    # 16x16 (after pool)
        self.pool = nn.MaxPool2d(2, 2)

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = DoubleConv(64, 128)    # 8x8

        # ── Decoder (upsample + concat skip + double conv) ────────────────────
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec3 = DoubleConv(128 + 64, 64)      # concat bottleneck-up(128) + skip3(64)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = DoubleConv(64 + 32, 32)       # concat dec3-up(64) + skip2(32)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = DoubleConv(32 + 16, 16)       # concat dec2-up(32) + skip1(16)

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)    # per-pixel class logits

    def forward(self, x):
        # Encoder path — save skip connections BEFORE pooling
        s1 = self.enc1(x)              # (B,16,64,64)
        s2 = self.enc2(self.pool(s1))  # (B,32,32,32)
        s3 = self.enc3(self.pool(s2))  # (B,64,16,16)

        b = self.bottleneck(self.pool(s3))    # (B,128,8,8)

        d3 = self.up3(b)                        # (B,128,16,16)
        d3 = self.dec3(torch.cat([d3, s3], dim=1))    # concat → (B,192,16,16) → (B,64,16,16)

        d2 = self.up2(d3)                       # (B,64,32,32)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))    # concat → (B,96,32,32) → (B,32,32,32)

        d1 = self.up1(d2)                       # (B,32,64,64)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))    # concat → (B,48,64,64) → (B,16,64,64)

        return self.final(d1)    # (B,num_classes,64,64)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION C — DICE LOSS & COMBINED LOSS
# ═════════════════════════════════════════════════════════════════════════════

def dice_loss_multiclass(logits: torch.Tensor, target: torch.Tensor,
                         num_classes=NUM_SEM_CLASSES, eps=1e-6) -> torch.Tensor:
    """
    Multi-class soft Dice loss.
    logits: (B,C,H,W) raw scores.  target: (B,H,W) int class labels.
    Computes per-class Dice (using softmax probabilities), averages over classes.
    """
    probs = F.softmax(logits, dim=1)                              # (B,C,H,W)
    target_onehot = F.one_hot(target, num_classes).permute(0,3,1,2).float()    # (B,C,H,W)

    dims = (0, 2, 3)    # sum over batch + spatial, keep per-class
    intersection = torch.sum(probs * target_onehot, dim=dims)
    cardinality  = torch.sum(probs + target_onehot, dim=dims)

    dice_per_class = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice_per_class.mean()


def combined_loss(logits, target, num_classes=NUM_SEM_CLASSES):
    """L = CrossEntropy + Dice — standard practical combination (theory.md §4.3)."""
    ce = F.cross_entropy(logits, target)
    dice = dice_loss_multiclass(logits, target, num_classes)
    return ce + dice, {"ce": ce.item(), "dice": dice.item()}


def section_c_dice_properties():
    """Sanity-check Dice loss: perfect prediction -> loss~0; random -> high loss."""
    print("\n" + "="*65)
    print("SECTION C — Dice Loss Sanity Checks")
    print("="*65)

    target = torch.randint(0, NUM_SEM_CLASSES, (2, 16, 16))

    # Perfect prediction: logits hugely favor the correct class
    perfect_logits = F.one_hot(target, NUM_SEM_CLASSES).permute(0,3,1,2).float() * 20.0
    perfect_loss = dice_loss_multiclass(perfect_logits, target)

    # Random prediction
    random_logits = torch.randn(2, NUM_SEM_CLASSES, 16, 16)
    random_loss = dice_loss_multiclass(random_logits, target)

    print(f"\n  Perfect prediction Dice loss: {perfect_loss.item():.5f}  (expect ≈0)")
    print(f"  Random  prediction Dice loss: {random_loss.item():.5f}  (expect higher)")
    assert perfect_loss.item() < 0.01
    assert random_loss.item() > perfect_loss.item()
    print("\n  ✓ Dice loss behaves correctly: perfect→~0, random→high")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION D — TRAIN U-NET
# ═════════════════════════════════════════════════════════════════════════════

def compute_miou(pred: torch.Tensor, target: torch.Tensor, num_classes=NUM_SEM_CLASSES) -> float:
    """Mean IoU across classes, for one batch. pred,target: (B,H,W) class indices."""
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0


def train_unet(train_loader, val_loader, n_epochs=30, lr=1e-3):
    print("\n" + "="*65)
    print("SECTION D — Training U-Net (from scratch)")
    print("="*65)

    model = UNet().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  U-Net parameters: {n_params:,}")

    history = {"train_loss": [], "val_loss": [], "val_miou": []}
    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()
        tl, tn = 0.0, 0
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            opt.zero_grad()
            logits = model(images)
            loss, _ = combined_loss(logits, masks)
            loss.backward()
            opt.step()
            tl += loss.item() * len(images)
            tn += len(images)

        model.eval()
        vl, vn, miou_sum, miou_n = 0.0, 0, 0.0, 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                logits = model(images)
                loss, _ = combined_loss(logits, masks)
                vl += loss.item() * len(images)
                vn += len(images)
                preds = logits.argmax(dim=1)
                miou_sum += compute_miou(preds, masks) * len(images)
                miou_n += len(images)

        history["train_loss"].append(tl/tn)
        history["val_loss"].append(vl/vn)
        history["val_miou"].append(miou_sum/miou_n)

        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{n_epochs} | train_loss={tl/tn:.4f} | "
                  f"val_loss={vl/vn:.4f} | val_mIoU={miou_sum/miou_n:.4f}")

    elapsed = time.time() - t0
    print(f"\n  ✓ U-Net training complete in {elapsed:.1f}s")
    return model, history


# ═════════════════════════════════════════════════════════════════════════════
# SECTION E — MASK R-CNN VIA TORCHVISION
# ═════════════════════════════════════════════════════════════════════════════

class TinyBackbone(nn.Module):
    """Same minimal backbone design as Phase 2 Topic 3, reused for Mask R-CNN."""
    def __init__(self, out_channels=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=2, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
        self.out_channels = out_channels

    def forward(self, x):
        return self.body(x)


def build_mask_rcnn(num_classes=NUM_CLASSES+1):
    backbone = TinyBackbone(out_channels=64)

    anchor_generator = AnchorGenerator(
        sizes=((16, 32, 48),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    box_roi_pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)
    mask_roi_pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=14, sampling_ratio=2)

    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=box_roi_pooler,
        mask_roi_pool=mask_roi_pooler,
        min_size=IMG_SIZE, max_size=IMG_SIZE,
    )
    return model


def train_mask_rcnn(train_loader, val_loader, n_epochs=6, lr=1e-3):
    print("\n" + "="*65)
    print("SECTION E — Training Mask R-CNN (torchvision, custom tiny backbone)")
    print("="*65)

    model = build_mask_rcnn().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Mask R-CNN (tiny backbone) parameters: {n_params:,}")

    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    history = {"train_loss": []}
    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()
        tl, tn = 0.0, 0
        for images, boxes_list, labels_list, masks_list in train_loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{"boxes": b.to(DEVICE), "labels": (l+1).to(DEVICE), "masks": m.to(DEVICE)}
                      for b, l, m in zip(boxes_list, labels_list, masks_list)]

            opt.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            opt.step()

            tl += loss.item() * len(images)
            tn += len(images)

        history["train_loss"].append(tl/tn)
        print(f"    Epoch {epoch+1:2d}/{n_epochs} | train_loss={tl/tn:.4f}")

    elapsed = time.time() - t0
    print(f"\n  ✓ Mask R-CNN training complete in {elapsed:.1f}s")
    return model, history


@torch.no_grad()
def mask_rcnn_predict(model, image: torch.Tensor, score_threshold=0.5):
    model.eval()
    output = model([image.to(DEVICE)])[0]
    keep = output["scores"] > score_threshold
    boxes = output["boxes"][keep].cpu().numpy()
    labels = output["labels"][keep].cpu().numpy() - 1
    scores = output["scores"][keep].cpu().numpy()
    masks = output["masks"][keep].cpu().numpy()[:, 0] > 0.5    # (N,H,W) binary
    return boxes, labels, scores, masks


# ═════════════════════════════════════════════════════════════════════════════
# SECTION F — EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def evaluate_mask_rcnn(model, inst_val_ds, score_threshold=0.5, iou_threshold=0.5):
    """Mask-IoU based precision/recall, analogous to Topic 3's box-based evaluation."""
    total_tp, total_fp, total_fn = 0, 0, 0

    for i in range(len(inst_val_ds)):
        img, gt_boxes, gt_labels, gt_masks = inst_val_ds[i]
        pred_boxes, pred_labels, pred_scores, pred_masks = mask_rcnn_predict(model, img, score_threshold)

        order = np.argsort(pred_scores)[::-1] if len(pred_scores) > 0 else np.array([], dtype=int)
        pred_labels_s = pred_labels[order]
        pred_masks_s = pred_masks[order]

        gt_masks_np = gt_masks.numpy().astype(bool)
        gt_labels_np = gt_labels.numpy()

        matched = set()
        for plabel, pmask in zip(pred_labels_s, pred_masks_s):
            best_iou, best_idx = 0.0, -1
            for idx in range(len(gt_labels_np)):
                if idx in matched or gt_labels_np[idx] != plabel:
                    continue
                iou = mask_iou(pmask, gt_masks_np[idx])
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            if best_iou >= iou_threshold:
                total_tp += 1
                matched.add(best_idx)
            else:
                total_fp += 1

        total_fn += len(gt_labels_np) - len(matched)

    precision = total_tp / (total_tp + total_fp) if (total_tp+total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp+total_fn) > 0 else 0.0
    return {"precision": precision, "recall": recall, "tp": total_tp, "fp": total_fp, "fn": total_fn}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION G — VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

SEM_COLORS = np.array([
    [10, 10, 10],      # background
    [231, 76, 60],     # circle (red)
    [52, 152, 219],    # square (blue)
    [46, 204, 113],    # triangle (green)
], dtype=np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    return SEM_COLORS[mask]


def build_figures(unet_model, unet_hist, mrcnn_model, mrcnn_hist,
                  sem_val_ds, inst_val_ds, mrcnn_metrics):
    # ── Figure 1: U-Net training curves ───────────────────────────────────────
    fig1, axes1 = plt.subplots(1, 2, figsize=(13, 4.5))
    fig1.suptitle("Phase 2 — Topic 4: U-Net Training", fontsize=13, fontweight="bold")

    ep = range(1, len(unet_hist["train_loss"])+1)
    axes1[0].plot(ep, unet_hist["train_loss"], color="#e74c3c", lw=2, label="Train")
    axes1[0].plot(ep, unet_hist["val_loss"],   color="#3498db", lw=2, label="Val")
    axes1[0].set_title("Combined CE+Dice Loss", fontweight="bold")
    axes1[0].set_xlabel("Epoch"); axes1[0].set_ylabel("Loss")
    axes1[0].legend(fontsize=9); axes1[0].grid(True, alpha=0.3)

    axes1[1].plot(ep, unet_hist["val_miou"], color="#27ae60", lw=2)
    axes1[1].set_title("Validation Mean IoU", fontweight="bold")
    axes1[1].set_xlabel("Epoch"); axes1[1].set_ylabel("mIoU")
    axes1[1].grid(True, alpha=0.3); axes1[1].set_ylim(0, 1.05)

    plt.tight_layout()
    path1 = os.path.join(RESULTS, "04_unet_training.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"\n  [VIZ] Figure 1 saved → {path1}")
    plt.close(fig1)

    # ── Figure 2: U-Net predictions vs ground truth ───────────────────────────
    n_show = 4
    fig2, axes2 = plt.subplots(3, n_show, figsize=(3.3*n_show, 9.5))
    fig2.suptitle("U-Net: Image | Ground-Truth Mask | Predicted Mask", fontsize=12, fontweight="bold")

    unet_model.eval()
    for col in range(n_show):
        img, gt_mask = sem_val_ds[col]
        with torch.no_grad():
            logits = unet_model(img.unsqueeze(0).to(DEVICE))
            pred_mask = logits.argmax(dim=1)[0].cpu().numpy()

        axes2[0, col].imshow(img.numpy().transpose(1,2,0))
        axes2[0, col].set_title(f"Image {col}", fontsize=9); axes2[0, col].axis("off")

        axes2[1, col].imshow(colorize_mask(gt_mask.numpy()))
        axes2[1, col].set_title("Ground Truth", fontsize=9); axes2[1, col].axis("off")

        axes2[2, col].imshow(colorize_mask(pred_mask))
        iou = compute_miou(torch.tensor(pred_mask).unsqueeze(0), gt_mask.unsqueeze(0))
        axes2[2, col].set_title(f"Predicted (mIoU={iou:.2f})", fontsize=9); axes2[2, col].axis("off")

    plt.tight_layout()
    path2 = os.path.join(RESULTS, "04_unet_predictions.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"  [VIZ] Figure 2 saved → {path2}")
    plt.close(fig2)

    # ── Figure 3: Mask R-CNN predictions vs ground truth ──────────────────────
    fig3, axes3 = plt.subplots(2, n_show, figsize=(3.3*n_show, 6.5))
    fig3.suptitle(
        f"Mask R-CNN: Ground Truth (top) vs Predicted Instance Masks (bottom)\n"
        f"Precision={mrcnn_metrics['precision']:.2f}  Recall={mrcnn_metrics['recall']:.2f}",
        fontsize=11, fontweight="bold")

    mrcnn_model.eval()
    for col in range(n_show):
        img, gt_boxes, gt_labels, gt_masks = inst_val_ds[col]
        img_np = img.numpy().transpose(1,2,0)

        # Ground truth overlay
        ax_gt = axes3[0, col]
        ax_gt.imshow(img_np)
        for mask, label in zip(gt_masks.numpy(), gt_labels.numpy()):
            color = np.array(plt.cm.tab10(label % 10)[:3])
            overlay = np.zeros((*mask.shape, 4))
            overlay[mask.astype(bool)] = [*color, 0.5]
            ax_gt.imshow(overlay)
        ax_gt.set_title(f"GT — image {col}", fontsize=9); ax_gt.axis("off")

        # Prediction overlay
        ax_pred = axes3[1, col]
        ax_pred.imshow(img_np)
        boxes, labels, scores, masks = mask_rcnn_predict(mrcnn_model, img, score_threshold=0.5)
        for mask, label, score in zip(masks, labels, scores):
            color = np.array(plt.cm.tab10(int(label) % 10)[:3])
            overlay = np.zeros((*mask.shape, 4))
            overlay[mask] = [*color, 0.5]
            ax_pred.imshow(overlay)
            ys, xs = np.where(mask)
            if len(xs) > 0:
                ax_pred.text(xs.min(), max(ys.min()-2,0), f"{CLASS_NAMES[label]} {score:.2f}",
                           fontsize=6.5, color="white", fontweight="bold",
                           bbox=dict(facecolor="black", alpha=0.6, pad=0.5))
        ax_pred.set_title(f"Pred — image {col}", fontsize=9); ax_pred.axis("off")

    plt.tight_layout()
    path3 = os.path.join(RESULTS, "04_maskrcnn_predictions.png")
    plt.savefig(path3, dpi=150, bbox_inches="tight")
    print(f"  [VIZ] Figure 3 saved → {path3}")
    plt.close(fig3)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "▓"*65)
    print("  Phase 2 — Topic 4: Segmentation — U-Net & Mask R-CNN")
    print("▓"*65)

    (sem_train_loader, sem_val_loader, sem_val_ds,
     inst_train_loader, inst_val_loader, inst_val_ds) = build_segmentation_data(
        n_train=350, n_val=70, batch_size=16, max_objects=2)

    section_c_dice_properties()

    unet_model, unet_hist = train_unet(sem_train_loader, sem_val_loader, n_epochs=30)
    mrcnn_model, mrcnn_hist = train_mask_rcnn(inst_train_loader, inst_val_loader, n_epochs=6)

    print("\n" + "="*65)
    print("SECTION F — Evaluation")
    print("="*65)
    final_miou = unet_hist["val_miou"][-1]
    print(f"\n  U-Net final validation mIoU: {final_miou:.4f}")

    mrcnn_metrics = evaluate_mask_rcnn(mrcnn_model, inst_val_ds)
    print(f"  Mask R-CNN: Precision={mrcnn_metrics['precision']:.3f}  "
          f"Recall={mrcnn_metrics['recall']:.3f}  "
          f"(TP={mrcnn_metrics['tp']}, FP={mrcnn_metrics['fp']}, FN={mrcnn_metrics['fn']})")

    build_figures(unet_model, unet_hist, mrcnn_model, mrcnn_hist,
                 sem_val_ds, inst_val_ds, mrcnn_metrics)

    print("\n" + "▓"*65)
    print("  ✓ Topic 4 complete.")
    print("▓"*65 + "\n")


if __name__ == "__main__":
    main()

