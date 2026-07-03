"""
Topic: Object Detection — Faster R-CNN & YOLO
=============================================================
Repository : deep-learning/cnns/03-object-detection-rcnn-yolo/
File       : implementation.py

Sections:
  A │ Synthetic object detection dataset (1-2 shapes/image, with bounding boxes)
  B │ IoU & NMS from scratch — verified against torchvision.ops
  C │ Anchor box generation — conceptual demo
  D │ YOLO-style detector — FULLY from scratch (backbone, grid head, custom loss)
  E │ Faster R-CNN via torchvision — custom tiny backbone, trained from scratch
  F │ Evaluation — decode predictions, apply NMS, compute simple precision/recall
  G │ Visualization dashboard — predicted vs ground-truth boxes
"""

import os, time, warnings
import numpy as np
import matplotlib, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.ops import nms as torchvision_nms, box_iou as torchvision_box_iou
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = "results"; os.makedirs(RESULTS, exist_ok=True)
np.random.seed(SEED); torch.manual_seed(SEED)
print(f"[CONFIG] Device: {DEVICE}  |  PyTorch: {torch.__version__}  |  TorchVision: {torchvision.__version__}")

IMG_SIZE = 64
CLASS_NAMES = ["circle", "square", "triangle"]
NUM_CLASSES = len(CLASS_NAMES)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION A — SYNTHETIC OBJECT DETECTION DATASET
# ═════════════════════════════════════════════════════════════════════════════

def _draw_one_shape(draw, cls, cx, cy, r, color):
    if cls == 0:    # circle
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color)
    elif cls == 1:  # square
        draw.rectangle([cx-r, cy-r, cx+r, cy+r], fill=color)
    elif cls == 2:  # triangle
        pts = [(cx, cy-r), (cx-r, cy+r), (cx+r, cy+r)]
        draw.polygon(pts, fill=color)


def generate_detection_image(rng, img_size=IMG_SIZE, max_objects=2):
    """
    Generate one synthetic image with 1-2 non-overlapping shapes.
    Returns: image (3,H,W) float32 in [0,1], boxes list [[x1,y1,x2,y2],...], labels list [int,...]
    """
    img = Image.new("RGB", (img_size, img_size), color=(10, 10, 10))
    draw = ImageDraw.Draw(img)

    n_objects = rng.integers(1, max_objects + 1)
    boxes, labels = [], []
    placed = []

    attempts = 0
    while len(boxes) < n_objects and attempts < 30:
        attempts += 1
        r = int(rng.integers(7, 11))
        cx = int(rng.integers(r + 2, img_size - r - 2))
        cy = int(rng.integers(r + 2, img_size - r - 2))

        # Reject if overlaps an already-placed object (keep detection unambiguous)
        too_close = any(abs(cx - pcx) < (r + pr + 4) and abs(cy - pcy) < (r + pr + 4)
                        for pcx, pcy, pr in placed)
        if too_close:
            continue

        cls = int(rng.integers(0, NUM_CLASSES))
        color = tuple(int(c) for c in rng.integers(110, 256, size=3))
        _draw_one_shape(draw, cls, cx, cy, r, color)

        boxes.append([cx - r, cy - r, cx + r, cy + r])
        labels.append(cls)
        placed.append((cx, cy, r))

    arr = np.array(img, dtype=np.float32) / 255.0
    noise = rng.normal(0, 0.03, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 1).transpose(2, 0, 1)    # (3,H,W)

    return arr, boxes, labels


class DetectionDataset(Dataset):
    """Synthetic object detection dataset — returns (image, boxes, labels) per sample."""
    def __init__(self, n_images, seed=SEED, max_objects=2):
        rng = np.random.default_rng(seed)
        self.images, self.boxes, self.labels = [], [], []
        for _ in range(n_images):
            img, boxes, labels = generate_detection_image(rng, max_objects=max_objects)
            self.images.append(img)
            self.boxes.append(boxes)
            self.labels.append(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx], dtype=torch.float32)
        boxes = torch.tensor(self.boxes[idx], dtype=torch.float32)
        labels = torch.tensor(self.labels[idx], dtype=torch.int64)
        return img, boxes, labels


def detection_collate_fn(batch):
    """Custom collate: images stack normally, boxes/labels stay as a list (variable length)."""
    images = torch.stack([b[0] for b in batch])
    boxes  = [b[1] for b in batch]
    labels = [b[2] for b in batch]
    return images, boxes, labels


def build_detection_data(n_train=400, n_val=80, batch_size=16, max_objects=2):
    print("\n" + "="*65)
    print("SECTION A — Synthetic Object Detection Dataset")
    print("="*65)

    train_ds = DetectionDataset(n_train, seed=SEED, max_objects=max_objects)
    val_ds   = DetectionDataset(n_val,   seed=SEED+1, max_objects=max_objects)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=detection_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=detection_collate_fn)

    avg_objs = np.mean([len(train_ds.labels[i]) for i in range(len(train_ds))])
    print(f"\n  Classes: {CLASS_NAMES}")
    print(f"  Train images: {n_train}  |  Val images: {n_val}")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}  |  Avg objects/image: {avg_objs:.2f}")

    return train_loader, val_loader, train_ds, val_ds


# ═════════════════════════════════════════════════════════════════════════════
# SECTION B — IoU & NMS FROM SCRATCH
# ═════════════════════════════════════════════════════════════════════════════

def iou_scratch(box1: np.ndarray, box2: np.ndarray) -> float:
    """IoU between two boxes [x1,y1,x2,y2]."""
    x_left   = max(box1[0], box2[0])
    y_top    = max(box1[1], box2[1])
    x_right  = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        intersection = 0.0
    else:
        intersection = (x_right - x_left) * (y_bottom - y_top)

    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def iou_batch_scratch(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Vectorized pairwise IoU. boxes1:(N,4), boxes2:(M,4) -> (N,M)"""
    N, M = len(boxes1), len(boxes2)
    iou_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            iou_mat[i, j] = iou_scratch(boxes1[i], boxes2[j])
    return iou_mat


def nms_scratch(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> list:
    """
    Non-Maximum Suppression from scratch.
    boxes: (N,4) [x1,y1,x2,y2],  scores: (N,)
    Returns: list of indices to KEEP
    """
    order = np.argsort(scores)[::-1]    # sort descending by score
    keep = []

    while len(order) > 0:
        current = order[0]
        keep.append(int(current))

        if len(order) == 1:
            break

        rest = order[1:]
        ious = np.array([iou_scratch(boxes[current], boxes[r]) for r in rest])
        order = rest[ious < iou_threshold]    # keep only boxes with LOW overlap

    return keep


def section_b_iou_nms():
    print("\n" + "="*65)
    print("SECTION B — IoU & NMS: Scratch vs torchvision.ops")
    print("="*65)

    # IoU verification
    rng = np.random.default_rng(SEED)
    boxes_a = rng.uniform(0, 50, size=(5, 2))
    boxes_a = np.concatenate([boxes_a, boxes_a + rng.uniform(5, 20, size=(5,2))], axis=1)
    boxes_b = rng.uniform(0, 50, size=(5, 2))
    boxes_b = np.concatenate([boxes_b, boxes_b + rng.uniform(5, 20, size=(5,2))], axis=1)

    iou_scratch_mat = iou_batch_scratch(boxes_a, boxes_b)
    iou_torch_mat = torchvision_box_iou(
        torch.tensor(boxes_a, dtype=torch.float32),
        torch.tensor(boxes_b, dtype=torch.float32)
    ).numpy()

    match_iou = np.allclose(iou_scratch_mat, iou_torch_mat, atol=1e-5)
    print(f"\n  IoU matrix shape: {iou_scratch_mat.shape} | match with torchvision: {match_iou}")
    assert match_iou

    # NMS verification
    boxes_nms = np.array([
        [10, 10, 50, 50],   # A
        [12, 12, 52, 52],   # B — heavily overlaps A
        [60, 60, 90, 90],   # C — separate
        [58, 58, 89, 89],   # D — heavily overlaps C
    ], dtype=np.float32)
    scores_nms = np.array([0.95, 0.90, 0.85, 0.30], dtype=np.float32)

    keep_scratch = nms_scratch(boxes_nms, scores_nms, iou_threshold=0.5)
    keep_torch = torchvision_nms(
        torch.tensor(boxes_nms), torch.tensor(scores_nms), iou_threshold=0.5
    ).numpy().tolist()

    match_nms = sorted(keep_scratch) == sorted(keep_torch)
    print(f"\n  NMS scratch keeps indices: {sorted(keep_scratch)}  (boxes A,C expected)")
    print(f"  NMS torchvision keeps:     {sorted(keep_torch)}")
    print(f"  Match: {match_nms}")
    assert match_nms

    print("\n  ✓ IoU and NMS verified against torchvision.ops")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION C — ANCHOR BOX GENERATION (conceptual demo)
# ═════════════════════════════════════════════════════════════════════════════

def generate_anchors(feature_size, stride, scales, aspect_ratios):
    """
    Generate anchor boxes tiled across a feature map grid.
    Returns: (feature_size*feature_size*len(scales)*len(aspect_ratios), 4) array of [x1,y1,x2,y2]
    """
    anchors = []
    for i in range(feature_size):
        for j in range(feature_size):
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride
            for scale in scales:
                for ratio in aspect_ratios:
                    w = scale * np.sqrt(ratio)
                    h = scale / np.sqrt(ratio)
                    anchors.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
    return np.array(anchors)


def section_c_anchors():
    print("\n" + "="*65)
    print("SECTION C — Anchor Box Generation (Conceptual Demo)")
    print("="*65)

    feature_size, stride = 4, 16   # 4x4 grid, 16px stride → covers 64x64 image
    scales = [16, 32]
    ratios = [0.5, 1.0, 2.0]

    anchors = generate_anchors(feature_size, stride, scales, ratios)
    print(f"\n  Feature grid: {feature_size}x{feature_size}, stride={stride}")
    print(f"  Scales: {scales}, Aspect ratios: {ratios}")
    print(f"  Total anchors: {feature_size}x{feature_size}x{len(scales)}x{len(ratios)} = {len(anchors)}")
    print(f"  Sample anchor (center cell, first scale/ratio): {anchors[len(anchors)//2].round(1)}")

    return anchors


# ═════════════════════════════════════════════════════════════════════════════
# SECTION D — YOLO-STYLE DETECTOR (fully from scratch)
# ═════════════════════════════════════════════════════════════════════════════

GRID_SIZE = 8                          # S: 8x8 grid
STRIDE = IMG_SIZE // GRID_SIZE         # 8 pixels per cell
PRED_DEPTH = 5 + NUM_CLASSES           # objectness + x,y,w,h + class scores


class YOLOMini(nn.Module):
    """
    Minimal YOLO-style single-stage detector.
    Backbone downsamples 64->8 (stride 8 total) via 3 stride-2 conv blocks,
    then a 1x1 conv head produces (5+C) channels per grid cell.

    Output: (batch, 5+C, S, S) raw logits — sigmoid applied during loss/decode.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),  # 64→32
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),  # 32→16
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), # 16→8
            nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(128, 5 + num_classes, kernel_size=1)    # per-cell prediction

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)    # (batch, 5+C, S, S) raw logits


def encode_yolo_target(boxes: torch.Tensor, labels: torch.Tensor,
                       img_size=IMG_SIZE, grid_size=GRID_SIZE, num_classes=NUM_CLASSES) -> torch.Tensor:
    """
    Encode ground-truth boxes/labels into a YOLO target tensor (5+C, S, S).

    For the cell containing each object's center:
      channel 0:     objectness = 1
      channel 1-2:   (x_offset, y_offset) relative to cell, in [0,1]
      channel 3-4:   (w, h) relative to WHOLE IMAGE, in [0,1]
      channel 5+:    one-hot class
    """
    stride = img_size / grid_size
    target = torch.zeros(5 + num_classes, grid_size, grid_size)

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box.tolist()
        cx, cy = (x1+x2)/2, (y1+y2)/2
        w, h   = (x2-x1), (y2-y1)

        col = min(int(cx // stride), grid_size - 1)
        row = min(int(cy // stride), grid_size - 1)

        x_offset = (cx - col*stride) / stride
        y_offset = (cy - row*stride) / stride
        w_norm   = w / img_size
        h_norm   = h / img_size

        target[0, row, col] = 1.0
        target[1, row, col] = x_offset
        target[2, row, col] = y_offset
        target[3, row, col] = w_norm
        target[4, row, col] = h_norm
        target[5 + int(label.item()), row, col] = 1.0

    return target


def yolo_loss(pred: torch.Tensor, target: torch.Tensor,
              lambda_coord=5.0, lambda_noobj=0.5) -> dict:
    """
    YOLO loss, computed exactly as the sum of 5 weighted MSE terms from theory.md §8.

    pred, target: (batch, 5+C, S, S)  — pred is RAW (pre-sigmoid) logits
    """
    pred_sig = torch.sigmoid(pred)    # bound all outputs to [0,1] — see explanation.md for rationale

    obj_mask   = target[:, 0:1, :, :]              # (batch,1,S,S) — 1 where object exists
    noobj_mask = 1.0 - obj_mask

    # ── Coordinate loss (x,y direct; w,h via sqrt) — only for object cells ───
    xy_loss = obj_mask * ((pred_sig[:,1:3] - target[:,1:3]) ** 2)
    # clamp before sqrt: predictions can be near 0, avoid sqrt(negative) from float error
    pred_wh_sqrt   = torch.sqrt(torch.clamp(pred_sig[:,3:5], min=1e-6))
    target_wh_sqrt = torch.sqrt(torch.clamp(target[:,3:5], min=1e-6))
    wh_loss = obj_mask * ((pred_wh_sqrt - target_wh_sqrt) ** 2)
    coord_loss = lambda_coord * (xy_loss.sum() + wh_loss.sum())

    # ── Confidence loss — object cells (target=1) and no-object cells (target=0) ─
    obj_conf_loss   = (obj_mask   * (pred_sig[:,0:1] - target[:,0:1]) ** 2).sum()
    noobj_conf_loss = lambda_noobj * (noobj_mask * (pred_sig[:,0:1] - target[:,0:1]) ** 2).sum()

    # ── Classification loss — only for object cells ──────────────────────────
    class_loss = (obj_mask * (pred_sig[:,5:] - target[:,5:]) ** 2).sum()

    total = coord_loss + obj_conf_loss + noobj_conf_loss + class_loss
    batch_size = pred.shape[0]
    return {
        "total": total / batch_size,
        "coord": coord_loss.item() / batch_size,
        "obj_conf": obj_conf_loss.item() / batch_size,
        "noobj_conf": noobj_conf_loss.item() / batch_size,
        "class": class_loss.item() / batch_size,
    }


def decode_yolo_predictions(pred: torch.Tensor, conf_threshold=0.3,
                            img_size=IMG_SIZE, grid_size=GRID_SIZE) -> tuple:
    """
    Decode a single image's raw YOLO output (5+C, S, S) into a list of
    (box[x1,y1,x2,y2], class_id, score) BEFORE NMS.
    """
    stride = img_size / grid_size
    pred_sig = torch.sigmoid(pred)

    boxes, class_ids, scores = [], [], []
    for row in range(grid_size):
        for col in range(grid_size):
            conf = pred_sig[0, row, col].item()
            if conf < conf_threshold:
                continue

            x_off, y_off = pred_sig[1, row, col].item(), pred_sig[2, row, col].item()
            w_norm, h_norm = pred_sig[3, row, col].item(), pred_sig[4, row, col].item()
            class_probs = pred_sig[5:, row, col]
            class_id = int(torch.argmax(class_probs).item())
            class_conf = float(class_probs[class_id].item())

            cx = (col + x_off) * stride
            cy = (row + y_off) * stride
            w  = w_norm * img_size
            h  = h_norm * img_size

            boxes.append([cx-w/2, cy-h/2, cx+w/2, cy+h/2])
            class_ids.append(class_id)
            scores.append(conf * class_conf)

    return np.array(boxes), np.array(class_ids), np.array(scores)


def train_yolo(train_loader, val_loader, n_epochs=40, lr=1e-3):
    print("\n" + "="*65)
    print("SECTION D — Training YOLO-mini (from scratch)")
    print("="*65)

    model = YOLOMini().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  YOLOMini parameters: {n_params:,}")
    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}  |  Stride: {STRIDE}px  |  Pred depth: {PRED_DEPTH}")

    history = {"train_loss": [], "val_loss": []}
    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()
        tl, tn = 0.0, 0
        for images, boxes_list, labels_list in train_loader:
            images = images.to(DEVICE)
            targets = torch.stack([
                encode_yolo_target(b, l) for b, l in zip(boxes_list, labels_list)
            ]).to(DEVICE)

            opt.zero_grad()
            preds = model(images)
            loss_dict = yolo_loss(preds, targets)
            loss_dict["total"].backward()
            opt.step()
            tl += loss_dict["total"].item() * len(images)
            tn += len(images)

        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for images, boxes_list, labels_list in val_loader:
                images = images.to(DEVICE)
                targets = torch.stack([
                    encode_yolo_target(b, l) for b, l in zip(boxes_list, labels_list)
                ]).to(DEVICE)
                preds = model(images)
                loss_dict = yolo_loss(preds, targets)
                vl += loss_dict["total"].item() * len(images)
                vn += len(images)

        history["train_loss"].append(tl/tn)
        history["val_loss"].append(vl/vn)

        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{n_epochs} | train_loss={tl/tn:8.3f} | val_loss={vl/vn:8.3f}")

    elapsed = time.time() - t0
    print(f"\n  ✓ YOLO-mini training complete in {elapsed:.1f}s")
    return model, history


# ═════════════════════════════════════════════════════════════════════════════
# SECTION E — FASTER R-CNN VIA TORCHVISION (custom tiny backbone)
# ═════════════════════════════════════════════════════════════════════════════

class TinyBackbone(nn.Module):
    """
    Minimal backbone for Faster R-CNN: 3 conv blocks, stride-2 each, producing
    a single feature map at 1/8 resolution. torchvision requires `.out_channels`
    to be set so the RPN/RoIHeads know the feature dimensionality.
    """
    def __init__(self, out_channels=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=2, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
        self.out_channels = out_channels    # REQUIRED attribute for torchvision FasterRCNN

    def forward(self, x):
        return self.body(x)    # single feature map, 1/8 resolution


def build_faster_rcnn(num_classes=NUM_CLASSES+1):    # +1 for background class
    """
    Build a Faster R-CNN model with a custom tiny backbone (no pretrained weights
    — none are downloadable in this environment, and none are needed since we
    train fully from scratch on our synthetic dataset).
    """
    backbone = TinyBackbone(out_channels=64)

    anchor_generator = AnchorGenerator(
        sizes=((16, 32, 48),),            # anchor scales appropriate for our small 64x64 images
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0"],               # single feature map, torchvision needs a name key
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=IMG_SIZE, max_size=IMG_SIZE,    # don't let torchvision resize our small images
    )
    return model


def train_faster_rcnn(train_loader, val_loader, n_epochs=8, lr=1e-3):
    print("\n" + "="*65)
    print("SECTION E — Training Faster R-CNN (torchvision, custom tiny backbone)")
    print("="*65)

    model = build_faster_rcnn().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Faster R-CNN (tiny backbone) parameters: {n_params:,}")

    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    history = {"train_loss": []}
    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()
        tl, tn = 0.0, 0
        for images, boxes_list, labels_list in train_loader:
            images = [img.to(DEVICE) for img in images]
            # torchvision convention: labels are 1-indexed (0 reserved for background)
            targets = [{"boxes": b.to(DEVICE), "labels": (l+1).to(DEVICE)}
                      for b, l in zip(boxes_list, labels_list)]

            opt.zero_grad()
            loss_dict = model(images, targets)            # returns dict of losses in train mode
            loss = sum(loss_dict.values())
            loss.backward()
            opt.step()

            tl += loss.item() * len(images)
            tn += len(images)

        history["train_loss"].append(tl/tn)
        if (epoch+1) % 2 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:2d}/{n_epochs} | train_loss={tl/tn:8.3f}")

    elapsed = time.time() - t0
    print(f"\n  ✓ Faster R-CNN training complete in {elapsed:.1f}s")
    return model, history


@torch.no_grad()
def faster_rcnn_predict(model, image: torch.Tensor, score_threshold=0.3):
    """Run inference, return boxes/labels/scores above threshold (already NMS'd internally)."""
    model.eval()
    output = model([image.to(DEVICE)])[0]
    keep = output["scores"] > score_threshold
    boxes = output["boxes"][keep].cpu().numpy()
    labels = output["labels"][keep].cpu().numpy() - 1    # convert back to 0-indexed
    scores = output["scores"][keep].cpu().numpy()
    return boxes, labels, scores


# ═════════════════════════════════════════════════════════════════════════════
# SECTION F — EVALUATION: SIMPLE PRECISION/RECALL AT IoU=0.5
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_detector(pred_boxes_per_image, pred_labels_per_image,
                      gt_boxes_per_image, gt_labels_per_image, iou_threshold=0.5):
    """
    Simple detection evaluation: for each image, greedily match predictions to
    ground truth (by descending confidence, already sorted by caller), count
    TP/FP/FN. Returns overall precision and recall across the dataset.
    """
    total_tp, total_fp, total_fn = 0, 0, 0

    for pred_boxes, pred_labels, gt_boxes, gt_labels in zip(
            pred_boxes_per_image, pred_labels_per_image,
            gt_boxes_per_image, gt_labels_per_image):

        matched_gt = set()
        for pbox, plabel in zip(pred_boxes, pred_labels):
            best_iou, best_idx = 0.0, -1
            for idx, (gbox, glabel) in enumerate(zip(gt_boxes, gt_labels)):
                if idx in matched_gt or glabel != plabel:
                    continue
                iou = iou_scratch(pbox, gbox)
                if iou > best_iou:
                    best_iou, best_idx = iou, idx

            if best_iou >= iou_threshold:
                total_tp += 1
                matched_gt.add(best_idx)
            else:
                total_fp += 1

        total_fn += len(gt_boxes) - len(matched_gt)

    precision = total_tp / (total_tp + total_fp) if (total_tp+total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp+total_fn) > 0 else 0.0
    return {"precision": precision, "recall": recall, "tp": total_tp, "fp": total_fp, "fn": total_fn}


def evaluate_yolo(model, val_ds, conf_threshold=0.3, nms_threshold=0.4):
    model.eval()
    all_pred_boxes, all_pred_labels, all_gt_boxes, all_gt_labels = [], [], [], []

    with torch.no_grad():
        for i in range(len(val_ds)):
            img, gt_boxes, gt_labels = val_ds[i]
            pred = model(img.unsqueeze(0).to(DEVICE))[0].cpu()
            boxes, class_ids, scores = decode_yolo_predictions(pred, conf_threshold)

            if len(boxes) > 0:
                keep = nms_scratch(boxes, scores, nms_threshold)
                boxes, class_ids, scores = boxes[keep], class_ids[keep], scores[keep]
                order = np.argsort(scores)[::-1]
                boxes, class_ids = boxes[order], class_ids[order]

            all_pred_boxes.append(boxes)
            all_pred_labels.append(class_ids)
            all_gt_boxes.append(gt_boxes.numpy())
            all_gt_labels.append(gt_labels.numpy())

    return evaluate_detector(all_pred_boxes, all_pred_labels, all_gt_boxes, all_gt_labels)


def evaluate_faster_rcnn(model, val_ds, score_threshold=0.5):
    all_pred_boxes, all_pred_labels, all_gt_boxes, all_gt_labels = [], [], [], []

    for i in range(len(val_ds)):
        img, gt_boxes, gt_labels = val_ds[i]
        boxes, labels, scores = faster_rcnn_predict(model, img, score_threshold)
        order = np.argsort(scores)[::-1] if len(scores) > 0 else []
        boxes, labels = (boxes[order], labels[order]) if len(order) > 0 else (boxes, labels)

        all_pred_boxes.append(boxes)
        all_pred_labels.append(labels)
        all_gt_boxes.append(gt_boxes.numpy())
        all_gt_labels.append(gt_labels.numpy())

    return evaluate_detector(all_pred_boxes, all_pred_labels, all_gt_boxes, all_gt_labels)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION G — VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

CLASS_COLORS = ["#e74c3c", "#3498db", "#2ecc71"]


def _draw_boxes_on_ax(ax, img_chw, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, title):
    ax.imshow(img_chw.transpose(1, 2, 0))
    for box, label in zip(gt_boxes, gt_labels):
        x1,y1,x2,y2 = box
        rect = patches.Rectangle((x1,y1), x2-x1, y2-y1, linewidth=2,
                                 edgecolor="white", facecolor="none", linestyle="--")
        ax.add_patch(rect)
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        x1,y1,x2,y2 = box
        color = CLASS_COLORS[int(label) % len(CLASS_COLORS)]
        rect = patches.Rectangle((x1,y1), x2-x1, y2-y1, linewidth=2,
                                 edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, max(y1-2,0), f"{CLASS_NAMES[int(label)]} {score:.2f}",
               fontsize=6.5, color=color, fontweight="bold",
               bbox=dict(facecolor="black", alpha=0.5, pad=0.5))
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.axis("off")


def build_figures(yolo_model, yolo_hist, frcnn_model, frcnn_hist, val_ds, yolo_metrics, frcnn_metrics):
    # ── Figure 1: training curves ─────────────────────────────────────────────
    fig1, axes1 = plt.subplots(1, 2, figsize=(13, 4.5))
    fig1.suptitle("Phase 2 — Topic 3: Training Curves", fontsize=13, fontweight="bold")

    ep_y = range(1, len(yolo_hist["train_loss"])+1)
    axes1[0].plot(ep_y, yolo_hist["train_loss"], color="#e74c3c", lw=2, label="Train")
    axes1[0].plot(ep_y, yolo_hist["val_loss"],   color="#3498db", lw=2, label="Val")
    axes1[0].set_title("YOLO-mini Loss", fontweight="bold")
    axes1[0].set_xlabel("Epoch"); axes1[0].set_ylabel("YOLO Loss")
    axes1[0].legend(fontsize=9); axes1[0].grid(True, alpha=0.3)

    ep_f = range(1, len(frcnn_hist["train_loss"])+1)
    axes1[1].plot(ep_f, frcnn_hist["train_loss"], color="#27ae60", lw=2, label="Train")
    axes1[1].set_title("Faster R-CNN Loss", fontweight="bold")
    axes1[1].set_xlabel("Epoch"); axes1[1].set_ylabel("Total Loss")
    axes1[1].legend(fontsize=9); axes1[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path1 = os.path.join(RESULTS, "03_training_curves.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"\n  [VIZ] Figure 1 saved → {path1}")
    plt.close(fig1)

    # ── Figure 2: predictions vs ground truth, both detectors ──────────────────
    n_show = 4
    fig2, axes2 = plt.subplots(2, n_show, figsize=(4*n_show, 8))
    fig2.suptitle(
        f"Predictions vs Ground Truth  (dashed white = GT, solid color = prediction)\n"
        f"YOLO-mini: P={yolo_metrics['precision']:.2f} R={yolo_metrics['recall']:.2f}   |   "
        f"Faster R-CNN: P={frcnn_metrics['precision']:.2f} R={frcnn_metrics['recall']:.2f}",
        fontsize=11, fontweight="bold")

    yolo_model.eval(); frcnn_model.eval()
    for col in range(n_show):
        img, gt_boxes, gt_labels = val_ds[col]
        img_np = img.numpy()

        # YOLO prediction
        with torch.no_grad():
            pred = yolo_model(img.unsqueeze(0).to(DEVICE))[0].cpu()
        boxes, class_ids, scores = decode_yolo_predictions(pred, conf_threshold=0.3)
        if len(boxes) > 0:
            keep = nms_scratch(boxes, scores, 0.4)
            boxes, class_ids, scores = boxes[keep], class_ids[keep], scores[keep]
        _draw_boxes_on_ax(axes2[0, col], img_np, gt_boxes.numpy(), gt_labels.numpy(),
                          boxes, class_ids, scores, f"YOLO-mini — image {col}")

        # Faster R-CNN prediction
        fboxes, flabels, fscores = faster_rcnn_predict(frcnn_model, img, score_threshold=0.5)
        _draw_boxes_on_ax(axes2[1, col], img_np, gt_boxes.numpy(), gt_labels.numpy(),
                          fboxes, flabels, fscores, f"Faster R-CNN — image {col}")

    plt.tight_layout()
    path2 = os.path.join(RESULTS, "03_predictions_comparison.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"  [VIZ] Figure 2 saved → {path2}")
    plt.close(fig2)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "▓"*65)
    print("  Phase 2 — Topic 3: Object Detection — Faster R-CNN & YOLO")
    print("▓"*65)

    train_loader, val_loader, train_ds, val_ds = build_detection_data(
        n_train=400, n_val=80, batch_size=16, max_objects=2)

    section_b_iou_nms()
    section_c_anchors()

    yolo_model, yolo_hist = train_yolo(train_loader, val_loader, n_epochs=40)
    frcnn_model, frcnn_hist = train_faster_rcnn(train_loader, val_loader, n_epochs=8)

    print("\n" + "="*65)
    print("SECTION F — Evaluation (Precision/Recall @ IoU=0.5)")
    print("="*65)
    yolo_metrics = evaluate_yolo(yolo_model, val_ds)
    frcnn_metrics = evaluate_faster_rcnn(frcnn_model, val_ds)
    print(f"\n  YOLO-mini:     Precision={yolo_metrics['precision']:.3f}  Recall={yolo_metrics['recall']:.3f}  "
          f"(TP={yolo_metrics['tp']}, FP={yolo_metrics['fp']}, FN={yolo_metrics['fn']})")
    print(f"  Faster R-CNN:  Precision={frcnn_metrics['precision']:.3f}  Recall={frcnn_metrics['recall']:.3f}  "
          f"(TP={frcnn_metrics['tp']}, FP={frcnn_metrics['fp']}, FN={frcnn_metrics['fn']})")

    build_figures(yolo_model, yolo_hist, frcnn_model, frcnn_hist, val_ds, yolo_metrics, frcnn_metrics)

    print("\n" + "▓"*65)
    print("  ✓ Topic 3 complete.")
    print("▓"*65 + "\n")


if __name__ == "__main__":
    main()
