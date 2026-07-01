"""
Phase 2 — Topic 2: CNN Architectures — LeNet, AlexNet, VGGNet, ResNet, DenseNet, GoogLeNet
=============================================================================================
Repository : deep-learning-mastery/phase-2-cnns/02-architectures-lenet-to-densenet/
File       : implementation.py

All architectures are SCALED DOWN versions of the originals (fewer channels/layers)
so that all 6 can be trained from scratch on CPU in reasonable time, while preserving
every architecturally-defining feature (residual connections, dense concatenation,
inception branches, etc.) described in theory.md.

Sections:
  A │ Synthetic "Shapes" dataset generator (5-class image classification, 32x32 RGB)
  B │ LeNet-5 (faithful to original, scaled for RGB 32x32 input)
  C │ AlexNet-mini (ReLU + Dropout, scaled channel counts)
  D │ VGG-mini (uniform 3x3 convs, VGG-style blocks)
  E │ GoogLeNet-mini (Inception module with parallel branches + 1x1 bottlenecks)
  F │ ResNet-mini (BasicBlock with residual/skip connections)
  G │ DenseNet-mini (DenseBlock with channel-concatenation)
  H │ Unified training loop + comparison across all 6 architectures
  I │ Visualization dashboard
"""

import os, time, warnings
import numpy as np
import matplotlib, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = "results"; os.makedirs(RESULTS, exist_ok=True)
np.random.seed(SEED); torch.manual_seed(SEED)
print(f"[CONFIG] Device: {DEVICE}  |  PyTorch: {torch.__version__}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION A — SYNTHETIC "SHAPES" DATASET
# 5 classes: circle, square, triangle, star, diamond — 32x32 RGB
# ═════════════════════════════════════════════════════════════════════════════

CLASS_NAMES = ["circle", "square", "triangle", "star", "diamond"]


def _draw_shape(cls: int, size: int = 32, rng: np.random.Generator = None) -> np.ndarray:
    """Draw one synthetic shape image with random color/position/rotation/noise."""
    if rng is None:
        rng = np.random.default_rng()

    img = Image.new("RGB", (size, size), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    color = tuple(int(c) for c in rng.integers(100, 256, size=3))
    cx, cy = size // 2 + rng.integers(-3, 4), size // 2 + rng.integers(-3, 4)
    r = rng.integers(8, 13)

    if cls == 0:    # circle
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color)
    elif cls == 1:  # square
        draw.rectangle([cx-r, cy-r, cx+r, cy+r], fill=color)
    elif cls == 2:  # triangle
        pts = [(cx, cy-r), (cx-r, cy+r), (cx+r, cy+r)]
        draw.polygon(pts, fill=color)
    elif cls == 3:  # 5-pointed star
        pts = []
        for i in range(10):
            ang = np.pi/2 + i * np.pi/5
            rad = r if i % 2 == 0 else r*0.45
            pts.append((cx + rad*np.cos(ang), cy - rad*np.sin(ang)))
        draw.polygon(pts, fill=color)
    elif cls == 4:  # diamond (rotated square)
        pts = [(cx, cy-r), (cx+r, cy), (cx, cy+r), (cx-r, cy)]
        draw.polygon(pts, fill=color)

    arr = np.array(img, dtype=np.float32) / 255.0    # (H,W,3) in [0,1]

    # Add Gaussian noise for realism / to prevent trivial pixel-perfect classification
    noise = rng.normal(0, 0.04, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 1)

    return arr.transpose(2, 0, 1)    # → (3, H, W) channel-first for PyTorch


def generate_shapes_dataset(n_per_class: int = 300, size: int = 32, seed: int = SEED):
    """Generate a balanced synthetic dataset across all 5 shape classes."""
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for cls in range(len(CLASS_NAMES)):
        for _ in range(n_per_class):
            images.append(_draw_shape(cls, size, rng))
            labels.append(cls)
    images = np.stack(images)            # (N, 3, H, W)
    labels = np.array(labels, dtype=np.int64)

    # Shuffle
    perm = rng.permutation(len(images))
    return images[perm], labels[perm]


def build_dataloaders(n_per_class=300, batch_size=32, val_frac=0.2):
    print("\n" + "="*65)
    print("SECTION A — Synthetic Shapes Dataset")
    print("="*65)

    X, y = generate_shapes_dataset(n_per_class=n_per_class)
    n_val = int(len(X) * val_frac)
    X_va, y_va = X[:n_val], y[:n_val]
    X_tr, y_tr = X[n_val:], y[n_val:]

    print(f"\n  Classes: {CLASS_NAMES}")
    print(f"  Total images: {len(X)}  |  Train: {len(X_tr)}  |  Val: {len(X_va)}")
    print(f"  Image shape: {X.shape[1:]} (channels, H, W)")

    train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    val_ds   = TensorDataset(torch.tensor(X_va), torch.tensor(y_va))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_va, y_va


# ═════════════════════════════════════════════════════════════════════════════
# SECTION B — LENET-5 (adapted for 3-channel 32x32 input, 5 classes)
# ═════════════════════════════════════════════════════════════════════════════

class LeNet5(nn.Module):
    """
    Faithful to the original 1998 architecture: Conv-Pool-Conv-Pool-FC-FC-FC,
    using Tanh activations and Average Pooling as in the original paper.
    Adapted for 3-channel 32x32 input and 5 output classes.
    """
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),    nn.Tanh(),   # 32→28
            nn.AvgPool2d(2, 2),                              # 28→14
            nn.Conv2d(6, 16, kernel_size=5),   nn.Tanh(),    # 14→10
            nn.AvgPool2d(2, 2),                              # 10→5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*5*5, 120), nn.Tanh(),
            nn.Linear(120, 84),     nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION C — ALEXNET-MINI (scaled for 32x32 input)
# ═════════════════════════════════════════════════════════════════════════════

class AlexNetMini(nn.Module):
    """
    Scaled-down AlexNet preserving its defining features: ReLU activations,
    Dropout in FC layers, and a deep stack of convolutions. Original AlexNet
    used 224x224 input with 11x11 first kernel; here we adapt kernel sizes
    and strides for 32x32 input while keeping the same LAYER COUNT and
    relative channel-width progression.
    """
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2), nn.ReLU(inplace=True),  # 32x32
            nn.MaxPool2d(2, 2),                                                              # 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                                              # 8x8
            nn.Conv2d(64, 96, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(96, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                                              # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*4*4, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256),     nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION D — VGG-MINI (uniform 3x3 convs, VGG-style blocks)
# ═════════════════════════════════════════════════════════════════════════════

class VGGMini(nn.Module):
    """
    Scaled-down VGG: ONLY 3x3 convolutions (the defining VGG characteristic),
    organized into [Conv-Conv-Pool] blocks with doubling channel counts,
    same structural pattern as VGG-16 but with fewer blocks/channels for speed.
    """
    def __init__(self, num_classes=5):
        super().__init__()
        def block(c_in, c_out, n_convs=2):
            layers = []
            for i in range(n_convs):
                layers += [nn.Conv2d(c_in if i==0 else c_out, c_out,
                                     kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            layers.append(nn.MaxPool2d(2, 2))
            return layers

        self.features = nn.Sequential(
            *block(3,  32, 2),     # 32→16
            *block(32, 64, 2),     # 16→8
            *block(64, 128, 2),    # 8→4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION E — GOOGLENET-MINI (Inception module with parallel branches)
# ═════════════════════════════════════════════════════════════════════════════

class InceptionBlock(nn.Module):
    """
    Faithful Inception module: FOUR parallel branches (1x1, 3x3-reduced,
    5x5-reduced, pool-projection) concatenated channel-wise. Uses 1x1
    bottleneck convolutions before the expensive 3x3/5x5 convs exactly as
    in the original GoogLeNet design.
    """
    def __init__(self, c_in, c1x1, c3x3_reduce, c3x3, c5x5_reduce, c5x5, c_pool):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(c_in, c1x1, kernel_size=1), nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(
            nn.Conv2d(c_in, c3x3_reduce, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(c3x3_reduce, c3x3, kernel_size=3, padding=1), nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(
            nn.Conv2d(c_in, c5x5_reduce, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(c5x5_reduce, c5x5, kernel_size=5, padding=2), nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(c_in, c_pool, kernel_size=1), nn.ReLU(inplace=True))

    def forward(self, x):
        b1, b2, b3, b4 = self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)    # concatenate along channel dim


class GoogLeNetMini(nn.Module):
    """Scaled-down GoogLeNet: stem conv + 2 Inception blocks + GAP + FC."""
    def __init__(self, num_classes=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),    # 32→16
        )
        # InceptionBlock(in, 1x1, 3x3reduce, 3x3, 5x5reduce, 5x5, pool) → out=1x1+3x3+5x5+pool
        self.inception1 = InceptionBlock(32,  16, 16, 24, 8, 8, 8)    # out = 16+24+8+8 = 56
        self.pool1 = nn.MaxPool2d(2, 2)    # 16→8
        self.inception2 = InceptionBlock(56,  32, 24, 32, 8, 16, 16)  # out = 32+32+16+16 = 96
        self.gap = nn.AdaptiveAvgPool2d(1)    # global average pool → 1x1
        self.classifier = nn.Linear(96, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception1(x)
        x = self.pool1(x)
        x = self.inception2(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION F — RESNET-MINI (residual/skip connections)
# ═════════════════════════════════════════════════════════════════════════════

class BasicBlock(nn.Module):
    """
    Faithful ResNet BasicBlock: two 3x3 convs with BatchNorm, plus an
    identity (or projection, if shape changes) skip connection added BEFORE
    the final ReLU — exactly the y=F(x)+x structure from theory.md.
    """
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(c_out)
        self.relu  = nn.ReLU(inplace=True)

        # Projection shortcut needed when shape changes (stride>1 or channel mismatch)
        self.shortcut = nn.Sequential()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out))

    def forward(self, x):
        identity = self.shortcut(x)             # identity OR projected shortcut
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity                       # the residual addition: F(x) + x
        return self.relu(out)


class ResNetMini(nn.Module):
    """Scaled-down ResNet: stem + 3 stages of BasicBlocks + GAP + FC."""
    def __init__(self, num_classes=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.stage1 = nn.Sequential(BasicBlock(32, 32), BasicBlock(32, 32))
        self.stage2 = nn.Sequential(BasicBlock(32, 64, stride=2), BasicBlock(64, 64))
        self.stage3 = nn.Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION G — DENSENET-MINI (channel-concatenation dense blocks)
# ═════════════════════════════════════════════════════════════════════════════

class DenseLayer(nn.Module):
    """
    Single layer within a DenseBlock: BN-ReLU-Conv producing `growth_rate`
    new channels. The CALLER (DenseBlock) handles concatenating this
    layer's output with all previous layers' outputs.
    """
    def __init__(self, c_in, growth_rate):
        super().__init__()
        self.bn   = nn.BatchNorm2d(c_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(c_in, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return self.conv(self.relu(self.bn(x)))


class DenseBlock(nn.Module):
    """
    Stacks `n_layers` DenseLayers. Each layer's input is the CONCATENATION
    of ALL previous layers' outputs (including the original block input) —
    this is the defining xₗ=Hₗ([x₀,...,x_{l-1}]) DenseNet equation.
    """
    def __init__(self, c_in, growth_rate, n_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(DenseLayer(c_in + i*growth_rate, growth_rate))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            concat_input = torch.cat(features, dim=1)    # concatenate ALL previous outputs
            new_feat = layer(concat_input)
            features.append(new_feat)
        return torch.cat(features, dim=1)    # final output: all features concatenated


class TransitionLayer(nn.Module):
    """1x1 conv (halve channels) + AvgPool (halve spatial dims) between dense blocks."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.bn   = nn.BatchNorm2d(c_in)
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        return self.pool(self.conv(torch.relu(self.bn(x))))


class DenseNetMini(nn.Module):
    """Scaled-down DenseNet: stem + 2 DenseBlocks + 1 Transition + GAP + FC."""
    def __init__(self, num_classes=5, growth_rate=12):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1, bias=False), nn.BatchNorm2d(24), nn.ReLU(inplace=True))

        self.dense1 = DenseBlock(24, growth_rate, n_layers=4)
        c_after_d1 = 24 + 4*growth_rate                      # 24+48=72
        self.trans1 = TransitionLayer(c_after_d1, c_after_d1 // 2)   # → 36

        self.dense2 = DenseBlock(c_after_d1 // 2, growth_rate, n_layers=4)
        c_after_d2 = (c_after_d1 // 2) + 4*growth_rate         # 36+48=84

        self.bn_final = nn.BatchNorm2d(c_after_d2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_after_d2, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = torch.relu(self.bn_final(x))
        x = self.gap(x).flatten(1)
        return self.classifier(x)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION H — UNIFIED TRAINING & COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_and_evaluate(name, model, train_loader, val_loader, n_epochs=20, lr=1e-3):
    model = model.to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=lr)
    crit  = nn.CrossEntropyLoss()

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for Xb, Yb in train_loader:
            Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
            opt.zero_grad()
            logits = model(Xb)
            loss = crit(logits, Yb)
            loss.backward()
            opt.step()
            tl += loss.item()*len(Xb)
            tc += (logits.argmax(1) == Yb).sum().item()
            tn += len(Xb)

        model.eval()
        vl, vc, vn = 0.0, 0, 0
        with torch.no_grad():
            for Xb, Yb in val_loader:
                Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
                logits = model(Xb)
                loss = crit(logits, Yb)
                vl += loss.item()*len(Xb)
                vc += (logits.argmax(1) == Yb).sum().item()
                vn += len(Xb)

        hist["train_loss"].append(tl/tn); hist["val_loss"].append(vl/vn)
        hist["train_acc"].append(tc/tn);  hist["val_acc"].append(vc/vn)

    elapsed = time.time() - t0
    params = count_params(model)
    print(f"  {name:14s} | params={params:>8,} | time={elapsed:5.1f}s | "
          f"train_acc={hist['train_acc'][-1]*100:5.1f}% | val_acc={hist['val_acc'][-1]*100:5.1f}%")

    return {"history": hist, "params": params, "time": elapsed, "model": model}


def run_architecture_comparison(train_loader, val_loader, n_epochs=20):
    print("\n" + "="*65)
    print("SECTION H — Architecture Comparison Training")
    print("="*65)
    print(f"\n  Training all 6 architectures for {n_epochs} epochs each...\n")
    print(f"  {'Architecture':14s} | {'Params':>8} | {'Time':>6} | {'Train Acc':>9} | {'Val Acc':>8}")
    print("  " + "─"*62)

    archs = {
        "LeNet-5":     LeNet5(),
        "AlexNet-mini": AlexNetMini(),
        "VGG-mini":     VGGMini(),
        "GoogLeNet-mini": GoogLeNetMini(),
        "ResNet-mini":  ResNetMini(),
        "DenseNet-mini": DenseNetMini(),
    }

    results = {}
    for name, model in archs.items():
        torch.manual_seed(SEED)
        results[name] = train_and_evaluate(name, model, train_loader, val_loader, n_epochs=n_epochs)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# SECTION I — VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def build_figures(results, X_va, y_va):
    colors = {"LeNet-5":"#95a5a6", "AlexNet-mini":"#e67e22", "VGG-mini":"#3498db",
              "GoogLeNet-mini":"#9b59b6", "ResNet-mini":"#e74c3c", "DenseNet-mini":"#27ae60"}

    # ── Figure 1: Sample dataset images ───────────────────────────────────────
    fig0, axes0 = plt.subplots(2, 5, figsize=(13, 5.5))
    fig0.suptitle("Synthetic Shapes Dataset — Sample Images", fontsize=13, fontweight="bold")
    shown = {c: 0 for c in range(5)}
    idx = 0
    for cls in range(5):
        for col in range(2):
            ax = axes0[col, cls]
            # find a sample of this class
            for j in range(len(X_va)):
                if y_va[j] == cls and shown[cls] <= col:
                    img = X_va[j].transpose(1,2,0)
                    ax.imshow(img)
                    ax.set_title(CLASS_NAMES[cls], fontsize=10, fontweight="bold")
                    ax.axis("off")
                    shown[cls] += 1
                    break
    plt.tight_layout()
    path0 = os.path.join(RESULTS, "02_dataset_samples.png")
    plt.savefig(path0, dpi=150, bbox_inches="tight")
    plt.close(fig0)
    print(f"\n  [VIZ] Dataset samples saved → {path0}")

    # ── Figure 2: Training curves + comparison bars ────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Phase 2 — Topic 2: CNN Architecture Comparison", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.32)
    a = [fig.add_subplot(gs[r,c]) for r in range(2) for c in range(3)]

    # Panel 1: Val loss curves
    for name, data in results.items():
        ep = range(1, len(data["history"]["val_loss"])+1)
        a[0].plot(ep, data["history"]["val_loss"], color=colors[name], lw=2, label=name)
    a[0].set_title("Validation Loss", fontweight="bold", fontsize=10)
    a[0].set_xlabel("Epoch"); a[0].set_ylabel("CE Loss")
    a[0].legend(fontsize=7); a[0].grid(True, alpha=0.3)

    # Panel 2: Val accuracy curves
    for name, data in results.items():
        ep = range(1, len(data["history"]["val_acc"])+1)
        a[1].plot(ep, [v*100 for v in data["history"]["val_acc"]], color=colors[name], lw=2, label=name)
    a[1].set_title("Validation Accuracy", fontweight="bold", fontsize=10)
    a[1].set_xlabel("Epoch"); a[1].set_ylabel("Accuracy (%)")
    a[1].legend(fontsize=7); a[1].grid(True, alpha=0.3)

    # Panel 3: Final accuracy bar
    names = list(results.keys())
    finals = [results[n]["history"]["val_acc"][-1]*100 for n in names]
    bars = a[2].bar(range(len(names)), finals, color=[colors[n] for n in names])
    a[2].set_xticks(range(len(names))); a[2].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    a[2].set_title("Final Validation Accuracy", fontweight="bold", fontsize=10)
    a[2].set_ylabel("Accuracy (%)"); a[2].set_ylim(0, 105)
    for bar, v in zip(bars, finals):
        a[2].text(bar.get_x()+bar.get_width()/2, v+1, f"{v:.0f}%", ha="center", fontsize=8)
    a[2].grid(True, axis="y", alpha=0.3)

    # Panel 4: Parameter count bar (log scale)
    p_counts = [results[n]["params"] for n in names]
    a[3].bar(range(len(names)), p_counts, color=[colors[n] for n in names])
    a[3].set_yscale("log")
    a[3].set_xticks(range(len(names))); a[3].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    a[3].set_title("Parameter Count (log scale)", fontweight="bold", fontsize=10)
    a[3].set_ylabel("Parameters"); a[3].grid(True, axis="y", alpha=0.3, which="both")

    # Panel 5: Training time bar
    times = [results[n]["time"] for n in names]
    a[4].bar(range(len(names)), times, color=[colors[n] for n in names])
    a[4].set_xticks(range(len(names))); a[4].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    a[4].set_title("Training Time (seconds)", fontweight="bold", fontsize=10)
    a[4].set_ylabel("Seconds"); a[4].grid(True, axis="y", alpha=0.3)

    # Panel 6: Accuracy per million params (efficiency)
    efficiency = [finals[i] / (p_counts[i]/1e6) for i in range(len(names))]
    a[5].bar(range(len(names)), efficiency, color=[colors[n] for n in names])
    a[5].set_xticks(range(len(names))); a[5].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    a[5].set_title("Accuracy per Million Params (Efficiency)", fontweight="bold", fontsize=10)
    a[5].set_ylabel("Val Acc% / 1M params"); a[5].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS, "02_architecture_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [VIZ] Comparison dashboard saved → {path}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "▓"*65)
    print("  Phase 2 — Topic 2: CNN Architectures (LeNet → DenseNet)")
    print("▓"*65)

    train_loader, val_loader, X_va, y_va = build_dataloaders(n_per_class=200, batch_size=64)
    results = run_architecture_comparison(train_loader, val_loader, n_epochs=15)
    build_figures(results, X_va, y_va)

    print("\n" + "▓"*65)
    print("  ✓ Topic 2 complete. All 6 architectures trained and compared.")
    print("▓"*65 + "\n")


if __name__ == "__main__":
    main()

