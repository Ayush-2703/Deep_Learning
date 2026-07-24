"""
Topic: Vision Transformer (ViT)
=========================================================
Repository : deep-learning/phase-4-attention-transformers/05-extra-vision-transformer/
File       : implementation.py

Task: same 5-class synthetic shapes classification used for Phase 2 Topic 2's
CNN architecture comparison (circle/square/triangle/star/diamond, 32x32 RGB),
enabling a DIRECT comparison between attention-based and convolution-based
image classifiers.

Sections:
  A | Synthetic shapes dataset (5-class, 32x32 RGB)
  B | Patch embedding -- verified EXACTLY equivalent to a strided Conv2d
  C | Full ViT model assembly ([CLS] + patches + position embed + encoder + head)
  D | Train ViT, compare vs Phase 2 Topic 2's CNN architecture results
  E | Attention visualization: which patches does [CLS] attend to?
  F | Data efficiency: ViT vs simple CNN across increasing training set sizes
  G | Visualization dashboard
"""

import os, time, warnings
import numpy as np
import matplotlib, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = "results"; os.makedirs(RESULTS, exist_ok=True)
np.random.seed(SEED); torch.manual_seed(SEED)
print(f"[CONFIG] Device: {DEVICE}  |  PyTorch: {torch.__version__}")

CLASS_NAMES = ["circle", "square", "triangle", "star", "diamond"]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 32
PATCH_SIZE = 8
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2   # 16


# =============================================================================
# SECTION A -- SYNTHETIC SHAPES DATASET (consistent with Phase 2 Topic 2)
# =============================================================================

def _draw_shape(cls, size, rng):
    img = Image.new("RGB", (size, size), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    color = tuple(int(c) for c in rng.integers(100, 256, size=3))
    cx, cy = size//2 + rng.integers(-3,4), size//2 + rng.integers(-3,4)
    r = rng.integers(8, 13)

    if cls == 0:   draw.ellipse([cx-r,cy-r,cx+r,cy+r], fill=color)
    elif cls == 1: draw.rectangle([cx-r,cy-r,cx+r,cy+r], fill=color)
    elif cls == 2: draw.polygon([(cx,cy-r),(cx-r,cy+r),(cx+r,cy+r)], fill=color)
    elif cls == 3:
        pts = []
        for i in range(10):
            ang = np.pi/2 + i*np.pi/5
            rad = r if i%2==0 else r*0.45
            pts.append((cx+rad*np.cos(ang), cy-rad*np.sin(ang)))
        draw.polygon(pts, fill=color)
    else: draw.polygon([(cx,cy-r),(cx+r,cy),(cx,cy+r),(cx-r,cy)], fill=color)

    arr = np.array(img, dtype=np.float32) / 255.0
    noise = rng.normal(0, 0.04, arr.shape).astype(np.float32)
    arr = np.clip(arr+noise, 0, 1)
    return arr.transpose(2,0,1)


def generate_shapes_dataset(n_per_class, seed):
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for cls in range(NUM_CLASSES):
        for _ in range(n_per_class):
            images.append(_draw_shape(cls, IMG_SIZE, rng))
            labels.append(cls)
    images = np.stack(images); labels = np.array(labels, dtype=np.int64)
    perm = rng.permutation(len(images))
    return images[perm], labels[perm]


def section_a_dataset_demo():
    print("\n" + "="*65)
    print("SECTION A -- Synthetic Shapes Dataset (5-class, 32x32 RGB)")
    print("="*65)

    X, y = generate_shapes_dataset(4, SEED)
    print(f"\n  Classes: {CLASS_NAMES}")
    print(f"  Image shape: {X.shape[1:]} | Patch size: {PATCH_SIZE}x{PATCH_SIZE} | "
          f"Patches/image: {NUM_PATCHES}")


# =============================================================================
# SECTION B -- PATCH EMBEDDING: VERIFIED EQUIVALENT TO STRIDED CONV2D
# =============================================================================

class PatchEmbedManual(nn.Module):
    """Flatten each non-overlapping patch, apply a SHARED linear projection."""
    def __init__(self, img_size, patch_size, in_channels, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches_per_side = img_size // patch_size
        self.proj = nn.Linear(patch_size*patch_size*in_channels, d_model)

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        n = self.n_patches_per_side
        # unfold into non-overlapping PxP patches: (B, C, n, P, n, P) -> (B, n*n, C*P*P)
        x = x.reshape(B, C, n, P, n, P).permute(0, 2, 4, 1, 3, 5).reshape(B, n*n, C*P*P)
        return self.proj(x)    # (B, n_patches, d_model)


class PatchEmbedConv(nn.Module):
    """Equivalent implementation via a strided Conv2d (theory.md sec 3)."""
    def __init__(self, img_size, patch_size, in_channels, d_model):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B = x.shape[0]
        out = self.conv(x)                  # (B, d_model, n, n)
        return out.flatten(2).transpose(1, 2)   # (B, n*n, d_model)


def section_b_patch_embed_equivalence():
    print("\n" + "="*65)
    print("SECTION B -- Patch Embedding: Manual vs Strided-Conv2d Equivalence")
    print("="*65)

    d_model = 32
    torch.manual_seed(SEED)
    manual = PatchEmbedManual(IMG_SIZE, PATCH_SIZE, 3, d_model)
    conv = PatchEmbedConv(IMG_SIZE, PATCH_SIZE, 3, d_model)

    # Inject IDENTICAL weights: Conv2d's weight (d_model,C,P,P) reshaped
    # matches Linear's weight (d_model, C*P*P) under the SAME flatten order
    with torch.no_grad():
        conv_weight_flat = conv.conv.weight.reshape(d_model, -1)   # (d_model, C*P*P)
        manual.proj.weight[:] = conv_weight_flat
        manual.proj.bias[:] = conv.conv.bias

    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    out_manual = manual(x)
    out_conv = conv(x)

    match = torch.allclose(out_manual, out_conv, atol=1e-5)
    print(f"\n  Input shape: {tuple(x.shape)}")
    print(f"  Manual (flatten+Linear) output: {tuple(out_manual.shape)}")
    print(f"  Conv2d (stride=kernel=P) output: {tuple(out_conv.shape)}")
    print(f"  Exact match: {match}")
    assert match
    print("\n  OK: Patch embedding verified EXACTLY equivalent to a strided convolution")
    print("      (directly extending Phase 2 Topic 1's im2col conv-as-matmul insight)")


# =============================================================================
# SECTION C -- FULL VIT MODEL ASSEMBLY
# =============================================================================

class ViTEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.norm1(x)
        attn_out, attn_w = self.self_attn(h, h, h, need_weights=True, average_attn_weights=True)
        x = x + attn_out
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x, attn_w


class VisionTransformer(nn.Module):
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_channels=3,
                d_model=64, num_heads=4, d_ff=128, num_layers=4, num_classes=NUM_CLASSES):
        super().__init__()
        self.patch_embed = PatchEmbedConv(img_size, patch_size, in_channels, d_model)
        n_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches+1, d_model) * 0.02)
        self.layers = nn.ModuleList([ViTEncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B = x.shape[0]
        patches = self.patch_embed(x)                              # (B, n_patches, d_model)
        cls_tok = self.cls_token.expand(B, -1, -1)                    # (B, 1, d_model)
        h = torch.cat([cls_tok, patches], dim=1)                       # (B, n_patches+1, d_model)
        h = h + self.pos_embed

        attn_w = None
        for layer in self.layers:
            h, attn_w = layer(h)
        h = self.final_norm(h)
        cls_final = h[:, 0, :]
        return self.classifier(cls_final), attn_w   # attn_w: (B, n_patches+1, n_patches+1)


def section_c_model_demo():
    print("\n" + "="*65)
    print("SECTION C -- Vision Transformer: Shape Verification")
    print("="*65)

    torch.manual_seed(SEED)
    model = VisionTransformer(num_layers=4)
    x = torch.randn(4, 3, IMG_SIZE, IMG_SIZE)
    logits, attn_w = model(x)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n  Total parameters: {n_params:,}")
    print(f"  Input shape: {tuple(x.shape)}")
    print(f"  Sequence length (1 CLS + {NUM_PATCHES} patches): {NUM_PATCHES+1}")
    print(f"  Output logits shape: {tuple(logits.shape)}")
    assert logits.shape == (4, NUM_CLASSES)
    print("\n  OK: ViT forward pass produces correct output shape")


# =============================================================================
# SECTION D -- TRAIN VIT, COMPARE vs PHASE 2 TOPIC 2's CNN RESULTS
# =============================================================================

def train_vit(n_epochs=25, n_per_class=200, batch_size=32):
    print("\n" + "="*65)
    print("SECTION D -- Training ViT on Synthetic Shapes")
    print("="*65)

    X, y = generate_shapes_dataset(n_per_class, SEED)
    n_val = int(len(X)*0.2)
    X_va, y_va = X[:n_val], y[:n_val]
    X_tr, y_tr = X[n_val:], y[n_val:]

    loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=batch_size, shuffle=True)
    X_va_t, y_va_t = torch.tensor(X_va).to(DEVICE), torch.tensor(y_va).to(DEVICE)

    torch.manual_seed(SEED)
    model = VisionTransformer(num_layers=4).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    print(f"\n  Train: {len(X_tr)} | Val: {len(X_va)} | Parameters: {sum(p.numel() for p in model.parameters()):,}")

    history = {"train_acc": [], "val_acc": []}
    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        correct, total = 0, 0
        for Xb, Yb in loader:
            Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
            opt.zero_grad()
            logits, _ = model(Xb)
            loss = crit(logits, Yb)
            loss.backward()
            opt.step()
            correct += (logits.argmax(-1) == Yb).sum().item(); total += len(Xb)
        history["train_acc"].append(correct/total)

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(X_va_t)
            val_acc = (val_logits.argmax(-1) == y_va_t).float().mean().item()
        history["val_acc"].append(val_acc)

    elapsed = time.time() - t0
    print(f"\n  Final: train_acc={history['train_acc'][-1]*100:.1f}% | val_acc={history['val_acc'][-1]*100:.1f}%")
    print(f"  Training time: {elapsed:.1f}s")

    print("\n  Phase 2 Topic 2 CNN reference results (5-class shapes, 15 epochs, 200/class):")
    print("    LeNet-5: 99.5% | AlexNet-mini: 100% | VGG-mini: 100% | GoogLeNet-mini: 93%")
    print("    ResNet-mini: 97% | DenseNet-mini: 100%")

    return model, history, elapsed, X_va, y_va


# =============================================================================
# SECTION E -- ATTENTION VISUALIZATION: WHICH PATCHES DOES [CLS] ATTEND TO?
# =============================================================================

def section_e_attention_viz(model, X_va, y_va, example_idx=0):
    print("\n" + "="*65)
    print("SECTION E -- Attention Visualization: [CLS] Token's Patch Attention")
    print("="*65)

    model.eval()
    x = torch.tensor(X_va[example_idx:example_idx+1]).to(DEVICE)
    with torch.no_grad():
        logits, attn_w = model(x)
    pred_class = logits.argmax(-1).item()
    true_class = y_va[example_idx]

    cls_attn = attn_w[0, 0, 1:].cpu().numpy()    # CLS's attention to each PATCH (excl. itself)
    grid_size = IMG_SIZE // PATCH_SIZE
    cls_attn_grid = cls_attn.reshape(grid_size, grid_size)

    print(f"\n  Example: true_class={CLASS_NAMES[true_class]}  predicted={CLASS_NAMES[pred_class]}")
    print(f"  [CLS] attention to each of the {NUM_PATCHES} patches (reshaped to {grid_size}x{grid_size} grid):")
    print(f"  {cls_attn_grid.round(3)}")
    print(f"\n  Attention concentration (max weight): {cls_attn.max():.3f}  (uniform would be {1/NUM_PATCHES:.3f})")

    return X_va[example_idx], true_class, pred_class, cls_attn_grid


# =============================================================================
# SECTION F -- DATA EFFICIENCY: VIT vs SIMPLE CNN
# =============================================================================

class SimpleCNN(nn.Module):
    """A small CNN with genuine convolutional inductive bias, for comparison."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),                                            # 32->16
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                            # 16->8
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        h = self.features(x).flatten(1)
        return self.classifier(h)


def train_model_generic(model, X_tr, y_tr, X_va, y_va, n_epochs=25, is_vit=False):
    loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=32, shuffle=True)
    X_va_t, y_va_t = torch.tensor(X_va).to(DEVICE), torch.tensor(y_va).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        model.train()
        for Xb, Yb in loader:
            Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
            opt.zero_grad()
            out = model(Xb)
            logits = out[0] if is_vit else out
            loss = crit(logits, Yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        out = model(X_va_t)
        logits = out[0] if is_vit else out
        return (logits.argmax(-1) == y_va_t).float().mean().item()


def section_f_data_efficiency():
    print("\n" + "="*65)
    print("SECTION F -- Data Efficiency: ViT vs Simple CNN")
    print("  (testing whether ViT's lack of inductive bias requires more data)")
    print("="*65)

    n_per_class_options = [5, 15, 40, 100]
    results = {"ViT": [], "CNN": []}

    X_va, y_va = generate_shapes_dataset(30, seed=SEED+500)   # FIXED val set across all conditions

    print(f"\n  {'Train/class':>12} | {'ViT val_acc':>12} | {'CNN val_acc':>12}")
    print("  " + "-"*42)

    for n_per_class in n_per_class_options:
        X_tr, y_tr = generate_shapes_dataset(n_per_class, seed=SEED+1)

        torch.manual_seed(SEED)
        vit = VisionTransformer(num_layers=4).to(DEVICE)
        vit_acc = train_model_generic(vit, X_tr, y_tr, X_va, y_va, n_epochs=25, is_vit=True)

        torch.manual_seed(SEED)
        cnn = SimpleCNN().to(DEVICE)
        cnn_acc = train_model_generic(cnn, X_tr, y_tr, X_va, y_va, n_epochs=25, is_vit=False)

        results["ViT"].append(vit_acc); results["CNN"].append(cnn_acc)
        print(f"  {n_per_class:>12} | {vit_acc*100:>11.1f}% | {cnn_acc*100:>11.1f}%")

    print("\n  OK: Data-efficiency comparison complete")
    return n_per_class_options, results


# =============================================================================
# SECTION G -- VISUALIZATION
# =============================================================================

def build_figures(train_hist, attn_example, data_eff_results):
    fig = plt.figure(figsize=(17, 10))
    fig.suptitle("Phase 4 -- Topic 5 (Extra): Vision Transformer (ViT)", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    a1, a2, a3, a4 = (fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]),
                      fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1]))

    ep = range(1, len(train_hist["train_acc"])+1)
    a1.plot(ep, [v*100 for v in train_hist["train_acc"]], color="#3498db", lw=2, label="Train")
    a1.plot(ep, [v*100 for v in train_hist["val_acc"]], color="#e74c3c", lw=2, label="Val")
    a1.set_title("ViT Training Curve (Shapes Classification)", fontweight="bold", fontsize=10)
    a1.set_xlabel("Epoch"); a1.set_ylabel("Accuracy (%)")
    a1.legend(fontsize=9); a1.grid(True, alpha=0.3)

    img, true_class, pred_class, attn_grid = attn_example
    a2.imshow(img.transpose(1,2,0))
    a2.set_title(f"Example: true={CLASS_NAMES[true_class]}, pred={CLASS_NAMES[pred_class]}",
                fontweight="bold", fontsize=10)
    a2.axis("off")

    im3 = a3.imshow(attn_grid, cmap="hot", interpolation="nearest")
    a3.set_title("[CLS] Attention to Each Patch", fontweight="bold", fontsize=10)
    a3.set_xlabel("Patch column"); a3.set_ylabel("Patch row")
    plt.colorbar(im3, ax=a3, fraction=0.046)

    lengths, data_eff = data_eff_results
    a4.plot(lengths, [v*100 for v in data_eff["ViT"]], "o-", color="#9b59b6", lw=2, label="ViT")
    a4.plot(lengths, [v*100 for v in data_eff["CNN"]], "s-", color="#27ae60", lw=2, label="Simple CNN")
    a4.set_title("Data Efficiency: ViT vs CNN", fontweight="bold", fontsize=10)
    a4.set_xlabel("Training images per class"); a4.set_ylabel("Val Accuracy (%)")
    a4.legend(fontsize=9); a4.grid(True, alpha=0.3); a4.set_xscale("log")

    plt.tight_layout()
    path = os.path.join(RESULTS, "05_vit_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  [VIZ] Dashboard saved -> {path}")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "#"*65)
    print("  Phase 4 -- Topic 5 (Extra): Vision Transformer (ViT)")
    print("#"*65)

    section_a_dataset_demo()
    section_b_patch_embed_equivalence()
    section_c_model_demo()

    model, train_hist, elapsed, X_va, y_va = train_vit(n_epochs=25)
    attn_example = section_e_attention_viz(model, X_va, y_va)
    data_eff_results = section_f_data_efficiency()

    build_figures(train_hist, attn_example, data_eff_results)

    print("\n" + "#"*65)
    print("  DONE: Topic 5 complete. Phase 4 -- Attention & Transformers is now FULLY complete.")
    print("#"*65 + "\n")


if __name__ == "__main__":
    main()
