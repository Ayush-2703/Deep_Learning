"""
Phase 2 — Topic 5: Transfer Learning & Fine-tuning
======================================================
Repository : deep-learning-mastery/phase-2-cnns/05-transfer-learning-finetuning/
File       : implementation.py

Since this environment has no internet access to download real pretrained
ImageNet weights, we honestly SIMULATE the full transfer learning workflow:
  1. Pretrain a ResNet-style CNN from scratch on a LARGE synthetic "source" task
  2. Fine-tune it on a SMALL, domain-shifted "target" task using three
     different strategies, comparing their convergence and final accuracy
This is methodologically identical to real-world transfer learning (the
mechanics of freezing/fine-tuning are unaffected by whether the source task
was ImageNet or a synthetic task) — only the specific pretrained features differ.

Sections:
  A │ Source dataset (large, solid-color shapes) + Target dataset (small, domain-shifted)
  B │ ResNet-style backbone (reused architecture pattern from Topic 2)
  C │ Pretrain on source task
  D │ Three target fine-tuning strategies: from-scratch, feature-extraction, full fine-tune
  E │ Bonus: discriminative (layer-wise) learning rates
  F │ Visualization dashboard
"""

import os, time, copy, warnings
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

IMG_SIZE = 32
CLASS_NAMES = ["circle", "square", "triangle", "star", "diamond"]
NUM_CLASSES = len(CLASS_NAMES)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION A — SOURCE (LARGE) AND TARGET (SMALL, DOMAIN-SHIFTED) DATASETS
# ═════════════════════════════════════════════════════════════════════════════

def _shape_geometry(cls, cx, cy, r):
    if cls == 0:    return ("ellipse", [cx-r, cy-r, cx+r, cy+r])
    elif cls == 1:  return ("rectangle", [cx-r, cy-r, cx+r, cy+r])
    elif cls == 2:  return ("polygon", [(cx, cy-r), (cx-r, cy+r), (cx+r, cy+r)])
    elif cls == 3:
        pts = []
        for i in range(10):
            ang = np.pi/2 + i * np.pi/5
            rad = r if i % 2 == 0 else r*0.45
            pts.append((cx + rad*np.cos(ang), cy - rad*np.sin(ang)))
        return ("polygon", pts)
    else:           return ("polygon", [(cx, cy-r), (cx+r, cy), (cx, cy+r), (cx-r, cy)])


def _draw(draw, kind, geom, fill):
    if kind == "ellipse":   draw.ellipse(geom, fill=fill)
    elif kind == "rectangle": draw.rectangle(geom, fill=fill)
    else: draw.polygon(geom, fill=fill)


def generate_shape_image(rng, cls, style="source", size=IMG_SIZE):
    """
    style="source": solid random color, near-black background, low noise
                     (large, "easy" pretraining domain)
    style="target": two-tone shaded fill, lighter gray background, higher
                     noise (small, domain-shifted target — same TASK, different
                     visual DOMAIN, simulating e.g. studio-photo -> real-world-photo shift)
    """
    if style == "source":
        bg = (8, 8, 8)
        noise_sigma = 0.03
    else:
        bg = tuple(int(c) for c in rng.integers(45, 75, size=3))
        noise_sigma = 0.07

    img = Image.new("RGB", (size, size), color=bg)
    draw = ImageDraw.Draw(img)

    cx = size//2 + int(rng.integers(-3, 4))
    cy = size//2 + int(rng.integers(-3, 4))
    r  = int(rng.integers(8, 13))

    color1 = tuple(int(c) for c in rng.integers(110, 256, size=3))
    kind, geom = _shape_geometry(cls, cx, cy, r)
    _draw(draw, kind, geom, color1)

    if style == "target":
        # Two-tone "shaded" effect: smaller inset shape in a second color —
        # simulates a gradient-like shading style absent from the source domain
        color2 = tuple(int(c) for c in rng.integers(110, 256, size=3))
        inset_r = max(2, int(r * 0.55))
        kind2, geom2 = _shape_geometry(cls, cx, cy, inset_r)
        _draw(draw, kind2, geom2, color2)

    arr = np.array(img, dtype=np.float32) / 255.0
    noise = rng.normal(0, noise_sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 1).transpose(2, 0, 1)
    return arr


def generate_dataset(n_per_class, style, seed):
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for cls in range(NUM_CLASSES):
        for _ in range(n_per_class):
            images.append(generate_shape_image(rng, cls, style=style))
            labels.append(cls)
    images = np.stack(images)
    labels = np.array(labels, dtype=np.int64)
    perm = rng.permutation(len(images))
    return images[perm], labels[perm]


def build_loaders(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def build_all_data():
    print("\n" + "="*65)
    print("SECTION A — Source (large) & Target (small, domain-shifted) Datasets")
    print("="*65)

    # Source: large, solid-color, low-noise — our "ImageNet stand-in"
    X_src, y_src = generate_dataset(n_per_class=300, style="source", seed=SEED)
    n_val_src = int(len(X_src) * 0.2)
    src_train_loader = build_loaders(X_src[n_val_src:], y_src[n_val_src:], 32, True)
    src_val_loader   = build_loaders(X_src[:n_val_src], y_src[:n_val_src], 32, False)

    # Target: small, domain-shifted (two-tone shading, lighter bg, more noise)
    X_tgt, y_tgt = generate_dataset(n_per_class=20, style="target", seed=SEED+100)
    n_val_tgt = int(len(X_tgt) * 0.33)
    X_tgt_va, y_tgt_va = X_tgt[:n_val_tgt], y_tgt[:n_val_tgt]
    X_tgt_tr, y_tgt_tr = X_tgt[n_val_tgt:], y_tgt[n_val_tgt:]
    tgt_train_loader = build_loaders(X_tgt_tr, y_tgt_tr, 16, True)
    tgt_val_loader   = build_loaders(X_tgt_va, y_tgt_va, 16, False)

    print(f"\n  Source domain: {len(X_src)} images  ({len(X_src)-n_val_src} train / {n_val_src} val)")
    print(f"  Target domain: {len(X_tgt)} images  ({len(X_tgt_tr)} train / {n_val_tgt} val)")
    print(f"  Target/Source train ratio: {len(X_tgt_tr)/(len(X_src)-n_val_src)*100:.1f}%  "
          f"(simulating realistic scarce target-domain labels)")

    return (src_train_loader, src_val_loader,
            tgt_train_loader, tgt_val_loader,
            X_src, y_src, X_tgt, y_tgt)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION B — RESNET-STYLE BACKBONE (consistent with Phase 2 Topic 2's pattern)
# ═════════════════════════════════════════════════════════════════════════════

class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(c_out)
        self.relu  = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class TransferNet(nn.Module):
    """
    ResNet-style network split explicitly into `backbone` (feature extractor)
    and `classifier` (task-specific head) — this clean separation is what
    makes freezing/fine-tuning strategies trivial to implement and reason about.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            BasicBlock(32, 32), BasicBlock(32, 32),
            BasicBlock(32, 64, stride=2), BasicBlock(64, 64),
            BasicBlock(64, 128, stride=2), BasicBlock(128, 128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )    # output: (batch, 128) — generic visual features
        self.classifier = nn.Linear(128, num_classes)    # task-specific head

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def feature_dim(self):
        return 128


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION C — PRETRAIN ON SOURCE TASK
# ═════════════════════════════════════════════════════════════════════════════

def evaluate(model, loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for Xb, Yb in loader:
            Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
            logits = model(Xb)
            loss_sum += crit(logits, Yb).item() * len(Xb)
            correct += (logits.argmax(1) == Yb).sum().item()
            total += len(Xb)
    return loss_sum/total, correct/total


def pretrain_on_source(train_loader, val_loader, n_epochs=15, lr=1e-3):
    print("\n" + "="*65)
    print("SECTION C — Pretraining on Source Task")
    print("="*65)

    model = TransferNet().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    print(f"\n  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        for Xb, Yb in train_loader:
            Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(Xb), Yb)
            loss.backward()
            opt.step()

        if (epoch+1) % 5 == 0 or epoch == 0:
            vl, va = evaluate(model, val_loader)
            print(f"    Epoch {epoch+1:3d}/{n_epochs} | val_loss={vl:.4f} | val_acc={va*100:.1f}%")

    elapsed = time.time() - t0
    vl, va = evaluate(model, val_loader)
    print(f"\n  ✓ Source pretraining complete in {elapsed:.1f}s | final val_acc={va*100:.1f}%")
    return model


# ═════════════════════════════════════════════════════════════════════════════
# SECTION D — THREE TARGET FINE-TUNING STRATEGIES
# ═════════════════════════════════════════════════════════════════════════════

def train_on_target(model, train_loader, val_loader, n_epochs, lr,
                    freeze_backbone=False, param_groups=None):
    """
    Generic target-task training loop.

    freeze_backbone : if True, backbone.requires_grad=False AND backbone is
                       kept in .eval() mode throughout (so its BatchNorm
                       running stats don't drift — theory.md §4)
    param_groups    : optional explicit optimizer param groups (for the
                       discriminative-LR experiment in Section E); if None,
                       a single Adam optimizer over all trainable params is used
    """
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    trainable = count_trainable(model)

    if param_groups is not None:
        opt = optim.Adam(param_groups)
    else:
        opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    crit = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(n_epochs):
        model.train()
        if freeze_backbone:
            model.backbone.eval()    # keep frozen backbone's BatchNorm stats fixed

        tl, tn = 0.0, 0
        for Xb, Yb in train_loader:
            Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(Xb), Yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(Xb); tn += len(Xb)

        vl, va = evaluate(model, val_loader)
        history["train_loss"].append(tl/tn)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)

    return model, history, trainable


def run_target_strategies(pretrained_model, tgt_train_loader, tgt_val_loader, n_epochs=40):
    print("\n" + "="*65)
    print("SECTION D — Three Target Fine-Tuning Strategies")
    print("="*65)

    results = {}

    # ── Strategy 1: From-scratch (random init, no transfer) ──────────────────
    print("\n  [1/3] From-scratch (random init, train on tiny target data only)")
    torch.manual_seed(SEED)
    scratch_model = TransferNet().to(DEVICE)
    scratch_model, scratch_hist, scratch_params = train_on_target(
        scratch_model, tgt_train_loader, tgt_val_loader, n_epochs, lr=1e-3)
    results["From-scratch"] = {"history": scratch_hist, "trainable_params": scratch_params}
    print(f"        trainable_params={scratch_params:,} | final val_acc={scratch_hist['val_acc'][-1]*100:.1f}%")

    # ── Strategy 2: Feature extraction (frozen backbone, new head only) ──────
    print("\n  [2/3] Feature extraction (frozen pretrained backbone, train new head)")
    fe_model = copy.deepcopy(pretrained_model)
    fe_model.classifier = nn.Linear(fe_model.feature_dim(), NUM_CLASSES).to(DEVICE)    # fresh head
    fe_model, fe_hist, fe_params = train_on_target(
        fe_model, tgt_train_loader, tgt_val_loader, n_epochs, lr=1e-3, freeze_backbone=True)
    results["Feature Extraction"] = {"history": fe_hist, "trainable_params": fe_params}
    print(f"        trainable_params={fe_params:,} | final val_acc={fe_hist['val_acc'][-1]*100:.1f}%")

    # ── Strategy 3: Full fine-tuning (unfrozen backbone, low LR) ──────────────
    print("\n  [3/3] Full fine-tuning (unfrozen pretrained backbone, LOW learning rate)")
    ft_model = copy.deepcopy(pretrained_model)
    ft_model.classifier = nn.Linear(ft_model.feature_dim(), NUM_CLASSES).to(DEVICE)    # fresh head
    ft_model, ft_hist, ft_params = train_on_target(
        ft_model, tgt_train_loader, tgt_val_loader, n_epochs, lr=1e-4, freeze_backbone=False)
    results["Full Fine-tuning"] = {"history": ft_hist, "trainable_params": ft_params}
    print(f"        trainable_params={ft_params:,} | final val_acc={ft_hist['val_acc'][-1]*100:.1f}%")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# SECTION E — BONUS: DISCRIMINATIVE (LAYER-WISE) LEARNING RATES
# ═════════════════════════════════════════════════════════════════════════════

def run_discriminative_lr(pretrained_model, tgt_train_loader, tgt_val_loader, n_epochs=40):
    print("\n" + "="*65)
    print("SECTION E — Bonus: Discriminative (Layer-wise) Learning Rates")
    print("="*65)

    model = copy.deepcopy(pretrained_model)
    model.classifier = nn.Linear(model.feature_dim(), NUM_CLASSES).to(DEVICE)

    # model.backbone is itself an nn.Sequential; index into it to split
    # "early" (generic) vs "late" (task-specific) layers for differential LR.
    backbone_modules = list(model.backbone.children())
    early_layers = nn.Sequential(*backbone_modules[:5])    # stem + first 2 BasicBlocks
    late_layers  = nn.Sequential(*backbone_modules[5:])    # remaining BasicBlocks + GAP

    param_groups = [
        {"params": early_layers.parameters(), "lr": 1e-5},   # most generic — barely adjust
        {"params": late_layers.parameters(),  "lr": 1e-4},   # more task-specific — adjust more
        {"params": model.classifier.parameters(), "lr": 1e-3},   # new head — full-strength
    ]

    print(f"\n  Early layers LR=1e-5 | Late layers LR=1e-4 | Classifier LR=1e-3")

    model, history, params = train_on_target(
        model, tgt_train_loader, tgt_val_loader, n_epochs, lr=None, param_groups=param_groups)

    print(f"  Final val_acc={history['val_acc'][-1]*100:.1f}%")
    return history


# ═════════════════════════════════════════════════════════════════════════════
# SECTION F — VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def build_figures(strategy_results, disc_lr_history, X_src, y_src, X_tgt, y_tgt):
    colors = {"From-scratch": "#e74c3c", "Feature Extraction": "#3498db",
              "Full Fine-tuning": "#27ae60", "Discriminative LR": "#9b59b6"}

    # ── Figure 1: sample images from both domains ─────────────────────────────
    fig0, axes0 = plt.subplots(2, 5, figsize=(13, 5.5))
    fig0.suptitle("Source Domain (top) vs Target Domain (bottom) — Same Classes, Different Style",
                 fontsize=12, fontweight="bold")
    for cls in range(5):
        src_idx = np.where(y_src == cls)[0][0]
        tgt_idx = np.where(y_tgt == cls)[0][0]
        axes0[0, cls].imshow(X_src[src_idx].transpose(1,2,0)); axes0[0, cls].axis("off")
        axes0[0, cls].set_title(CLASS_NAMES[cls], fontsize=10, fontweight="bold")
        axes0[1, cls].imshow(X_tgt[tgt_idx].transpose(1,2,0)); axes0[1, cls].axis("off")
    plt.tight_layout()
    path0 = os.path.join(RESULTS, "05_domain_comparison.png")
    plt.savefig(path0, dpi=150, bbox_inches="tight")
    plt.close(fig0)
    print(f"\n  [VIZ] Domain comparison saved → {path0}")

    # ── Figure 2: strategy comparison ──────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Phase 2 — Topic 5: Transfer Learning Strategy Comparison",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)
    a1, a2, a3, a4 = (fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]),
                      fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1]))

    all_results = dict(strategy_results)
    if disc_lr_history is not None:
        all_results["Discriminative LR"] = {"history": disc_lr_history, "trainable_params": None}

    for name, data in all_results.items():
        ep = range(1, len(data["history"]["val_loss"])+1)
        a1.plot(ep, data["history"]["val_loss"], color=colors[name], lw=2, label=name)
        a2.plot(ep, [v*100 for v in data["history"]["val_acc"]], color=colors[name], lw=2, label=name)

    a1.set_title("Target Validation Loss", fontweight="bold", fontsize=10)
    a1.set_xlabel("Epoch"); a1.set_ylabel("CE Loss"); a1.legend(fontsize=8); a1.grid(True, alpha=0.3)

    a2.set_title("Target Validation Accuracy", fontweight="bold", fontsize=10)
    a2.set_xlabel("Epoch"); a2.set_ylabel("Accuracy (%)"); a2.legend(fontsize=8); a2.grid(True, alpha=0.3)

    # Final accuracy bar
    names = list(all_results.keys())
    finals = [all_results[n]["history"]["val_acc"][-1]*100 for n in names]
    bars = a3.bar(names, finals, color=[colors[n] for n in names])
    for bar, v in zip(bars, finals):
        a3.text(bar.get_x()+bar.get_width()/2, v+1, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    a3.set_title("Final Target Validation Accuracy", fontweight="bold", fontsize=10)
    a3.set_ylabel("Accuracy (%)"); a3.set_ylim(0, 105)
    a3.tick_params(axis="x", rotation=15, labelsize=8)
    a3.grid(True, axis="y", alpha=0.3)

    # Trainable parameter count bar (only for the 3 main strategies)
    main_names = list(strategy_results.keys())
    param_counts = [strategy_results[n]["trainable_params"] for n in main_names]
    a4.bar(main_names, param_counts, color=[colors[n] for n in main_names])
    a4.set_yscale("log")
    a4.set_title("Trainable Parameters (log scale)", fontweight="bold", fontsize=10)
    a4.set_ylabel("Trainable Params"); a4.tick_params(axis="x", rotation=15, labelsize=8)
    a4.grid(True, axis="y", alpha=0.3, which="both")

    plt.tight_layout()
    path = os.path.join(RESULTS, "05_strategy_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [VIZ] Strategy comparison saved → {path}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "▓"*65)
    print("  Phase 2 — Topic 5: Transfer Learning & Fine-tuning")
    print("▓"*65)

    (src_train, src_val, tgt_train, tgt_val,
     X_src, y_src, X_tgt, y_tgt) = build_all_data()

    pretrained_model = pretrain_on_source(src_train, src_val, n_epochs=15)

    strategy_results = run_target_strategies(pretrained_model, tgt_train, tgt_val, n_epochs=40)
    disc_lr_history = run_discriminative_lr(pretrained_model, tgt_train, tgt_val, n_epochs=40)

    print("\n" + "="*65)
    print("SUMMARY — Final Target-Domain Validation Accuracy")
    print("="*65)
    for name, data in strategy_results.items():
        print(f"  {name:22s} | acc={data['history']['val_acc'][-1]*100:5.1f}% | "
              f"trainable_params={data['trainable_params']:,}")
    print(f"  {'Discriminative LR':22s} | acc={disc_lr_history['val_acc'][-1]*100:5.1f}%")

    build_figures(strategy_results, disc_lr_history, X_src, y_src, X_tgt, y_tgt)

    print("\n" + "▓"*65)
    print("  ✓ Topic 5 complete. Phase 2 — CNNs & Computer Vision is now FULLY complete.")
    print("▓"*65 + "\n")


if __name__ == "__main__":
    main()
