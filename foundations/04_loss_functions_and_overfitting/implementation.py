"""
Phase 1 — Topic 4: Loss Functions, Overfitting & Bias-Variance Trade-off
=========================================================================
Repository : deep-learning-mastery/phase-1-foundations/04-loss-functions-and-overfitting/
File       : implementation.py

Sections:
  A │ Loss function implementations and comparisons (MSE, MAE, Huber, BCE, CE, Focal)
  B │ Overfitting demonstration — model complexity sweep
  C │ Bias-variance decomposition experiment
  D │ Learning curves (training set size effect)
  E │ Visualization dashboard
"""

import os, warnings
import numpy as np
import matplotlib, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

SEED = 42; DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = "results"; os.makedirs(RESULTS, exist_ok=True)
np.random.seed(SEED); torch.manual_seed(SEED)
print(f"[CONFIG] Device: {DEVICE}")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION A — LOSS FUNCTION IMPLEMENTATIONS
# ═════════════════════════════════════════════════════════════════════════════

class MSELoss(nn.Module):
    """Mean Squared Error: (1/N)Σ(y-ŷ)²"""
    def forward(self, pred, target):
        return torch.mean((pred - target) ** 2)

class MAELoss(nn.Module):
    """Mean Absolute Error: (1/N)Σ|y-ŷ|"""
    def forward(self, pred, target):
        return torch.mean(torch.abs(pred - target))

class HuberLoss(nn.Module):
    """Huber (Smooth L1): quadratic for |e|≤δ, linear for |e|>δ"""
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta
    def forward(self, pred, target):
        e   = torch.abs(pred - target)
        qdr = 0.5 * e ** 2
        lin = self.delta * e - 0.5 * self.delta ** 2
        return torch.mean(torch.where(e <= self.delta, qdr, lin))

class FocalLoss(nn.Module):
    """Focal Loss: −(1−p)^γ · BCE  for imbalanced classification."""
    def __init__(self, gamma: float = 2.0, eps: float = 1e-7):
        super().__init__()
        self.gamma = gamma; self.eps = eps
    def forward(self, pred, target):
        p   = torch.sigmoid(pred)              # pred = logits
        bce = -(target * torch.log(p + self.eps) +
                (1 - target) * torch.log(1 - p + self.eps))
        focal_weight = (1 - p) ** self.gamma * target + p ** self.gamma * (1 - target)
        return torch.mean(focal_weight * bce)

def demo_loss_functions():
    """Evaluate all loss functions on a range of prediction errors."""
    print("\n" + "="*65)
    print("SECTION A — Loss Function Comparison")
    print("="*65)
    errors = np.linspace(-3, 3, 200)
    target = torch.zeros(len(errors))

    losses = {
        "MSE":       MSELoss(),
        "MAE":       MAELoss(),
        "Huber(δ=1)":HuberLoss(1.0),
    }
    curves = {}
    for name, fn in losses.items():
        vals = []
        for e in errors:
            pred = torch.tensor([e])
            vals.append(fn(pred, torch.tensor([0.0])).item())
        curves[name] = np.array(vals)
        val_at_2 = curves[name][np.argmin(np.abs(errors - 2))]
        print(f"  {name:12s} | at error=2: {val_at_2:.4f}")
    return errors, curves

# ═════════════════════════════════════════════════════════════════════════════
# SECTION B — OVERFITTING DEMONSTRATION
# Train models of increasing complexity; observe train vs val divergence
# ═════════════════════════════════════════════════════════════════════════════

def _build_mlp(hidden_sizes: list, input_size: int = 2) -> nn.Module:
    layers = []
    prev = input_size
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, 1))
    layers.append(nn.Sigmoid())
    model = nn.Sequential(*layers)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    return model

def _count_params(model): return sum(p.numel() for p in model.parameters())

def _train_eval(model, train_loader, val_loader, n_epochs=200, lr=1e-3):
    model = model.to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=lr)
    crit  = nn.BCELoss()
    hist  = {"train": [], "val": []}
    for _ in range(n_epochs):
        model.train()
        tl, tn = 0.0, 0
        for Xb, Yb in train_loader:
            Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
            opt.zero_grad(); pred = model(Xb); loss = crit(pred, Yb)
            loss.backward(); opt.step()
            tl += loss.item()*len(Xb); tn += len(Xb)
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for Xb, Yb in val_loader:
                Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
                pred = model(Xb); loss = crit(pred, Yb)
                vl += loss.item()*len(Xb); vn += len(Xb)
        hist["train"].append(tl/tn); hist["val"].append(vl/vn)
    return hist

def overfitting_demo(train_loader, val_loader):
    """
    Train 5 architectures of increasing capacity.
    Tiny model → underfitting; massive model → overfitting.
    """
    print("\n" + "="*65)
    print("SECTION B — Overfitting Demonstration (Complexity Sweep)")
    print("="*65)

    architectures = {
        "Tiny  [2]":          [2],
        "Small [16,16]":      [16, 16],
        "Medium [64,64]":     [64, 64],
        "Large [256,256,256]":[256, 256, 256],
        "Huge  [512]×5":      [512, 512, 512, 512, 512],
    }

    results = {}
    print(f"\n  {'Model':25s} | {'Params':>8} | {'Train L':>8} | {'Val L':>8} | {'Gap':>8}")
    print("  " + "─"*62)
    for name, hidden in architectures.items():
        torch.manual_seed(SEED)
        model = _build_mlp(hidden)
        hist  = _train_eval(model, train_loader, val_loader, n_epochs=200)
        tl, vl = hist["train"][-1], hist["val"][-1]
        gap    = vl - tl
        results[name] = {"history": hist, "params": _count_params(model)}
        print(f"  {name:25s} | {_count_params(model):>8,} | {tl:>8.4f} | {vl:>8.4f} | {gap:>+8.4f}")

    return results

# ═════════════════════════════════════════════════════════════════════════════
# SECTION C — BIAS-VARIANCE DECOMPOSITION
# Train many models on different bootstrap samples; measure bias and variance
# ═════════════════════════════════════════════════════════════════════════════

def bias_variance_experiment(X_np, y_np, n_trials=25, n_epochs=150):
    """
    Empirical bias-variance decomposition for two model complexities.

    For each trial t:
      1. Sample a bootstrap training set
      2. Train model → get predictions on fixed test set
    After all trials:
      bias²  = (mean_prediction − true_label)²  (averaged over test set)
      variance = Var(predictions across trials)   (averaged over test set)
    """
    print("\n" + "="*65)
    print("SECTION C — Bias-Variance Decomposition")
    print(f"  Trials={n_trials} | Epochs per trial={n_epochs}")
    print("="*65)

    # Fixed test set
    X_tr, X_te, y_tr, y_te = train_test_split(X_np, y_np, test_size=0.3,
                                               random_state=SEED, stratify=y_np)
    scaler = StandardScaler().fit(X_tr)
    X_tr   = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    X_te_t = torch.tensor(X_te_s, dtype=torch.float32).to(DEVICE)
    y_te_t = torch.tensor(y_te,   dtype=torch.float32).to(DEVICE)

    models_cfg = {
        "Low-complexity\n[4]":    [4],
        "High-complexity\n[128,128]": [128, 128],
    }

    results = {}
    for label, hidden in models_cfg.items():
        all_preds = []  # (n_trials, n_test_samples)

        for trial in range(n_trials):
            rng  = np.random.default_rng(trial)
            idx  = rng.integers(0, len(X_tr), size=len(X_tr))   # bootstrap sample
            Xb, Yb = X_tr[idx], y_tr[idx]

            Xbt = torch.tensor(Xb, dtype=torch.float32)
            Ybt = torch.tensor(Yb, dtype=torch.float32).unsqueeze(1)
            loader = DataLoader(TensorDataset(Xbt, Ybt), batch_size=32, shuffle=True)

            torch.manual_seed(trial)
            model = _build_mlp(hidden).to(DEVICE)
            opt   = optim.Adam(model.parameters(), lr=1e-3)
            crit  = nn.BCELoss()

            model.train()
            for _ in range(n_epochs):
                for Xmb, Ymb in loader:
                    Xmb, Ymb = Xmb.to(DEVICE), Ymb.to(DEVICE)
                    opt.zero_grad(); loss = crit(model(Xmb), Ymb); loss.backward(); opt.step()

            model.eval()
            with torch.no_grad():
                preds = model(X_te_t).cpu().numpy().flatten()
            all_preds.append(preds)

        all_preds = np.array(all_preds)   # (n_trials, n_test)
        mean_pred = all_preds.mean(axis=0)
        y_true    = y_te.astype(float)

        bias_sq  = np.mean((mean_pred - y_true) ** 2)
        variance = np.mean(all_preds.var(axis=0))
        mse      = np.mean((all_preds - y_true[None, :]) ** 2)

        results[label] = {"bias_sq": bias_sq, "variance": variance,
                          "mse": mse, "all_preds": all_preds}
        print(f"\n  {label.replace(chr(10), ' '):30s}")
        print(f"    Bias²    = {bias_sq:.4f}")
        print(f"    Variance = {variance:.4f}")
        print(f"    Total MSE= {mse:.4f}  (≈ Bias²+Var = {bias_sq+variance:.4f})")

    return results

# ═════════════════════════════════════════════════════════════════════════════
# SECTION D — LEARNING CURVES (Training Set Size)
# ═════════════════════════════════════════════════════════════════════════════

def learning_curve_experiment(X_np, y_np, n_epochs=150):
    """
    Train on increasing fractions of training data.
    High-variance model: train error low, val error high, gap closes with more data.
    High-bias model:     both errors high, gap small, more data doesn't help.
    """
    print("\n" + "="*65)
    print("SECTION D — Learning Curves (Training Set Size)")
    print("="*65)

    X_tr, X_va, y_tr, y_va = train_test_split(X_np, y_np, test_size=0.25,
                                               random_state=SEED, stratify=y_np)
    scaler = StandardScaler().fit(X_tr)
    X_tr   = scaler.transform(X_tr); X_va = scaler.transform(X_va)

    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32).unsqueeze(1)
    val_ds = TensorDataset(X_va_t, y_va_t)
    val_ld = DataLoader(val_ds, batch_size=64)

    sizes   = [30, 60, 100, 150, 200, 300, len(X_tr)]
    configs = {
        "High-Variance [256,256]": [256, 256],
        "High-Bias     [4]":       [4],
    }
    results = {}
    crit = nn.BCELoss()

    for label, hidden in configs.items():
        tr_errs, va_errs = [], []
        for sz in sizes:
            idx  = np.random.default_rng(SEED).choice(len(X_tr), size=sz, replace=False)
            Xsub = torch.tensor(X_tr[idx], dtype=torch.float32)
            Ysub = torch.tensor(y_tr[idx], dtype=torch.float32).unsqueeze(1)
            ld   = DataLoader(TensorDataset(Xsub, Ysub), batch_size=min(32, sz), shuffle=True)

            torch.manual_seed(SEED)
            model = _build_mlp(hidden).to(DEVICE)
            opt   = optim.Adam(model.parameters(), lr=1e-3)

            for _ in range(n_epochs):
                model.train()
                for Xb, Yb in ld:
                    Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
                    opt.zero_grad(); crit(model(Xb), Yb).backward(); opt.step()

            model.eval()
            with torch.no_grad():
                t_loss = crit(model(Xsub.to(DEVICE)), Ysub.to(DEVICE)).item()
                v_loss = sum(crit(model(Xb.to(DEVICE)), Yb.to(DEVICE)).item()*len(Xb)
                             for Xb, Yb in val_ld) / len(X_va)
            tr_errs.append(t_loss); va_errs.append(v_loss)

        results[label] = {"sizes": sizes, "train": tr_errs, "val": va_errs}
        print(f"  {label}: final train={tr_errs[-1]:.4f} | val={va_errs[-1]:.4f}")

    return results

# ═════════════════════════════════════════════════════════════════════════════
# SECTION E — VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def build_figures(errors, loss_curves, overfit_results, bv_results, lc_results):
    # ── Figure 1: Loss functions + overfitting ────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Phase 1 — Topic 4: Loss Functions, Overfitting & Bias-Variance",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    ax1, ax2, ax3 = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[0,2])
    ax4, ax5, ax6 = fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1]), fig.add_subplot(gs[1,2])

    # Panel 1: Regression loss curves
    colors_l = {"MSE": "#e74c3c", "MAE": "#27ae60", "Huber(δ=1)": "#3498db"}
    for name, curve in loss_curves.items():
        ax1.plot(errors, curve, lw=2.2, color=colors_l[name], label=name)
    ax1.set_title("Regression Loss Functions", fontweight="bold", fontsize=10)
    ax1.set_xlabel("Prediction Error (ŷ−y)"); ax1.set_ylabel("Loss value")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3); ax1.set_xlim(-3, 3)

    # Panel 2: Overfitting — train/val gap per model
    names  = list(overfit_results.keys())
    params = [overfit_results[n]["params"] for n in names]
    t_loss = [overfit_results[n]["history"]["train"][-1] for n in names]
    v_loss = [overfit_results[n]["history"]["val"][-1]   for n in names]
    x_pos  = range(len(names))
    ax2.bar([x-0.2 for x in x_pos], t_loss, 0.38, color="#27ae60", label="Train", alpha=0.85)
    ax2.bar([x+0.2 for x in x_pos], v_loss, 0.38, color="#e74c3c", label="Val",   alpha=0.85)
    ax2.set_xticks(list(x_pos))
    ax2.set_xticklabels([n.split("[")[0].strip() for n in names], fontsize=8, rotation=15)
    ax2.set_title("Complexity Sweep: Train vs Val Loss", fontweight="bold", fontsize=10)
    ax2.set_ylabel("BCE Loss"); ax2.legend(fontsize=9); ax2.grid(True, axis="y", alpha=0.3)

    # Panel 3: Overfitting convergence curves for extreme models
    colors_m = ["#9b59b6", "#27ae60", "#2980b9", "#e67e22", "#e74c3c"]
    for (name, data), col in zip(overfit_results.items(), colors_m):
        ep = range(1, len(data["history"]["train"])+1)
        ax3.plot(ep, data["history"]["train"], color=col, lw=1.5, ls="-",  alpha=0.9)
        ax3.plot(ep, data["history"]["val"],   color=col, lw=1.5, ls="--", alpha=0.9,
                 label=name.split("[")[0].strip())
    ax3.set_title("Training Curves (─train, --val)", fontweight="bold", fontsize=10)
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("BCE Loss")
    ax3.legend(fontsize=7, loc="upper right"); ax3.grid(True, alpha=0.3)

    # Panel 4: Bias-Variance bar chart
    labels_bv = [k.replace("\n", " ") for k in bv_results.keys()]
    bias_vals  = [bv_results[k]["bias_sq"]  for k in bv_results]
    var_vals   = [bv_results[k]["variance"]  for k in bv_results]
    x_bv = range(len(labels_bv))
    ax4.bar(x_bv, bias_vals, 0.35, color="#e74c3c", label="Bias²")
    ax4.bar(x_bv, var_vals,  0.35, color="#3498db", label="Variance",
            bottom=bias_vals)
    ax4.set_xticks(list(x_bv)); ax4.set_xticklabels(labels_bv, fontsize=9)
    ax4.set_title("Bias-Variance Decomposition", fontweight="bold", fontsize=10)
    ax4.set_ylabel("Expected Squared Error"); ax4.legend(fontsize=9)
    ax4.grid(True, axis="y", alpha=0.3)

    # Panel 5 & 6: Learning curves
    colors_lc = ["#e74c3c", "#2980b9"]
    for i, (label, data) in enumerate(lc_results.items()):
        ax = ax5 if i == 0 else ax6
        ax.plot(data["sizes"], data["train"], "o-", color="#27ae60", lw=2, label="Train")
        ax.plot(data["sizes"], data["val"],   "s-", color="#e74c3c", lw=2, label="Val")
        ax.fill_between(data["sizes"], data["train"], data["val"],
                        alpha=0.12, color="#e74c3c")
        ax.set_title(f"Learning Curve — {label.split('[')[0].strip()}",
                     fontweight="bold", fontsize=10)
        ax.set_xlabel("Training Set Size"); ax.set_ylabel("BCE Loss")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS, "04_loss_overfitting_bv.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  [VIZ] Figure saved → {path}")
    plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "▓"*65)
    print("  Phase 1 — Topic 4: Loss Functions, Overfitting & Bias-Variance")
    print("▓"*65)

    # Dataset
    X, y = make_moons(n_samples=600, noise=0.25, random_state=SEED)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.25,
                                               stratify=y, random_state=SEED)
    sc = StandardScaler().fit(X_tr)
    X_tr = sc.transform(X_tr); X_va = sc.transform(X_va)

    def _ld(X_a, y_a, bs=32, shuffle=True):
        ds = TensorDataset(torch.tensor(X_a, dtype=torch.float32),
                           torch.tensor(y_a, dtype=torch.float32).unsqueeze(1))
        return DataLoader(ds, batch_size=bs, shuffle=shuffle)

    train_loader = _ld(X_tr, y_tr)
    val_loader   = _ld(X_va, y_va, shuffle=False)

    # Run all sections
    errors, loss_curves = demo_loss_functions()
    overfit_results     = overfitting_demo(train_loader, val_loader)
    bv_results          = bias_variance_experiment(X, y, n_trials=20)
    lc_results          = learning_curve_experiment(X, y)

    build_figures(errors, loss_curves, overfit_results, bv_results, lc_results)
    print("\n  ✓ Topic 4 complete.\n")

if __name__ == "__main__":
    main()
