"""
Phase 1 — Topic 5: Regularization, Optimizers, BatchNorm & Early Stopping
==========================================================================
Repository : deep-learning-mastery/phase-1-foundations/05-regularization-optimizers-batchnorm/
File       : implementation.py

Sections:
  A │ L1 vs L2 regularization — weight magnitude and sparsity comparison
  B │ Dropout — train/eval mode difference, effect on generalization
  C │ Batch Normalization — training stability and convergence speed
  D │ Optimizer comparison — SGD, SGD+Momentum, RMSprop, Adam, AdamW
  E │ Early stopping — callback implementation with best-weight restoration
  F │ Visualization dashboard
"""

import os, warnings, copy
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

# ─────────────────────────────────────────────────────────────────────────────
# DATA PIPELINE (shared across all sections)
# ─────────────────────────────────────────────────────────────────────────────
def get_loaders(n=800, noise=0.25, bs=32):
    X, y = make_moons(n_samples=n, noise=noise, random_state=SEED)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2,
                                           stratify=y, random_state=SEED)
    sc  = StandardScaler().fit(Xtr)
    Xtr = sc.transform(Xtr); Xva = sc.transform(Xva)
    def _ld(X_, y_, shuffle):
        ds = TensorDataset(torch.tensor(X_, dtype=torch.float32),
                           torch.tensor(y_, dtype=torch.float32).unsqueeze(1))
        return DataLoader(ds, batch_size=bs, shuffle=shuffle)
    return _ld(Xtr, ytr, True), _ld(Xva, yva, False)

def run_training(model, train_ld, val_ld, n_epochs=150,
                 optimizer=None, criterion=None, early_stop=None):
    """Generic training loop returning history dict."""
    if criterion is None: criterion = nn.BCELoss()
    if optimizer is None: optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model = model.to(DEVICE)
    hist  = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, n_epochs + 1):
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for Xb, Yb in train_ld:
            Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(Xb); loss = criterion(pred, Yb)
            loss.backward(); optimizer.step()
            tl += loss.item()*len(Xb); tc += ((pred>=.5)==Yb).sum().item(); tn += len(Xb)
        model.eval()
        vl, vc, vn = 0.0, 0, 0
        with torch.no_grad():
            for Xb, Yb in val_ld:
                Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
                pred = model(Xb); loss = criterion(pred, Yb)
                vl += loss.item()*len(Xb); vc += ((pred>=.5)==Yb).sum().item(); vn += len(Xb)
        hist["train_loss"].append(tl/tn); hist["val_loss"].append(vl/vn)
        hist["train_acc"].append(tc/tn);  hist["val_acc"].append(vc/vn)
        if early_stop and early_stop.step(vl/vn, model):
            early_stop.restore(model)
            print(f"    Early stop at epoch {epoch}")
            break
    return hist

# ═════════════════════════════════════════════════════════════════════════════
# SECTION A — L1 vs L2 REGULARIZATION
# ═════════════════════════════════════════════════════════════════════════════

class RegMLP(nn.Module):
    """MLP with optional L1/L2 regularization computed in forward pass."""
    def __init__(self, reg_type="none", lam=1e-3):
        super().__init__()
        self.reg_type = reg_type
        self.lam      = lam
        self.fc1 = nn.Linear(2, 64); self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.act = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

    def regularization_loss(self):
        weights = [self.fc1.weight, self.fc2.weight, self.fc3.weight]
        if self.reg_type == "l2":
            return self.lam * sum(w.pow(2).sum() for w in weights)
        elif self.reg_type == "l1":
            return self.lam * sum(w.abs().sum() for w in weights)
        return torch.tensor(0.0)

def regularization_experiment(train_ld, val_ld):
    print("\n" + "="*65)
    print("SECTION A — L1 vs L2 Regularization")
    print("="*65)
    configs = [("No Reg",   "none", 0),
               ("L2 λ=1e-3","l2",  1e-3),
               ("L1 λ=1e-4","l1",  1e-4)]
    results = {}
    for name, rtype, lam in configs:
        torch.manual_seed(SEED)
        model  = RegMLP(rtype, lam)
        opt    = optim.Adam(model.parameters(), lr=1e-3)
        bce    = nn.BCELoss()
        model  = model.to(DEVICE)
        tr_losses, va_losses = [], []

        for _ in range(150):
            model.train()
            tl, tn = 0.0, 0
            for Xb, Yb in train_ld:
                Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
                opt.zero_grad()
                pred = model(Xb)
                loss = bce(pred, Yb) + model.regularization_loss()
                loss.backward(); opt.step()
                tl += loss.item()*len(Xb); tn += len(Xb)
            model.eval()
            vl, vn = 0.0, 0
            with torch.no_grad():
                for Xb, Yb in val_ld:
                    Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
                    vl += bce(model(Xb), Yb).item()*len(Xb); vn += len(Xb)
            tr_losses.append(tl/tn); va_losses.append(vl/vn)

        # Collect weight distribution
        w_all = torch.cat([p.data.cpu().flatten() for p in model.parameters()
                           if p.dim() > 1]).numpy()
        sparsity = float((np.abs(w_all) < 1e-3).mean())
        results[name] = {"train": tr_losses, "val": va_losses,
                         "weights": w_all, "sparsity": sparsity}
        print(f"  {name:15s} | val_loss={va_losses[-1]:.4f} | "
              f"weight_std={w_all.std():.4f} | near-zero={sparsity*100:.1f}%")
    return results

# ═════════════════════════════════════════════════════════════════════════════
# SECTION B — DROPOUT
# ═════════════════════════════════════════════════════════════════════════════

class DropoutMLP(nn.Module):
    """MLP with configurable dropout rate."""
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, 1), nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x)

def dropout_experiment(train_ld, val_ld):
    print("\n" + "="*65)
    print("SECTION B — Dropout Comparison")
    print("="*65)
    configs = [("No Dropout (p=0)", 0.0),
               ("Dropout p=0.3",    0.3),
               ("Dropout p=0.5",    0.5)]
    results = {}
    for name, p in configs:
        torch.manual_seed(SEED)
        model = DropoutMLP(p)
        hist  = run_training(model, train_ld, val_ld, n_epochs=150)
        results[name] = hist
        print(f"  {name:22s} | train={hist['train_loss'][-1]:.4f} "
              f"| val={hist['val_loss'][-1]:.4f} "
              f"| gap={hist['val_loss'][-1]-hist['train_loss'][-1]:+.4f}")
    return results

# ═════════════════════════════════════════════════════════════════════════════
# SECTION C — BATCH NORMALIZATION
# ═════════════════════════════════════════════════════════════════════════════

class BNModel(nn.Module):
    """MLP with BatchNorm after each linear layer."""
    def __init__(self, use_bn: bool = True):
        super().__init__()
        layers = [nn.Linear(2, 128)]
        if use_bn: layers.append(nn.BatchNorm1d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 128))
        if use_bn: layers.append(nn.BatchNorm1d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 1)); layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x)

def batchnorm_experiment(train_ld, val_ld):
    print("\n" + "="*65)
    print("SECTION C — Batch Normalization Effect")
    print("="*65)
    results = {}
    for use_bn, name in [(False, "Without BatchNorm"), (True, "With BatchNorm")]:
        torch.manual_seed(SEED)
        model = BNModel(use_bn)
        # Use higher LR to show BatchNorm's stabilizing effect
        opt   = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        hist  = run_training(model, train_ld, val_ld, n_epochs=100, optimizer=opt)
        results[name] = hist
        print(f"  {name:20s} | final_val_loss={hist['val_loss'][-1]:.4f} "
              f"| final_val_acc={hist['val_acc'][-1]*100:.1f}%")
    return results

# ═════════════════════════════════════════════════════════════════════════════
# SECTION D — OPTIMIZER COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

def optimizer_comparison(train_ld, val_ld):
    print("\n" + "="*65)
    print("SECTION D — Optimizer Comparison")
    print("="*65)

    def _fresh():
        torch.manual_seed(SEED)
        m = nn.Sequential(nn.Linear(2,64), nn.ReLU(),
                          nn.Linear(64,64), nn.ReLU(),
                          nn.Linear(64,1), nn.Sigmoid())
        for layer in m.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)
        return m

    configs = [
        ("SGD",           lambda p: optim.SGD(p, lr=0.1)),
        ("SGD+Momentum",  lambda p: optim.SGD(p, lr=0.1, momentum=0.9)),
        ("RMSprop",       lambda p: optim.RMSprop(p, lr=1e-3)),
        ("Adam",          lambda p: optim.Adam(p, lr=1e-3)),
        ("AdamW",         lambda p: optim.AdamW(p, lr=1e-3, weight_decay=1e-2)),
    ]
    colors = ["#95a5a6", "#e67e22", "#3498db", "#27ae60", "#9b59b6"]
    results = {}
    print(f"\n  {'Optimizer':15s} | {'Val Acc':>8} | {'Val Loss':>9}")
    print("  " + "─"*38)
    for (name, opt_fn), col in zip(configs, colors):
        model = _fresh()
        opt   = opt_fn(model.parameters())
        hist  = run_training(model, train_ld, val_ld, n_epochs=100, optimizer=opt)
        results[name] = {**hist, "color": col}
        print(f"  {name:15s} | {hist['val_acc'][-1]*100:7.2f}%  | {hist['val_loss'][-1]:.5f}")
    return results

# ═════════════════════════════════════════════════════════════════════════════
# SECTION E — EARLY STOPPING
# ═════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Monitor validation loss. Stop training when it stops improving.
    Restores best model weights automatically.
    """
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None
        self.stopped_epoch = 0

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter    = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        """Load the best recorded weights back into model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

def early_stopping_experiment(train_ld, val_ld):
    print("\n" + "="*65)
    print("SECTION E — Early Stopping")
    print("="*65)
    results = {}

    for name, use_es in [("No Early Stop", False), ("Early Stop (p=15)", True)]:
        torch.manual_seed(SEED)
        model = nn.Sequential(nn.Linear(2,128), nn.ReLU(),
                              nn.Linear(128,128), nn.ReLU(),
                              nn.Linear(128,1), nn.Sigmoid())
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        es   = EarlyStopping(patience=15) if use_es else None
        hist = run_training(model, train_ld, val_ld, n_epochs=300,
                            early_stop=es)
        results[name] = {"history": hist,
                         "stopped": len(hist["train_loss"])}
        best_val = min(hist["val_loss"])
        final_val= hist["val_loss"][-1]
        print(f"  {name:22s} | epochs={len(hist['train_loss']):3d} "
              f"| best_val={best_val:.4f} | final_val={final_val:.4f}")
    return results

# ═════════════════════════════════════════════════════════════════════════════
# SECTION F — VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def build_figures(reg_res, drop_res, bn_res, opt_res, es_res):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Phase 1 — Topic 5: Regularization, Optimizers, BatchNorm & Early Stopping",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(3)]
    a = axes

    # ── Weight distribution histogram (L1 vs L2) ─────────────────────────────
    for name, data in reg_res.items():
        a[0].hist(data["weights"], bins=60, alpha=0.55, density=True, label=name)
    a[0].set_title("Weight Distributions (Reg Types)", fontweight="bold", fontsize=10)
    a[0].set_xlabel("Weight value"); a[0].set_ylabel("Density")
    a[0].legend(fontsize=7); a[0].set_xlim(-0.5, 0.5)

    # ── Reg val loss curves ───────────────────────────────────────────────────
    colors_r = ["#95a5a6", "#27ae60", "#e74c3c"]
    for (name, data), col in zip(reg_res.items(), colors_r):
        ep = range(1, len(data["val"])+1)
        a[1].plot(ep, data["train"], col, lw=1.5, ls="-",  alpha=0.6)
        a[1].plot(ep, data["val"],   col, lw=2.0, ls="--", label=name)
    a[1].set_title("Regularization: Val Loss (--) / Train Loss (-)", fontweight="bold", fontsize=10)
    a[1].set_xlabel("Epoch"); a[1].set_ylabel("BCE Loss")
    a[1].legend(fontsize=8); a[1].grid(True, alpha=0.3)

    # ── Dropout val loss ──────────────────────────────────────────────────────
    colors_d = ["#e74c3c", "#27ae60", "#9b59b6"]
    for (name, hist), col in zip(drop_res.items(), colors_d):
        ep = range(1, len(hist["val_loss"])+1)
        a[2].plot(ep, hist["val_loss"], lw=2, color=col, label=name)
    a[2].set_title("Dropout: Validation Loss", fontweight="bold", fontsize=10)
    a[2].set_xlabel("Epoch"); a[2].set_ylabel("BCE Loss")
    a[2].legend(fontsize=8); a[2].grid(True, alpha=0.3)

    # ── BatchNorm: training loss ──────────────────────────────────────────────
    for (name, hist), col in zip(bn_res.items(), ["#e74c3c", "#27ae60"]):
        ep = range(1, len(hist["train_loss"])+1)
        a[3].plot(ep, hist["train_loss"], lw=2, color=col, label=name)
    a[3].set_title("BatchNorm: Training Loss Convergence", fontweight="bold", fontsize=10)
    a[3].set_xlabel("Epoch"); a[3].set_ylabel("BCE Loss")
    a[3].legend(fontsize=9); a[3].grid(True, alpha=0.3)

    # ── BatchNorm: val accuracy ────────────────────────────────────────────────
    for (name, hist), col in zip(bn_res.items(), ["#e74c3c", "#27ae60"]):
        ep = range(1, len(hist["val_acc"])+1)
        a[4].plot(ep, [v*100 for v in hist["val_acc"]], lw=2, color=col, label=name)
    a[4].set_title("BatchNorm: Val Accuracy", fontweight="bold", fontsize=10)
    a[4].set_xlabel("Epoch"); a[4].set_ylabel("Accuracy (%)")
    a[4].legend(fontsize=9); a[4].grid(True, alpha=0.3)

    # ── Optimizer val loss ───────────────────────────────────────────────────
    for name, data in opt_res.items():
        ep = range(1, len(data["val_loss"])+1)
        a[5].plot(ep, data["val_loss"], lw=2, color=data["color"], label=name, alpha=0.9)
    a[5].set_title("Optimizer Comparison: Val Loss", fontweight="bold", fontsize=10)
    a[5].set_xlabel("Epoch"); a[5].set_ylabel("BCE Loss")
    a[5].legend(fontsize=8); a[5].grid(True, alpha=0.3)

    # ── Optimizer val accuracy ────────────────────────────────────────────────
    for name, data in opt_res.items():
        ep = range(1, len(data["val_acc"])+1)
        a[6].plot(ep, [v*100 for v in data["val_acc"]], lw=2,
                  color=data["color"], label=name, alpha=0.9)
    a[6].set_title("Optimizer Comparison: Val Accuracy", fontweight="bold", fontsize=10)
    a[6].set_xlabel("Epoch"); a[6].set_ylabel("Accuracy (%)")
    a[6].legend(fontsize=8); a[6].grid(True, alpha=0.3)

    # ── Early Stopping val loss ────────────────────────────────────────────────
    for (name, data), col in zip(es_res.items(), ["#e74c3c", "#27ae60"]):
        hist = data["history"]
        ep   = range(1, len(hist["val_loss"])+1)
        a[7].plot(ep, hist["val_loss"], lw=2, color=col, label=f"{name} ({len(ep)} ep)")
    a[7].set_title("Early Stopping: Val Loss", fontweight="bold", fontsize=10)
    a[7].set_xlabel("Epoch"); a[7].set_ylabel("BCE Loss")
    a[7].legend(fontsize=8); a[7].grid(True, alpha=0.3)

    # ── Early stopping train vs val for no-early-stop model ──────────────────
    hist_no_es = es_res["No Early Stop"]["history"]
    ep = range(1, len(hist_no_es["train_loss"])+1)
    a[8].plot(ep, hist_no_es["train_loss"], lw=2, color="#27ae60", label="Train")
    a[8].plot(ep, hist_no_es["val_loss"],   lw=2, color="#e74c3c", label="Val")
    a[8].fill_between(ep, hist_no_es["train_loss"], hist_no_es["val_loss"],
                      alpha=0.15, color="#e74c3c")
    a[8].set_title("Without Early Stop: Overfitting Gap", fontweight="bold", fontsize=10)
    a[8].set_xlabel("Epoch"); a[8].set_ylabel("BCE Loss")
    a[8].legend(fontsize=9); a[8].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS, "05_regularization_optimizers_bn.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  [VIZ] Figure saved → {path}")
    plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "▓"*65)
    print("  Phase 1 — Topic 5: Regularization, Optimizers, BatchNorm, Early Stopping")
    print("▓"*65)
    train_ld, val_ld = get_loaders()
    reg_res  = regularization_experiment(train_ld, val_ld)
    drop_res = dropout_experiment(train_ld, val_ld)
    bn_res   = batchnorm_experiment(train_ld, val_ld)
    opt_res  = optimizer_comparison(train_ld, val_ld)
    es_res   = early_stopping_experiment(train_ld, val_ld)
    build_figures(reg_res, drop_res, bn_res, opt_res, es_res)
    print("\n  ✓ Topic 5 complete.\n")

if __name__ == "__main__":
    main()
