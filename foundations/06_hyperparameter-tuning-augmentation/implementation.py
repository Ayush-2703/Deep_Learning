"""
Topic: Hyperparameter Tuning & Data Augmentation
===============================================================
Repository : deep-learning/foundations/06-hyperparameter-tuning-augmentation/
File       : implementation.py

Sections:
  A │ Grid Search vs Random Search — coverage comparison + best config
  B │ Learning rate schedules — step, cosine, warmup+cosine, plateau
  C │ Data Augmentation — Gaussian noise, feature dropout, Mixup
  D │ K-Fold Cross-Validation for hyperparameter selection
  E │ Visualization dashboard
"""

import os, warnings, itertools
import numpy as np
import matplotlib, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

SEED = 42; DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = "results"; os.makedirs(RESULTS, exist_ok=True)
np.random.seed(SEED); torch.manual_seed(SEED)
print(f"[CONFIG] Device: {DEVICE}")

def _build_mlp(hidden=(64,64), input_size=2):
    layers, prev = [], input_size
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers += [nn.Linear(prev, 1), nn.Sigmoid()]
    m = nn.Sequential(*layers)
    for layer in m.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)
    return m

def get_data(n=800, noise=0.25):
    X, y = make_moons(n_samples=n, noise=noise, random_state=SEED)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xva), ytr, yva, X, y, sc

# ═════════════════════════════════════════════════════════════════════════════
# SECTION A — GRID SEARCH vs RANDOM SEARCH
# ═════════════════════════════════════════════════════════════════════════════

def _quick_train_eval(lr, batch_size, hidden_size, Xtr, ytr, Xva, yva, n_epochs=60):
    """Fast training run for HPO — returns final validation accuracy."""
    torch.manual_seed(SEED)
    model = _build_mlp((hidden_size, hidden_size)).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=lr)
    crit  = nn.BCELoss()

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True)

    Xva_t = torch.tensor(Xva, dtype=torch.float32).to(DEVICE)
    yva_t = torch.tensor(yva, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    model.train()
    for _ in range(n_epochs):
        for Xb, Yb in loader:
            Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
            opt.zero_grad(); crit(model(Xb), Yb).backward(); opt.step()

    model.eval()
    with torch.no_grad():
        acc = ((model(Xva_t) >= 0.5).float() == yva_t).float().mean().item()
    return acc

def grid_vs_random_search(Xtr, ytr, Xva, yva):
    print("\n" + "="*65)
    print("SECTION A — Grid Search vs Random Search")
    print("="*65)

    lr_grid     = [1e-4, 1e-3, 1e-2]
    hidden_grid = [8, 32, 128]

    # ── Grid Search: full cartesian product (9 configs) ──────────────────────
    print("\n  [Grid Search] 3×3 = 9 configurations")
    grid_results = []
    for lr, h in itertools.product(lr_grid, hidden_grid):
        acc = _quick_train_eval(lr, 32, h, Xtr, ytr, Xva, yva)
        grid_results.append({"lr": lr, "hidden": h, "acc": acc})
        print(f"    lr={lr:.0e}  hidden={h:3d}  → acc={acc*100:.1f}%")
    best_grid = max(grid_results, key=lambda r: r["acc"])

    # ── Random Search: 9 random samples (log-uniform lr, choice hidden) ──────
    print("\n  [Random Search] 9 random configurations")
    rng = np.random.default_rng(SEED)
    random_results = []
    for _ in range(9):
        lr = float(10 ** rng.uniform(-4, -2))             # log-uniform [1e-4,1e-2]
        h  = int(rng.choice([8, 16, 32, 64, 128, 256]))   # wider discrete choice
        acc = _quick_train_eval(lr, 32, h, Xtr, ytr, Xva, yva)
        random_results.append({"lr": lr, "hidden": h, "acc": acc})
        print(f"    lr={lr:.2e}  hidden={h:3d}  → acc={acc*100:.1f}%")
    best_random = max(random_results, key=lambda r: r["acc"])

    print(f"\n  Best Grid:   lr={best_grid['lr']:.0e}, hidden={best_grid['hidden']}, "
          f"acc={best_grid['acc']*100:.1f}%")
    print(f"  Best Random: lr={best_random['lr']:.2e}, hidden={best_random['hidden']}, "
          f"acc={best_random['acc']*100:.1f}%")

    return {"grid": grid_results, "random": random_results,
            "best_grid": best_grid, "best_random": best_random}

# ═════════════════════════════════════════════════════════════════════════════
# SECTION B — LEARNING RATE SCHEDULES
# ═════════════════════════════════════════════════════════════════════════════

def lr_schedule_comparison(Xtr, ytr, Xva, yva, n_epochs=100):
    print("\n" + "="*65)
    print("SECTION B — Learning Rate Schedule Comparison")
    print("="*65)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    Xva_t = torch.tensor(Xva, dtype=torch.float32).to(DEVICE)
    yva_t = torch.tensor(yva, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=32, shuffle=True)
    crit = nn.BCELoss()

    def warmup_cosine(epoch, warmup=10, total=n_epochs, base_lr=1e-2):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    schedulers_cfg = {
        "Constant lr=1e-2": ("none", None),
        "Step Decay":        ("step", None),
        "Cosine Annealing":  ("cosine", None),
        "Warmup+Cosine":     ("warmup", None),
        "ReduceLROnPlateau": ("plateau", None),
    }

    results = {}
    for name, (sched_type, _) in schedulers_cfg.items():
        torch.manual_seed(SEED)
        model = _build_mlp((64, 64)).to(DEVICE)
        opt   = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

        if sched_type == "step":
            sched = StepLR(opt, step_size=30, gamma=0.3)
        elif sched_type == "cosine":
            sched = CosineAnnealingLR(opt, T_max=n_epochs)
        elif sched_type == "plateau":
            sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
        elif sched_type == "warmup":
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=warmup_cosine)
        else:
            sched = None

        lr_history, loss_history = [], []
        for epoch in range(n_epochs):
            model.train()
            tl, tn = 0.0, 0
            for Xb, Yb in loader:
                Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
                opt.zero_grad(); loss = crit(model(Xb), Yb)
                loss.backward(); opt.step()
                tl += loss.item()*len(Xb); tn += len(Xb)

            epoch_loss = tl / tn
            lr_history.append(opt.param_groups[0]["lr"])
            loss_history.append(epoch_loss)

            if sched_type == "plateau":
                sched.step(epoch_loss)
            elif sched is not None:
                sched.step()

        model.eval()
        with torch.no_grad():
            final_acc = ((model(Xva_t) >= 0.5).float() == yva_t).float().mean().item()

        results[name] = {"lr": lr_history, "loss": loss_history, "final_acc": final_acc}
        print(f"  {name:20s} | final_loss={loss_history[-1]:.4f} | "
              f"final_acc={final_acc*100:.1f}%")

    return results

# ═════════════════════════════════════════════════════════════════════════════
# SECTION C — DATA AUGMENTATION
# ═════════════════════════════════════════════════════════════════════════════

def gaussian_noise_augment(X: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """Inject Gaussian noise: x' = x + ε, ε~N(0,σ²)."""
    return X + torch.randn_like(X) * sigma

def feature_dropout_augment(X: torch.Tensor, p: float = 0.2) -> torch.Tensor:
    """Randomly zero out individual features (not whole samples)."""
    mask = (torch.rand_like(X) > p).float()
    return X * mask

def mixup(X: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """
    Mixup augmentation:
      λ ~ Beta(α,α);  x̃=λxᵢ+(1−λ)xⱼ;  ỹ=λyᵢ+(1−λ)yⱼ
    Returns mixed inputs and mixed (soft) targets.
    """
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(X.size(0))
    X_mixed = lam * X + (1 - lam) * X[perm]
    y_mixed = lam * y + (1 - lam) * y[perm]
    return X_mixed, y_mixed

def augmentation_comparison(X_full, y_full, n_epochs=120):
    print("\n" + "="*65)
    print("SECTION C — Data Augmentation Comparison")
    print("="*65)

    # Use a SMALL training set to make overfitting/augmentation benefits visible
    Xtr, Xva, ytr, yva = train_test_split(X_full, y_full, test_size=0.7,
                                           stratify=y_full, random_state=SEED)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)
    print(f"\n  Deliberately small train set: {len(Xtr_s)} samples "
          f"(val set: {len(Xva_s)}) to expose overfitting/augmentation gap")

    Xtr_t = torch.tensor(Xtr_s, dtype=torch.float32)
    ytr_t = torch.tensor(ytr,   dtype=torch.float32).unsqueeze(1)
    Xva_t = torch.tensor(Xva_s, dtype=torch.float32).to(DEVICE)
    yva_t = torch.tensor(yva,   dtype=torch.float32).unsqueeze(1).to(DEVICE)

    configs = ["No Augmentation", "Gaussian Noise", "Feature Dropout", "Mixup"]
    results = {}

    for cfg in configs:
        torch.manual_seed(SEED)
        model = _build_mlp((128, 128)).to(DEVICE)
        opt   = optim.Adam(model.parameters(), lr=1e-3)
        crit  = nn.BCELoss()
        loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=16, shuffle=True)

        hist = {"train": [], "val": []}
        for _ in range(n_epochs):
            model.train()
            tl, tn = 0.0, 0
            for Xb, Yb in loader:
                Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)

                if cfg == "Gaussian Noise":
                    Xb = gaussian_noise_augment(Xb, sigma=0.15)
                elif cfg == "Feature Dropout":
                    Xb = feature_dropout_augment(Xb, p=0.2)
                elif cfg == "Mixup":
                    Xb, Yb = mixup(Xb, Yb, alpha=0.4)

                opt.zero_grad(); loss = crit(model(Xb), Yb)
                loss.backward(); opt.step()
                tl += loss.item()*len(Xb); tn += len(Xb)

            model.eval()
            with torch.no_grad():
                vl = crit(model(Xva_t), yva_t).item()
            hist["train"].append(tl/tn); hist["val"].append(vl)

        model.eval()
        with torch.no_grad():
            acc = ((model(Xva_t)>=0.5).float()==yva_t).float().mean().item()
        results[cfg] = {**hist, "final_acc": acc}
        gap = hist["val"][-1] - hist["train"][-1]
        print(f"  {cfg:18s} | val_acc={acc*100:.1f}% | "
              f"train={hist['train'][-1]:.4f} | val={hist['val'][-1]:.4f} | gap={gap:+.4f}")

    return results

# ═════════════════════════════════════════════════════════════════════════════
# SECTION D — K-FOLD CROSS-VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def kfold_cv_experiment(X_full, y_full, k=5, n_epochs=80):
    print("\n" + "="*65)
    print(f"SECTION D — {k}-Fold Stratified Cross-Validation")
    print("="*65)

    candidates = [
        {"lr": 1e-3, "hidden": 16},
        {"lr": 1e-3, "hidden": 64},
        {"lr": 1e-2, "hidden": 64},
    ]

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    results = {}

    for cand in candidates:
        label = f"lr={cand['lr']:.0e}, hidden={cand['hidden']}"
        fold_accs = []

        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_full, y_full)):
            Xtr_f, Xva_f = X_full[tr_idx], X_full[va_idx]
            ytr_f, yva_f = y_full[tr_idx], y_full[va_idx]
            sc = StandardScaler().fit(Xtr_f)
            Xtr_f = sc.transform(Xtr_f); Xva_f = sc.transform(Xva_f)

            torch.manual_seed(SEED + fold_idx)
            model = _build_mlp((cand["hidden"], cand["hidden"])).to(DEVICE)
            opt   = optim.Adam(model.parameters(), lr=cand["lr"])
            crit  = nn.BCELoss()

            Xtr_t = torch.tensor(Xtr_f, dtype=torch.float32)
            ytr_t = torch.tensor(ytr_f, dtype=torch.float32).unsqueeze(1)
            loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=32, shuffle=True)
            Xva_t = torch.tensor(Xva_f, dtype=torch.float32).to(DEVICE)
            yva_t = torch.tensor(yva_f, dtype=torch.float32).unsqueeze(1).to(DEVICE)

            model.train()
            for _ in range(n_epochs):
                for Xb, Yb in loader:
                    Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
                    opt.zero_grad(); crit(model(Xb), Yb).backward(); opt.step()

            model.eval()
            with torch.no_grad():
                acc = ((model(Xva_t)>=0.5).float()==yva_t).float().mean().item()
            fold_accs.append(acc)

        mean_acc, std_acc = np.mean(fold_accs), np.std(fold_accs)
        results[label] = {"fold_accs": fold_accs, "mean": mean_acc, "std": std_acc}
        print(f"  {label:25s} | folds={[f'{a*100:.1f}%' for a in fold_accs]}")
        print(f"  {'':25s} | mean={mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

    best = max(results.items(), key=lambda kv: kv[1]["mean"])
    print(f"\n  Best config: {best[0]} (mean acc={best[1]['mean']*100:.2f}%)")
    return results

# ═════════════════════════════════════════════════════════════════════════════
# SECTION E — VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def build_figures(search_res, sched_res, aug_res, cv_res):
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("Phase 1 — Topic 6: Hyperparameter Tuning & Data Augmentation",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    a = [fig.add_subplot(gs[r,c]) for r in range(2) for c in range(3)]

    # Panel 1: Grid vs Random search coverage (scatter on lr-hidden plane)
    grid_lrs = [r["lr"] for r in search_res["grid"]]
    grid_hs  = [r["hidden"] for r in search_res["grid"]]
    rand_lrs = [r["lr"] for r in search_res["random"]]
    rand_hs  = [r["hidden"] for r in search_res["random"]]
    a[0].scatter(grid_lrs, grid_hs, s=80, c="#e74c3c", marker="s",
                label="Grid (9 pts)", alpha=0.8, edgecolors="white")
    a[0].scatter(rand_lrs, rand_hs, s=80, c="#3498db", marker="o",
                label="Random (9 pts)", alpha=0.8, edgecolors="white")
    a[0].set_xscale("log")
    a[0].set_title("Search Coverage: Grid vs Random", fontweight="bold", fontsize=10)
    a[0].set_xlabel("Learning Rate (log)"); a[0].set_ylabel("Hidden Size")
    a[0].legend(fontsize=8); a[0].grid(True, alpha=0.3)

    # Panel 2: Best accuracy comparison bar
    accs = [search_res["best_grid"]["acc"]*100, search_res["best_random"]["acc"]*100]
    bars = a[1].bar(["Best Grid", "Best Random"], accs, color=["#e74c3c", "#3498db"])
    for bar, acc in zip(bars, accs):
        a[1].text(bar.get_x()+bar.get_width()/2, acc+0.5, f"{acc:.1f}%",
                  ha="center", fontweight="bold")
    a[1].set_title("Best Found Accuracy", fontweight="bold", fontsize=10)
    a[1].set_ylabel("Val Accuracy (%)"); a[1].set_ylim(0, 105)
    a[1].grid(True, axis="y", alpha=0.3)

    # Panel 3: LR schedules over epochs
    colors_s = ["#95a5a6", "#e67e22", "#3498db", "#27ae60", "#9b59b6"]
    for (name, data), col in zip(sched_res.items(), colors_s):
        a[2].plot(data["lr"], color=col, lw=2, label=name)
    a[2].set_title("Learning Rate Schedules", fontweight="bold", fontsize=10)
    a[2].set_xlabel("Epoch"); a[2].set_ylabel("Learning Rate")
    a[2].legend(fontsize=7); a[2].grid(True, alpha=0.3)

    # Panel 4: Loss curves per schedule
    for (name, data), col in zip(sched_res.items(), colors_s):
        a[3].plot(data["loss"], color=col, lw=2, label=name)
    a[3].set_title("Training Loss by Schedule", fontweight="bold", fontsize=10)
    a[3].set_xlabel("Epoch"); a[3].set_ylabel("BCE Loss")
    a[3].legend(fontsize=7); a[3].grid(True, alpha=0.3)

    # Panel 5: Augmentation — val loss curves
    colors_a = ["#e74c3c", "#27ae60", "#3498db", "#9b59b6"]
    for (name, data), col in zip(aug_res.items(), colors_a):
        a[4].plot(data["val"], color=col, lw=2, label=f"{name} ({data['final_acc']*100:.0f}%)")
    a[4].set_title("Augmentation: Val Loss", fontweight="bold", fontsize=10)
    a[4].set_xlabel("Epoch"); a[4].set_ylabel("BCE Loss")
    a[4].legend(fontsize=7); a[4].grid(True, alpha=0.3)

    # Panel 6: K-Fold CV results with error bars
    labels = list(cv_res.keys())
    means  = [cv_res[l]["mean"]*100 for l in labels]
    stds   = [cv_res[l]["std"]*100 for l in labels]
    a[5].bar(range(len(labels)), means, yerr=stds, capsize=6,
             color=["#3498db","#27ae60","#e67e22"], alpha=0.85)
    a[5].set_xticks(range(len(labels)))
    a[5].set_xticklabels([l.replace(", ", "\n") for l in labels], fontsize=8)
    a[5].set_title("K-Fold CV: Mean ± Std Accuracy", fontweight="bold", fontsize=10)
    a[5].set_ylabel("Accuracy (%)"); a[5].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS, "06_hpo_augmentation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  [VIZ] Figure saved → {path}")
    plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "▓"*65)
    print("  Phase 1 — Topic 6: Hyperparameter Tuning & Data Augmentation")
    print("▓"*65)

    Xtr, Xva, ytr, yva, X_full, y_full, _ = get_data()

    search_res = grid_vs_random_search(Xtr, ytr, Xva, yva)
    sched_res  = lr_schedule_comparison(Xtr, ytr, Xva, yva)
    aug_res    = augmentation_comparison(X_full, y_full)
    cv_res     = kfold_cv_experiment(X_full, y_full)

    build_figures(search_res, sched_res, aug_res, cv_res)
    print("\n  ✓ Topic 6 complete.\n")

if __name__ == "__main__":
    main()
