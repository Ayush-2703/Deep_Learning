"""
Topic: Activation Functions
========================================
Repository : deep-learning-mastery/phase-1-foundations/02-activation-functions/
File       : implementation.py
Framework  : PyTorch 2.x | NumPy | scikit-learn | matplotlib
Python     : 3.10+

Run:
    pip install torch numpy scikit-learn matplotlib
    python implementation.py

What this file demonstrates
────────────────────────────
  SECTION A │ NumPy implementations of 10 activation functions + derivatives
  SECTION B │ PyTorch nn.Module implementations (factory pattern)
  SECTION C │ Gradient flow experiment — vanishing gradient visualization
  SECTION D │ Training comparison — all activations on make_circles
  SECTION E │ Dead neuron analysis — ReLU vs Leaky ReLU under high LR
  SECTION F │ Softmax numerical stability demo
  SECTION G │ Comprehensive visualization dashboard (3 figures)
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SEED       = 42
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS    = "results"
os.makedirs(RESULTS, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"[CONFIG] Device  : {DEVICE}")
print(f"[CONFIG] PyTorch : {torch.__version__}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION A — NumPy Activation Functions
# Explicit implementations for visualization and educational clarity.
# All classes expose: forward(z), derivative(z), name, color
# ═════════════════════════════════════════════════════════════════════════════

class ActivationFn:
    """Abstract base for NumPy activation functions."""
    name:  str = "base"
    color: str = "#000000"

    def forward(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def derivative(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.forward(z)

    def __repr__(self) -> str:
        return self.name


class StepFn(ActivationFn):
    """Heaviside step: H(z) = 1 if z≥0 else 0. Gradient = 0 → not trainable."""
    name  = "Step"
    color = "#95a5a6"

    def forward(self, z):
        return np.where(z >= 0, 1.0, 0.0)

    def derivative(self, z):
        return np.zeros_like(z)          # undefined at z=0; 0 elsewhere


class SigmoidFn(ActivationFn):
    """σ(z) = 1/(1+e^{-z}). Range (0,1). Max derivative = 0.25."""
    name  = "Sigmoid"
    color = "#e74c3c"

    def forward(self, z):
        # Two-branch stable implementation: avoid exp(-z) overflow for z→-∞
        #   Branch z≥0: 1/(1+exp(-z))          — no overflow
        #   Branch z<0: exp(z)/(1+exp(z))       — no overflow (exp(z)→0)
        return np.where(
            z >= 0,
            1.0 / (1.0 + np.exp(-np.abs(z))),
            np.exp(z) / (1.0 + np.exp(z))
        )

    def derivative(self, z):
        s = self.forward(z)
        return s * (1.0 - s)             # σ'(z) = σ(z)(1-σ(z)) ≤ 0.25


class TanhFn(ActivationFn):
    """tanh(z). Range (-1,1). Max derivative = 1.0 at z=0. Zero-centered."""
    name  = "Tanh"
    color = "#e67e22"

    def forward(self, z):
        return np.tanh(z)

    def derivative(self, z):
        return 1.0 - np.tanh(z) ** 2    # tanh'(z) = 1 - tanh²(z)


class ReLUFn(ActivationFn):
    """ReLU: max(0,z). Gradient = 1 for z>0 (no vanishing), 0 for z≤0 (dead neurons)."""
    name  = "ReLU"
    color = "#27ae60"

    def forward(self, z):
        return np.maximum(0.0, z)

    def derivative(self, z):
        return (z > 0).astype(float)     # 1 if positive, 0 otherwise


class LeakyReLUFn(ActivationFn):
    """Leaky ReLU: max(αz, z). Fixes dead neurons — gradient always ≥ α."""
    name  = "Leaky ReLU"
    color = "#2ecc71"

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def forward(self, z):
        return np.where(z > 0, z, self.alpha * z)

    def derivative(self, z):
        return np.where(z > 0, 1.0, self.alpha)


class ELUFn(ActivationFn):
    """ELU: z for z>0, α(e^z - 1) for z≤0. Smooth, no dead neurons, ~zero-centered."""
    name  = "ELU"
    color = "#3498db"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def forward(self, z):
        # np.expm1(z) = e^z - 1 (numerically stable for small z)
        return np.where(z > 0, z, self.alpha * np.expm1(z))

    def derivative(self, z):
        # d/dz α(e^z-1) = αe^z = ELU(z) + α (useful identity)
        return np.where(z > 0, 1.0, self.forward(z) + self.alpha)


class SELUFn(ActivationFn):
    """SELU: λ·ELU(z, α). Self-normalizing: E[out]≈0, Var[out]≈1 when in~N(0,1)."""
    name  = "SELU"
    color = "#2980b9"
    # Fixed-point constants derived by solving E[SELU(z)]=0, Var[SELU(z)]=1
    LAMBDA = 1.0507009873554804934193349852946
    ALPHA  = 1.6732632423543772848170429916717

    def forward(self, z):
        return self.LAMBDA * np.where(z > 0, z, self.ALPHA * np.expm1(z))

    def derivative(self, z):
        return self.LAMBDA * np.where(z > 0, 1.0, self.ALPHA * np.exp(z))


class GELUFn(ActivationFn):
    """GELU: z·Φ(z). Smooth stochastic gating. Standard in BERT, GPT."""
    name  = "GELU"
    color = "#9b59b6"
    _K    = np.sqrt(2.0 / np.pi)   # constant precomputed once

    def forward(self, z):
        # Tanh approximation (Hendrycks & Gimpel 2016 — used in BERT)
        return 0.5 * z * (1.0 + np.tanh(self._K * (z + 0.044715 * z ** 3)))

    def derivative(self, z):
        # Numerical gradient — analytic form is verbose
        eps = 1e-5
        return (self.forward(z + eps) - self.forward(z - eps)) / (2.0 * eps)


class SwishFn(ActivationFn):
    """SiLU/Swish: z·σ(z). Non-monotonic, smooth. Used in EfficientNet."""
    name  = "SiLU/Swish"
    color = "#8e44ad"

    def forward(self, z):
        sig = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))   # stable sigmoid
        return z * sig

    def derivative(self, z):
        # f'(z) = σ(z) + z·σ(z)·(1-σ(z)) = σ(z)·(1 + z·(1-σ(z)))
        sig = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        return sig * (1.0 + z * (1.0 - sig))


class MishFn(ActivationFn):
    """Mish: z·tanh(softplus(z)). Smooth, bounded below ~-0.31. Used in YOLOv4."""
    name  = "Mish"
    color = "#c0392b"

    def forward(self, z):
        # np.log1p(np.exp(z)) = softplus(z) but log1p avoids overflow
        sp = np.log1p(np.exp(np.clip(z, -500, 20)))   # clip avoids exp overflow
        return z * np.tanh(sp)

    def derivative(self, z):
        eps = 1e-5
        return (self.forward(z + eps) - self.forward(z - eps)) / (2.0 * eps)


# ── Collection for visualization ─────────────────────────────────────────────
ALL_NUMPY_ACTS = [
    StepFn(), SigmoidFn(), TanhFn(),
    ReLUFn(), LeakyReLUFn(0.01), ELUFn(),
    SELUFn(), GELUFn(), SwishFn(), MishFn()
]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION B — PyTorch Implementations
# Factory pattern: each key maps to a callable that returns a FRESH nn.Module.
# We NEVER share module instances between networks — modules have state.
# ═════════════════════════════════════════════════════════════════════════════

class MishModule(nn.Module):
    """Mish activation as nn.Module: f(z) = z · tanh(softplus(z))."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


# Each value is a zero-argument callable returning a fresh nn.Module instance
TORCH_ACT_FACTORY = {
    "Sigmoid":    lambda: nn.Sigmoid(),
    "Tanh":       lambda: nn.Tanh(),
    "ReLU":       lambda: nn.ReLU(),
    "Leaky ReLU": lambda: nn.LeakyReLU(negative_slope=0.01),
    "ELU":        lambda: nn.ELU(alpha=1.0),
    "SELU":       lambda: nn.SELU(),
    "GELU":       lambda: nn.GELU(),
    "SiLU/Swish": lambda: nn.SiLU(),
    "Mish":       lambda: MishModule(),
}

# Consistent colors across all figures
PLOT_COLORS = {
    "Sigmoid":    "#e74c3c",
    "Tanh":       "#e67e22",
    "ReLU":       "#27ae60",
    "Leaky ReLU": "#2ecc71",
    "ELU":        "#3498db",
    "SELU":       "#2980b9",
    "GELU":       "#9b59b6",
    "SiLU/Swish": "#8e44ad",
    "Mish":       "#c0392b",
}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION C — Gradient Flow Experiment
# Deep network (15 layers) with different activations.
# Track gradient L2-norm at each layer to reveal vanishing/exploding gradients.
# ═════════════════════════════════════════════════════════════════════════════

def _build_deep_net(n_layers: int, n_hidden: int,
                    act_factory, init_std: float) -> nn.Module:
    """
    Build a deep fully-connected network.

    Architecture:
        Linear(n_hidden, n_hidden) → Activation
        ... repeated n_layers times ...
        Linear(n_hidden, 1) → Sigmoid

    All Linear weights initialized with N(0, init_std²).
    """
    layers = []
    for _ in range(n_layers):
        layers.append(nn.Linear(n_hidden, n_hidden, bias=True))
        layers.append(act_factory())          # fresh activation instance
    layers.append(nn.Linear(n_hidden, 1))
    layers.append(nn.Sigmoid())

    model = nn.Sequential(*layers)

    # Custom weight init — uniform std allows vanishing/exploding comparison
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=init_std)
            nn.init.zeros_(m.bias)

    return model


def gradient_flow_experiment(n_layers: int = 15,
                              n_hidden: int = 64) -> dict:
    """
    For each activation:
      1. Build deep network with std = 1/sqrt(n_hidden) weight init
      2. Forward pass random data
      3. Backward pass
      4. Collect gradient L2-norm from every Linear layer

    Returns dict: name → list of grad norms (one per Linear layer)
    """
    print("\n" + "="*65)
    print("SECTION C — Gradient Flow Experiment")
    print(f"  Depth={n_layers} layers | Width={n_hidden} | "
          f"init_std=1/√{n_hidden}")
    print("="*65)

    # Weight std calibrated so z ~ N(0,1) at init → triggers sigmoid saturation
    init_std = 1.0 / np.sqrt(n_hidden)

    # Fixed batch — same for all activations
    torch.manual_seed(SEED)
    X_rnd = torch.randn(128, n_hidden)
    y_rnd = (torch.rand(128, 1) > 0.5).float()
    criterion = nn.BCELoss()

    # Test these activations for gradient flow
    test_acts = {
        "Sigmoid":    lambda: nn.Sigmoid(),
        "Tanh":       lambda: nn.Tanh(),
        "ReLU":       lambda: nn.ReLU(),
        "Leaky ReLU": lambda: nn.LeakyReLU(0.01),
        "ELU":        lambda: nn.ELU(),
        "GELU":       lambda: nn.GELU(),
    }

    results = {}

    for name, factory in test_acts.items():
        torch.manual_seed(SEED)
        model = _build_deep_net(n_layers, n_hidden, factory, init_std)

        # Single forward + backward
        pred = model(X_rnd)
        loss = criterion(pred, y_rnd)
        loss.backward()

        # Collect gradient norms — only from Linear layers (not activations)
        grad_norms = []
        for m in model.modules():
            if isinstance(m, nn.Linear) and m.weight.grad is not None:
                grad_norms.append(m.weight.grad.norm().item())

        results[name] = grad_norms

        ratio = grad_norms[0] / (grad_norms[-1] + 1e-20)
        print(f"  {name:15s} | input-layer norm: {grad_norms[0]:.2e} | "
              f"output-layer norm: {grad_norms[-1]:.2e} | "
              f"ratio: {ratio:.1e}")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# SECTION D — Data Pipeline + Training Comparison
# Same architecture, same dataset, different activation → compare learning.
# Dataset: make_circles (concentric rings — non-linearly separable)
# ═════════════════════════════════════════════════════════════════════════════

def build_data_pipeline(n_samples: int = 1000, noise: float = 0.1,
                        test_size: float = 0.2, batch_size: int = 32):
    """Generate make_circles, standardize, wrap in DataLoaders."""
    X, y = make_circles(n_samples=n_samples, noise=noise,
                        factor=0.5, random_state=SEED)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_va   = scaler.transform(X_va)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_va_t, y_va_t),
                              batch_size=batch_size, shuffle=False)

    print(f"\n  Dataset: make_circles | n={n_samples} | "
          f"noise={noise} | factor=0.5")
    print(f"  Train: {len(X_tr_t)}  Val: {len(X_va_t)}")

    return train_loader, val_loader, X_tr_t, X_va_t, y_tr_t, y_va_t


def _build_classifier(act_factory, hidden: int = 64) -> nn.Module:
    """
    3-layer MLP for binary classification.
    Architecture: Linear(2,h) → Act → Linear(h,h) → Act → Linear(h,1) → Sigmoid
    """
    return nn.Sequential(
        nn.Linear(2, hidden),
        act_factory(),                         # fresh instance
        nn.Linear(hidden, hidden),
        act_factory(),                         # another fresh instance
        nn.Linear(hidden, 1),
        nn.Sigmoid()
    )


def _init_model(model: nn.Module, act_name: str) -> None:
    """Apply correct weight initialization per activation family."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if act_name == "SELU":
                # Lecun normal: required for SELU self-normalization
                nn.init.kaiming_normal_(m.weight, mode="fan_in",
                                        nonlinearity="linear")
            elif act_name in ("Sigmoid", "Tanh"):
                nn.init.xavier_uniform_(m.weight)    # Glorot for sigmoid/tanh
            else:
                nn.init.xavier_uniform_(m.weight)    # Xavier for ReLU-family
            nn.init.zeros_(m.bias)


def _train_single(name: str, factory,
                  train_loader: DataLoader,
                  val_loader:   DataLoader,
                  n_epochs:     int = 100,
                  hidden:       int = 64) -> tuple:
    """
    Train one classifier configuration.

    Returns (history dict, trained model).
    history keys: train_loss, val_loss, train_acc, val_acc
    """
    torch.manual_seed(SEED)
    model = _build_classifier(factory, hidden=hidden).to(DEVICE)
    _init_model(model, name)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  []}

    for epoch in range(n_epochs):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            tl += loss.item() * len(Xb)
            tc += ((pred >= 0.5).float() == yb).sum().item()
            tn += len(Xb)

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        vl, vc, vn = 0.0, 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                pred = model(Xb)
                loss = criterion(pred, yb)
                vl += loss.item() * len(Xb)
                vc += ((pred >= 0.5).float() == yb).sum().item()
                vn += len(Xb)

        history["train_loss"].append(tl / tn)
        history["val_loss"  ].append(vl / vn)
        history["train_acc" ].append(tc / tn)
        history["val_acc"   ].append(vc / vn)

    return history, model


def run_training_comparison(train_loader: DataLoader,
                             val_loader:   DataLoader,
                             n_epochs: int = 100) -> tuple:
    """Train each activation and collect histories and final models."""
    print("\n" + "="*65)
    print("SECTION D — Training Comparison: All Activations on make_circles")
    print("="*65)
    print(f"\n  {'Activation':15s} | {'Val Acc':>9} | {'Val Loss':>9}")
    print("  " + "─"*38)

    all_hist   = {}
    all_models = {}

    for name, factory in TORCH_ACT_FACTORY.items():
        hist, model = _train_single(name, factory, train_loader, val_loader,
                                    n_epochs=n_epochs)
        all_hist[name]   = hist
        all_models[name] = model

        vacc  = hist["val_acc"][-1]  * 100
        vloss = hist["val_loss"][-1]
        print(f"  {name:15s} | {vacc:8.2f}%  | {vloss:.5f}")

    return all_hist, all_models


# ═════════════════════════════════════════════════════════════════════════════
# SECTION E — Dead Neuron Analysis
# Train ReLU networks under different conditions.
# Use forward hooks to capture per-neuron activation statistics.
# Measure fraction of neurons that output 0 for EVERY training sample.
# ═════════════════════════════════════════════════════════════════════════════

def dead_neuron_analysis(train_loader: DataLoader) -> dict:
    """
    Demonstrate the dead ReLU neuron problem under high learning rates.

    Three configurations:
      1. ReLU + lr=0.5  (high lr → many dead neurons expected)
      2. ReLU + lr=1e-3 (proper lr → few/no dead neurons)
      3. Leaky ReLU + lr=0.5 (high lr but no dead neurons by definition)
    """
    print("\n" + "="*65)
    print("SECTION E — Dead Neuron Analysis")
    print("="*65)

    configs = [
        ("ReLU (lr=0.5  — high)",     lambda: nn.ReLU(),           0.5),
        ("ReLU (lr=0.001 — proper)",  lambda: nn.ReLU(),           0.001),
        ("Leaky ReLU (lr=0.5)",       lambda: nn.LeakyReLU(0.01),  0.5),
    ]

    results = {}
    N_HIDDEN = 128    # wider to make the statistics meaningful

    for label, factory, lr in configs:
        torch.manual_seed(SEED)

        # Build model: 2 hidden layers with the specified activation
        model = nn.Sequential(
            nn.Linear(2, N_HIDDEN), factory(),
            nn.Linear(N_HIDDEN, N_HIDDEN), factory(),
            nn.Linear(N_HIDDEN, 1), nn.Sigmoid()
        ).to(DEVICE)

        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        optimizer = optim.SGD(model.parameters(), lr=lr)  # SGD, not Adam
        criterion = nn.BCELoss()

        # Train for 30 epochs
        model.train()
        for _ in range(30):
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                criterion(model(Xb), yb).backward()
                optimizer.step()

        # ── Register hooks on activation layers ───────────────────────────────
        # Each activation layer accumulates its outputs across all batches
        model.eval()
        layer_outputs = []   # list of lists: one inner list per activation layer

        def make_hook(store: list):
            """Returns a hook that appends outputs to `store`."""
            def hook(module, inp, out):
                store.append(out.detach().cpu())
            return hook

        hooks = []
        for m in model.modules():
            if isinstance(m, (nn.ReLU, nn.LeakyReLU)):
                bucket = []
                layer_outputs.append(bucket)
                hooks.append(m.register_forward_hook(make_hook(bucket)))

        with torch.no_grad():
            for Xb, _ in train_loader:
                model(Xb.to(DEVICE))      # forward pass triggers hooks

        for h in hooks:
            h.remove()

        # ── Count dead neurons ─────────────────────────────────────────────
        # A neuron is dead if its output == 0.0 for EVERY sample
        total_dead = 0
        total_n    = 0
        per_layer  = []

        for bucket in layer_outputs:
            if bucket:
                combined  = torch.cat(bucket, dim=0)   # (N_total, n_hidden)
                # Exactly 0 for all samples → dead
                dead_mask = (combined == 0.0).all(dim=0)
                n_dead    = dead_mask.sum().item()
                n_total   = combined.shape[1]
                total_dead += n_dead
                total_n    += n_total
                per_layer.append((n_dead, n_total))

        dead_pct = (total_dead / total_n * 100) if total_n > 0 else 0.0

        results[label] = {
            "dead_pct":   dead_pct,
            "n_dead":     total_dead,
            "n_total":    total_n,
            "per_layer":  per_layer,
        }

        print(f"\n  {label}")
        for i, (nd, nt) in enumerate(per_layer):
            print(f"    Layer {i+1}: {nd}/{nt} dead ({nd/nt*100:.1f}%)")
        print(f"    Total  : {total_dead}/{total_n} dead ({dead_pct:.1f}%)")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# SECTION F — Softmax Numerical Stability
# ═════════════════════════════════════════════════════════════════════════════

def softmax_stability_demo() -> None:
    """
    Show why naive softmax fails on large inputs and how subtracting max fixes it.
    Also demonstrates PyTorch's nn.CrossEntropyLoss which handles this internally.
    """
    print("\n" + "="*65)
    print("SECTION F — Softmax Numerical Stability")
    print("="*65)

    z = np.array([1000.0, 1001.0, 1002.0])

    print(f"\n  Input logits: {z}\n")

    # ── Naïve: overflows to inf → NaN ────────────────────────────────────────
    exp_naive = np.exp(z)                     # [inf, inf, inf]
    naive     = exp_naive / exp_naive.sum()   # inf/inf = NaN
    print(f"  Naïve exp:   {exp_naive}")
    print(f"  Naïve softmax: {naive}  ← NaN! Broken.\n")

    # ── Stable: subtract max before exponentiation ────────────────────────────
    z_shift   = z - z.max()                   # [-2, -1, 0]
    exp_stable = np.exp(z_shift)
    stable     = exp_stable / exp_stable.sum()
    print(f"  Shifted z = z - max(z): {z_shift}")
    print(f"  Stable exp:   {np.round(exp_stable, 5)}")
    print(f"  Stable softmax: {np.round(stable, 5)}  (sums to {stable.sum():.6f})")

    # ── PyTorch BCELoss vs CrossEntropyLoss ────────────────────────────────────
    print("\n  PyTorch advice:")
    print("  ✓ Use nn.CrossEntropyLoss()  — combines LogSoftmax + NLLLoss stably")
    print("  ✗ Never: nn.Softmax() → nn.NLLLoss()  — numerically unstable")
    print("  ✗ Never: manual log(softmax(x))         — use F.log_softmax instead")

    # Demonstrate in PyTorch
    logits = torch.tensor([1000.0, 1001.0, 1002.0])
    target = torch.tensor(2)     # class 2 is correct

    # Wrong way: explicit softmax then log → NaN
    wrong = torch.log(torch.softmax(logits, dim=0))
    print(f"\n  Wrong (log+softmax): {wrong}")

    # Right way: log_softmax → stable
    right = F.log_softmax(logits, dim=0)
    print(f"  Right (log_softmax): {right.round(decimals=4)}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION G — Visualization
# ═════════════════════════════════════════════════════════════════════════════

def _plot_fn_and_derivatives(ax_fn: plt.Axes, ax_d: plt.Axes) -> None:
    """Panel pair: f(z) curves and f'(z) curves for all numpy activations."""
    z = np.linspace(-4.0, 4.0, 600)

    for fn in ALL_NUMPY_ACTS:
        ls = "--" if isinstance(fn, StepFn) else "-"
        lw = 1.5  if isinstance(fn, StepFn) else 2.0

        ax_fn.plot(z, fn.forward(z),    color=fn.color, lw=lw,
                   ls=ls, alpha=0.85, label=fn.name)
        ax_d.plot( z, fn.derivative(z), color=fn.color, lw=lw,
                   ls=ls, alpha=0.85, label=fn.name)

    for ax in (ax_fn, ax_d):
        ax.axhline(0, color="black", lw=0.5, alpha=0.4)
        ax.axvline(0, color="black", lw=0.5, alpha=0.4)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("z (pre-activation)", fontsize=9)

    ax_fn.set_ylim(-2.0, 3.8)
    ax_fn.set_title("Activation Functions  f(z)", fontweight="bold", fontsize=10)
    ax_fn.set_ylabel("f(z)", fontsize=9)
    ax_fn.legend(fontsize=7, loc="upper left", ncol=2, framealpha=0.7)

    ax_d.set_ylim(-0.15, 2.0)
    ax_d.set_title("Derivatives  f'(z)", fontweight="bold", fontsize=10)
    ax_d.set_ylabel("f'(z)", fontsize=9)
    ax_d.legend(fontsize=7, loc="upper right", ncol=2, framealpha=0.7)


def _plot_gradient_flow(ax: plt.Axes, grad_results: dict) -> None:
    """Panel: gradient norm per layer (reversed so input=left)."""
    for name, norms in grad_results.items():
        color   = PLOT_COLORS.get(name, "#333333")
        n       = len(norms)
        # Index 0 = output layer, index n-1 = input layer
        # Reverse so x-axis reads: input (left) → output (right)
        x_vals  = list(range(1, n + 1))
        y_vals  = list(reversed(norms))          # input layer on left

        ax.semilogy(x_vals, y_vals, color=color, lw=2,
                    marker="o", ms=4, label=name)

    ax.set_title("Gradient Flow: Norm by Layer Depth",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("Layer (1=input side)", fontsize=9)
    ax.set_ylabel("Gradient L2-Norm (log)", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")


def _plot_dead_neurons(ax: plt.Axes, dead_results: dict) -> None:
    """Bar chart of dead neuron percentage for each configuration."""
    labels = list(dead_results.keys())
    pcts   = [dead_results[l]["dead_pct"] for l in labels]
    bars   = ax.bar(range(len(labels)), pcts,
                    color=["#e74c3c", "#27ae60", "#3498db"],
                    edgecolor="white", linewidth=1.2)

    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5, f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(
        [l.split("(")[0].strip() + "\n" + l.split("(")[1].rstrip(")")
         for l in labels],
        fontsize=7
    )
    ax.set_ylabel("Dead Neurons (%)", fontsize=9)
    ax.set_title("Dead Neuron Analysis", fontweight="bold", fontsize=10)
    ax.set_ylim(0, max(pcts) * 1.25 + 5)
    ax.grid(True, axis="y", alpha=0.3)


def figure1_overview(grad_results: dict, dead_results: dict) -> None:
    """Figure 1: 2×2 dashboard — functions, derivatives, gradient flow, dead neurons."""
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle("Phase 1 — Topic 2: Activation Functions  │  Overview Dashboard",
                 fontsize=13, fontweight="bold")

    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    _plot_fn_and_derivatives(ax1, ax2)
    _plot_gradient_flow(ax3, grad_results)
    _plot_dead_neurons(ax4, dead_results)

    path = os.path.join(RESULTS, "02_activation_overview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  [VIZ] Overview saved → {path}")
    plt.close(fig)


def figure2_training(histories: dict) -> None:
    """Figure 2: val-loss and val-accuracy learning curves for all activations."""
    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Comparison: make_circles  |  All Activations",
                 fontsize=12, fontweight="bold")

    epochs = None
    for name, h in histories.items():
        color  = PLOT_COLORS.get(name, "#000000")
        if epochs is None:
            epochs = range(1, len(h["val_loss"]) + 1)
        ax_l.plot(epochs, h["val_loss"],
                  color=color, lw=1.8, label=name, alpha=0.85)
        ax_a.plot(epochs, [a * 100 for a in h["val_acc"]],
                  color=color, lw=1.8, label=name, alpha=0.85)

    ax_l.set_title("Validation Loss", fontweight="bold")
    ax_l.set_xlabel("Epoch"); ax_l.set_ylabel("BCE Loss")
    ax_l.legend(fontsize=8, loc="upper right"); ax_l.grid(True, alpha=0.3)

    ax_a.set_title("Validation Accuracy", fontweight="bold")
    ax_a.set_xlabel("Epoch"); ax_a.set_ylabel("Accuracy (%)")
    ax_a.legend(fontsize=8, loc="lower right"); ax_a.grid(True, alpha=0.3)

    path = os.path.join(RESULTS, "02_training_comparison.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [VIZ] Training curves saved → {path}")
    plt.close(fig)


def figure3_boundaries(models: dict,
                        X_va: torch.Tensor, y_va: torch.Tensor,
                        histories: dict) -> None:
    """Figure 3: 3×3 grid of decision boundaries for each activation."""
    names  = list(models.keys())
    n_cols = 3
    n_rows = (len(names) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5 * n_cols, 4.5 * n_rows))
    axes = axes.flatten()

    # Build mesh grid once
    margin = 0.6
    x1_min = X_va[:, 0].min().item() - margin
    x1_max = X_va[:, 0].max().item() + margin
    x2_min = X_va[:, 1].min().item() - margin
    x2_max = X_va[:, 1].max().item() + margin
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 250),
                            np.linspace(x2_min, x2_max, 250))
    grid    = torch.tensor(np.c_[xx1.ravel(), xx2.ravel()],
                           dtype=torch.float32).to(DEVICE)
    y_flat  = y_va.squeeze().cpu().numpy().astype(int)

    for idx, (name, model) in enumerate(models.items()):
        ax = axes[idx]
        model.eval()
        with torch.no_grad():
            Z = model(grid).cpu().numpy().reshape(xx1.shape)

        ax.contourf(xx1, xx2, Z, alpha=0.3, cmap="RdBu", levels=50)
        ax.contour(xx1, xx2, Z, levels=[0.5], colors="black", linewidths=1.8)

        for cls, col in enumerate(["#e74c3c", "#2980b9"]):
            mask = y_flat == cls
            ax.scatter(X_va[mask, 0].cpu(), X_va[mask, 1].cpu(),
                       c=col, s=12, alpha=0.65,
                       edgecolors="white", linewidths=0.3)

        final_vacc = histories[name]["val_acc"][-1] * 100
        ax.set_title(f"{name}  ({final_vacc:.1f}%)",
                     fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    for idx in range(len(names), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Decision Boundaries by Activation Function  (make_circles Val Set)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS, "02_decision_boundaries.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [VIZ] Decision boundaries saved → {path}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "▓"*65)
    print("  DEEP LEARNING MASTERY REPOSITORY")
    print("  Phase 1 — Topic 2: Activation Functions")
    print("▓"*65)

    # ── A: NumPy activations (used in visualization) ─────────────────────────
    print("\n[SECTION A] NumPy Activation Functions — ready.")
    print(f"  Functions defined: {[f.name for f in ALL_NUMPY_ACTS]}")

    # ── B: PyTorch factories (used in all torch experiments) ──────────────────
    print(f"\n[SECTION B] PyTorch Factories — ready.")
    print(f"  Activations: {list(TORCH_ACT_FACTORY.keys())}")

    # ── C: Gradient flow experiment ───────────────────────────────────────────
    grad_results = gradient_flow_experiment(n_layers=15, n_hidden=64)

    # ── D: Data pipeline + training comparison ────────────────────────────────
    print("\n" + "="*65)
    print("SECTION D — Building Data Pipeline")
    print("="*65)
    (train_loader, val_loader,
     X_tr, X_va, y_tr, y_va) = build_data_pipeline(
        n_samples=1000, noise=0.1, batch_size=32
    )

    all_histories, all_models = run_training_comparison(
        train_loader, val_loader, n_epochs=100
    )

    # ── E: Dead neuron analysis ───────────────────────────────────────────────
    dead_results = dead_neuron_analysis(train_loader)

    # ── F: Softmax stability demo ─────────────────────────────────────────────
    softmax_stability_demo()

    # ── G: Visualizations ─────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("SECTION G — Generating Visualizations")
    print("="*65)

    figure1_overview(grad_results, dead_results)
    figure2_training(all_histories)
    figure3_boundaries(all_models, X_va, y_va, all_histories)

    print("\n" + "▓"*65)
    print("  ✓  All sections complete.")
    print(f"  ✓  Three figures saved to ./{RESULTS}/")
    print("▓"*65 + "\n")


if __name__ == "__main__":
    main()
