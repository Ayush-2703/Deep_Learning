"""
Phase 1 — Topic 3: Gradient Descent & Backpropagation
======================================================
Repository : deep-learning-mastery/phase-1-foundations/03-gradient-descent-backprop/
File       : implementation.py
Framework  : NumPy (manual backprop) + PyTorch (autograd demo) + matplotlib
Python     : 3.10+

Run:
    pip install torch numpy scikit-learn matplotlib
    python implementation.py

What this file demonstrates
────────────────────────────
  SECTION A │ Manual backpropagation (NumPy) — full derivation, no autograd
  SECTION B │ Numerical gradient check — verify analytical gradients
  SECTION C │ GD variants — Full-Batch vs Mini-Batch vs Stochastic comparison
  SECTION D │ Learning rate effects — convergence, oscillation, divergence
  SECTION E │ Loss surface + GD trajectory visualization (2D quadratic)
  SECTION F │ PyTorch autograd demo — computation graphs, accumulation bug
  SECTION G │ Visualization dashboard (3 figures)
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
from matplotlib.patches import FancyArrowPatch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SEED    = 42
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"[CONFIG] Device  : {DEVICE}")
print(f"[CONFIG] PyTorch : {torch.__version__}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION A — MANUAL BACKPROPAGATION (NumPy)
# 2-layer MLP: Input → Hidden(ReLU) → Output(Sigmoid) + BCE Loss
# Every gradient computed by hand — no autograd
# ═════════════════════════════════════════════════════════════════════════════

class ManualMLP:
    """
    2-layer MLP with hand-coded backpropagation.

    Column convention: X ∈ ℝ^(n⁰ × N) where N = batch size.
    This mirrors the mathematical notation in theory.md exactly.

    Gradient derivation (BCE loss + Sigmoid output):
      δ²      = (1/N)(Â − Y)               ← "output error" (prediction residual)
      ∂L/∂W²  = δ²(A¹)ᵀ
      ∂L/∂b²  = Σᵢ δ²ᵢ
      ∂L/∂A¹  = (W²)ᵀ δ²                  ← propagate error through W²
      δ¹      = ∂L/∂A¹ ⊙ ReLU'(Z¹)        ← gate by ReLU derivative
      ∂L/∂W¹  = δ¹ Xᵀ
      ∂L/∂b¹  = Σᵢ δ¹ᵢ

    Parameters
    ----------
    n_input  : int   — input feature dimension
    n_hidden : int   — hidden layer width
    lr       : float — learning rate for vanilla GD update
    """

    def __init__(self, n_input: int, n_hidden: int, lr: float = 0.1):
        rng = np.random.default_rng(SEED)

        # Xavier initialisation: std = √(2 / fan_in)
        self.W1 = rng.standard_normal((n_hidden, n_input)) * np.sqrt(2.0 / n_input)
        self.b1 = np.zeros((n_hidden, 1))                  # (n¹, 1)
        self.W2 = rng.standard_normal((1, n_hidden))        * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros((1, 1))                          # (1,  1)

        self.lr           = lr
        self.cache: dict  = {}          # stores forward-pass tensors for backprop
        self.loss_history: list = []
        self.grad_history: list = []    # L2-norm of all gradients per step

    # ── Activation helpers ────────────────────────────────────────────────────

    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def _relu_grad(z: np.ndarray) -> np.ndarray:
        """ReLU derivative: 1 where z > 0, else 0."""
        return (z > 0.0).astype(np.float64)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    # ── Forward Pass ─────────────────────────────────────────────────────────

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass. Caches every intermediate tensor needed by backward().

        X  : (n⁰, N)   — input batch, features as columns
        returns Â : (1, N)  — predicted probabilities
        """
        Z1 = self.W1 @ X   + self.b1      # (n¹, N) — hidden pre-activation
        A1 = self._relu(Z1)               # (n¹, N) — hidden activation
        Z2 = self.W2 @ A1  + self.b2      # (1,  N) — output pre-activation
        A2 = self._sigmoid(Z2)            # (1,  N) — predictions Â ∈ (0,1)

        # Cache ALL intermediate values: required for backward()
        self.cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2

    # ── Loss ─────────────────────────────────────────────────────────────────

    @staticmethod
    def bce_loss(A2: np.ndarray, Y: np.ndarray) -> float:
        """
        Binary Cross-Entropy:
          L = -(1/N) Σᵢ [yᵢ log(âᵢ) + (1-yᵢ) log(1-âᵢ)]
        """
        N   = Y.shape[1]
        eps = 1e-9          # numerical guard: avoid log(0)
        return float(
            -(1.0 / N) * np.sum(
                Y * np.log(A2 + eps) + (1 - Y) * np.log(1 - A2 + eps)
            )
        )

    # ── Backward Pass ─────────────────────────────────────────────────────────

    def backward(self, Y: np.ndarray) -> dict:
        """
        Full analytical backpropagation via chain rule.

        Key derivation — BCE loss + Sigmoid output (see theory.md §6):
          ∂L/∂Z² = (∂L/∂Â)(∂Â/∂Z²)
                 = [-(Y/Â) + (1-Y)/(1-Â)] · Â(1-Â)
                 = Â - Y      ← elegant cancellation
          δ² = (1/N)(Â - Y)

        Y : (1, N) — true binary labels
        Returns dict of gradients for all four parameters.
        """
        X, Z1, A1, Z2, A2 = (
            self.cache["X"],  self.cache["Z1"],
            self.cache["A1"], self.cache["Z2"], self.cache["A2"],
        )
        N = Y.shape[1]

        # ── Output layer ─────────────────────────────────────────────────────
        # δ² = (1/N)(Â - Y): prediction residual, averaged over batch
        dZ2 = (1.0 / N) * (A2 - Y)                        # (1,  N)
        dW2 = dZ2 @ A1.T                                   # (1,  n¹)
        db2 = np.sum(dZ2, axis=1, keepdims=True)           # (1,  1)

        # ── Hidden layer ─────────────────────────────────────────────────────
        # Propagate error backward through W² (use transpose)
        dA1 = self.W2.T @ dZ2                              # (n¹, N)
        # Gate by ReLU gradient (Hadamard ⊙ element-wise multiplication)
        dZ1 = dA1 * self._relu_grad(Z1)                   # (n¹, N)
        dW1 = dZ1 @ X.T                                    # (n¹, n⁰)
        db1 = np.sum(dZ1, axis=1, keepdims=True)           # (n¹, 1)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    # ── GD Update ────────────────────────────────────────────────────────────

    def step(self, grads: dict) -> None:
        """Vanilla gradient descent: θ ← θ − η · ∂L/∂θ"""
        self.W1 -= self.lr * grads["dW1"]
        self.b1 -= self.lr * grads["db1"]
        self.W2 -= self.lr * grads["dW2"]
        self.b2 -= self.lr * grads["db2"]

    # ── Training Loop ─────────────────────────────────────────────────────────

    def train(self, X: np.ndarray, Y: np.ndarray,
              n_epochs: int = 3000, print_every: int = 600) -> None:
        """
        Full training: forward → loss → backward → GD step.
        X : (n⁰, N),  Y : (1, N)
        """
        print(f"\n  Training ManualMLP ({n_epochs} epochs, lr={self.lr})...")
        for epoch in range(1, n_epochs + 1):
            A2    = self.forward(X)
            loss  = self.bce_loss(A2, Y)
            grads = self.backward(Y)
            self.step(grads)

            # Track total gradient L2-norm for convergence analysis
            gnorm = float(np.sqrt(
                sum(np.sum(g ** 2) for g in grads.values())
            ))
            self.loss_history.append(loss)
            self.grad_history.append(gnorm)

            if epoch % print_every == 0:
                acc = float(((A2 >= 0.5).astype(int) == Y).mean())
                print(f"    Epoch {epoch:5d} | Loss: {loss:.6f} | "
                      f"Acc: {acc * 100:.1f}% | ‖∇‖: {gnorm:.5f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.forward(X) >= 0.5).astype(int)     # (1, N) binary


def run_manual_backprop(X_col: np.ndarray, Y_col: np.ndarray) -> ManualMLP:
    """Train ManualMLP and report final performance."""
    print("\n" + "=" * 65)
    print("SECTION A — Manual Backpropagation (NumPy)")
    print("=" * 65)

    mlp = ManualMLP(n_input=2, n_hidden=8, lr=0.5)
    mlp.train(X_col, Y_col, n_epochs=3000, print_every=600)

    acc = float(((mlp.predict(X_col)) == Y_col).mean())
    print(f"\n  Final train accuracy: {acc * 100:.1f}%")
    return mlp


# ═════════════════════════════════════════════════════════════════════════════
# SECTION B — NUMERICAL GRADIENT CHECK
# Verify ManualMLP.backward() against central finite differences
# ═════════════════════════════════════════════════════════════════════════════

def numerical_gradient_check(model: ManualMLP,
                              X: np.ndarray, Y: np.ndarray,
                              eps: float = 1e-5) -> dict:
    """
    Numerical gradient via central finite differences:
        ∂L/∂θᵢ ≈ [L(θ + εeᵢ) − L(θ − εeᵢ)] / (2ε)

    Relative error (Karpathy criterion):
        e_rel = ‖∇_anal − ∇_num‖₂ / (‖∇_anal‖₂ + ‖∇_num‖₂ + 1e-8)

    Thresholds:
        e_rel < 1e-5  → ✓ PASS
        e_rel > 1e-3  → ✗ FAIL (implementation bug likely)
    """
    print("\n" + "=" * 65)
    print("SECTION B — Numerical Gradient Check")
    print(f"  Perturbation ε = {eps:.0e}")
    print("=" * 65)

    # Compute analytical gradients
    model.forward(X)
    analytical = model.backward(Y)

    param_names = ["W1", "b1", "W2", "b2"]
    results = {}

    for name in param_names:
        param     = getattr(model, name)      # reference — modifying it changes model
        g_anal    = analytical[f"d{name}"]
        g_num     = np.zeros_like(param)

        # Iterate over every scalar element of param
        for idx in np.ndindex(param.shape):
            orig = float(param[idx])

            param[idx] = orig + eps
            lp = model.bce_loss(model.forward(X), Y)  # L(θ + εeᵢ)

            param[idx] = orig - eps
            lm = model.bce_loss(model.forward(X), Y)  # L(θ − εeᵢ)

            param[idx] = orig                           # restore original value

            g_num[idx] = (lp - lm) / (2.0 * eps)

        num   = np.linalg.norm(g_anal - g_num)
        denom = np.linalg.norm(g_anal) + np.linalg.norm(g_num) + 1e-8
        e_rel = num / denom

        passed = e_rel < 1e-4
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:3s} | rel_error = {e_rel:.3e}  {status}")

        results[name] = {
            "analytical": g_anal,
            "numerical":  g_num,
            "rel_error":  e_rel,
            "pass":       passed,
        }

    all_pass = all(r["pass"] for r in results.values())
    print(f"\n  Overall: {'✓ ALL PASSED' if all_pass else '✗ SOME FAILED'}")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# SECTION C — GRADIENT DESCENT VARIANTS COMPARISON
# Full-Batch GD vs Mini-Batch (32) vs Stochastic (1)
# ═════════════════════════════════════════════════════════════════════════════

def _make_torch_mlp(seed: int = SEED) -> nn.Module:
    """Build a small reproducible 2-layer PyTorch MLP."""
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(2, 16), nn.ReLU(),
        nn.Linear(16, 1), nn.Sigmoid(),
    )
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return model


def compare_gd_variants(X_np: np.ndarray, Y_np: np.ndarray,
                         max_steps: int = 800) -> dict:
    """
    Train the same architecture with three batch sizes.
    Track loss vs number of gradient steps (not epochs) for a fair comparison
    — each gradient step represents the same unit of computational work.

    Variants:
      Full-Batch  : one GD step per epoch using ALL data
      Mini-Batch  : one GD step per 32-sample batch
      Stochastic  : one GD step per single sample
    """
    print("\n" + "=" * 65)
    print("SECTION C — GD Variants Comparison")
    print(f"  Max gradient steps tracked: {max_steps}")
    print("=" * 65)

    variants = [
        ("Full-Batch GD",   len(X_np), "#e74c3c"),
        ("Mini-Batch (32)", 32,         "#27ae60"),
        ("Stochastic (1)",  1,           "#3498db"),
    ]

    X_t = torch.tensor(X_np, dtype=torch.float32)
    Y_t = torch.tensor(Y_np, dtype=torch.float32).unsqueeze(1)
    criterion = nn.BCELoss()
    results = {}

    for name, batch_size, color in variants:
        torch.manual_seed(SEED)
        model     = _make_torch_mlp()
        optimizer = optim.SGD(model.parameters(), lr=0.05)
        loader    = DataLoader(TensorDataset(X_t, Y_t),
                               batch_size=batch_size, shuffle=True,
                               generator=torch.Generator().manual_seed(SEED))

        loss_per_step = []
        step = 0

        while step < max_steps:
            model.train()
            for Xb, Yb in loader:
                optimizer.zero_grad()
                pred = model(Xb)
                loss = criterion(pred, Yb)
                loss.backward()
                optimizer.step()
                loss_per_step.append(loss.item())
                step += 1
                if step >= max_steps:
                    break

        # Final accuracy on full training set
        model.eval()
        with torch.no_grad():
            preds    = (model(X_t) >= 0.5).float()
            final_acc = (preds == Y_t).float().mean().item()

        results[name] = {
            "loss_per_step": loss_per_step,
            "final_acc":     final_acc,
            "color":         color,
            "batch_size":    batch_size,
        }
        print(f"  {name:20s} | steps: {step:4d} | "
              f"final_acc: {final_acc * 100:.1f}%")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# SECTION D — LEARNING RATE EFFECTS
# Same architecture, same data, five learning rates
# ═════════════════════════════════════════════════════════════════════════════

def compare_learning_rates(X_np: np.ndarray, Y_np: np.ndarray,
                            n_epochs: int = 120) -> dict:
    """
    Train with five learning rates demonstrating:
      lr=0.001  → too small: converges slowly
      lr=0.01   → conservative but stable
      lr=0.1    → near-optimal for this problem
      lr=0.5    → oscillating: overshoots loss bowl
      lr=2.0    → diverges: loss grows to NaN/inf
    """
    print("\n" + "=" * 65)
    print("SECTION D — Learning Rate Effects")
    print("=" * 65)

    lrs    = [0.001, 0.01,     0.1,     0.5,     2.0]
    colors = ["#9b59b6", "#27ae60", "#2980b9", "#e67e22", "#e74c3c"]
    labels = [f"lr={lr}" for lr in lrs]

    X_t  = torch.tensor(X_np, dtype=torch.float32)
    Y_t  = torch.tensor(Y_np, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(X_t, Y_t), batch_size=32,
                        shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    criterion = nn.BCELoss()
    results = {}

    for lr, color, label in zip(lrs, colors, labels):
        torch.manual_seed(SEED)
        model     = _make_torch_mlp()
        optimizer = optim.SGD(model.parameters(), lr=lr)

        epoch_losses = []

        for _ in range(n_epochs):
            model.train()
            tl, tn = 0.0, 0
            for Xb, Yb in loader:
                optimizer.zero_grad()
                pred = model(Xb)
                loss = criterion(pred, Yb)
                if torch.isfinite(loss):
                    loss.backward()
                    optimizer.step()
                    tl += loss.item() * len(Xb)
                else:
                    tl += 10.0 * len(Xb)   # cap at 10 for plotting
                tn += len(Xb)
            epoch_losses.append(tl / tn if tn > 0 else 10.0)

        final = epoch_losses[-1]
        results[label] = {"losses": epoch_losses, "color": color, "lr": lr}
        print(f"  {label:8s} | final_loss: {final:.4f} "
              f"{'← diverged' if final > 5 else ''}")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# SECTION E — LOSS SURFACE + GD TRAJECTORY VISUALIZATION
# 2D quadratic loss: f(w₁,w₂) = (w₁-1)² + 4(w₂-1)²
# Minimum at (1,1). Condition number κ = 4. Convergence limit η < 0.25.
# ═════════════════════════════════════════════════════════════════════════════

def loss_surface_and_trajectories() -> dict:
    """
    Compute the 2D loss surface and GD trajectories for three learning rates.

    Loss function:   f(w₁, w₂) = (w₁ − 1)² + 4(w₂ − 1)²
    Gradient:        ∇f = [2(w₁−1),  8(w₂−1)]
    Minimum:         (w₁*, w₂*) = (1, 1)
    Curvatures:      ∂²f/∂w₁² = 2,  ∂²f/∂w₂² = 8
    Condition number: κ = 8/2 = 4
    Stability limit:  η < 2/λ_max = 2/8 = 0.25
    """
    print("\n" + "=" * 65)
    print("SECTION E — Loss Surface + GD Trajectories")
    print("  f(w₁,w₂) = (w₁-1)² + 4(w₂-1)²  |  minimum at (1,1)")
    print("=" * 65)

    def f(w1, w2):
        return (w1 - 1.0) ** 2 + 4.0 * (w2 - 1.0) ** 2

    def grad_f(w1, w2):
        return np.array([2.0 * (w1 - 1.0), 8.0 * (w2 - 1.0)])

    def run_gd(lr: float, n_steps: int = 60,
               start: tuple = (-2.0, 3.0)) -> np.ndarray:
        w    = np.array(list(start), dtype=np.float64)
        traj = [w.copy()]
        for _ in range(n_steps):
            g = grad_f(w[0], w[1])
            w = w - lr * g
            traj.append(w.copy())
            # Stop if diverged
            if np.any(np.abs(w) > 20):
                break
        return np.array(traj)

    # Mesh grid for contour plot
    w1g = np.linspace(-3.5, 4.5, 400)
    w2g = np.linspace(-0.5, 4.5, 400)
    W1, W2 = np.meshgrid(w1g, w2g)
    Z       = f(W1, W2)

    # Three learning rates showing distinct behaviours
    lr_configs = [
        (0.05,  "#9b59b6", "η=0.05  (slow)"),
        (0.12,  "#27ae60", "η=0.12  (optimal)"),
        (0.24,  "#e74c3c", "η=0.24  (oscillating)"),
    ]

    trajectories = []
    for lr, color, label in lr_configs:
        traj = run_gd(lr)
        loss_traj = [f(w[0], w[1]) for w in traj]
        trajectories.append({"traj": traj, "loss": loss_traj,
                              "color": color, "label": label, "lr": lr})
        n = len(traj)
        print(f"  {label:30s} | steps={n-1:3d} | "
              f"final_loss={loss_traj[-1]:.5f}")

    return {"W1": W1, "W2": W2, "Z": Z, "trajectories": trajectories}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION F — PYTORCH AUTOGRAD DEMO
# ═════════════════════════════════════════════════════════════════════════════

def autograd_demo() -> None:
    """
    Show PyTorch's dynamic computation graph and automatic differentiation.

    Demonstrates:
      1. Scalar gradient: f(x) = 3x² + 2x + 1
      2. Chain rule on multi-input function
      3. Gradient accumulation bug (and fix)
      4. no_grad() context for inference
    """
    print("\n" + "=" * 65)
    print("SECTION F — PyTorch Autograd Demo")
    print("=" * 65)

    # ── 1. Scalar derivative ─────────────────────────────────────────────────
    print("\n  [1] Scalar: f(x) = 3x² + 2x + 1  at x = 2")
    x = torch.tensor(2.0, requires_grad=True)
    f = 3 * x ** 2 + 2 * x + 1              # builds computation graph
    f.backward()                              # df/dx = 6x + 2 = 14
    print(f"      f(2)  = {f.item():.1f}   (expected 17)")
    print(f"      f'(2) = {x.grad.item():.1f}   (expected 14 = 6·2+2)")

    # ── 2. Chain rule: z = x² + y², g = sin(z) ───────────────────────────────
    print("\n  [2] Chain rule: z = x²+y², g = sin(z)  at x=1, y=1")
    x = torch.tensor(1.0, requires_grad=True)
    y = torch.tensor(1.0, requires_grad=True)
    z = x ** 2 + y ** 2                      # z = 2
    g = torch.sin(z)                          # g = sin(2)
    g.backward()
    # Analytically: ∂g/∂x = cos(z)·2x = cos(2)·2 ≈ -0.8323
    expected = np.cos(2.0) * 2.0
    print(f"      ∂g/∂x = {x.grad.item():.6f}   (expected {expected:.6f})")
    print(f"      ∂g/∂y = {y.grad.item():.6f}   (expected {expected:.6f})")

    # ── 3. Gradient accumulation bug ─────────────────────────────────────────
    print("\n  [3] Gradient accumulation — common bug:")
    x = torch.tensor(3.0, requires_grad=True)
    for i in range(3):
        loss = x ** 2             # f(3) = 9,  f'(3) = 6
        loss.backward()
        note = "← correct: f'(3)=6" if i == 0 else "← ACCUMULATES! BUG"
        print(f"      After backward() #{i + 1}: x.grad = {x.grad.item():.1f}  {note}")

    print("\n      Fix: call x.grad.zero_() or optimizer.zero_grad() each step")
    x.grad.zero_()
    loss = x ** 2
    loss.backward()
    print(f"      After zero_() + backward(): x.grad = {x.grad.item():.1f}  ← correct")

    # ── 4. no_grad inference ─────────────────────────────────────────────────
    print("\n  [4] torch.no_grad() for inference:")
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    with torch.no_grad():
        y = x * 2 + 1
    print(f"      y.requires_grad = {y.requires_grad}   (False = no graph built)")
    print(f"      y.grad_fn       = {y.grad_fn}   (None = no gradient function)")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION G — VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def figure1_backprop_and_checks(mlp: ManualMLP,
                                  grad_check: dict) -> None:
    """
    Figure 1 (2×2): Manual backprop curves + gradient check error bars.
    """
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Phase 1 — Topic 3: Gradient Descent & Backpropagation  │  Fig 1",
                 fontsize=13, fontweight="bold")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])   # loss curve
    ax2 = fig.add_subplot(gs[0, 1])   # gradient norm curve
    ax3 = fig.add_subplot(gs[1, 0])   # gradient check relative errors
    ax4 = fig.add_subplot(gs[1, 1])   # analytical vs numerical comparison

    # ── Panel 1: Training loss ────────────────────────────────────────────────
    epochs = range(1, len(mlp.loss_history) + 1)
    ax1.plot(epochs, mlp.loss_history, color="#e74c3c", lw=2)
    ax1.set_title("Manual Backprop — Training Loss", fontweight="bold", fontsize=10)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE Loss")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Gradient L2-norm ─────────────────────────────────────────────
    ax2.semilogy(epochs, mlp.grad_history, color="#2980b9", lw=2)
    ax2.set_title("Gradient L2-Norm Over Training", fontweight="bold", fontsize=10)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("‖∇θ L‖₂  (log scale)")
    ax2.grid(True, alpha=0.3, which="both")

    # ── Panel 3: Gradient check relative errors (bar) ─────────────────────────
    names  = list(grad_check.keys())
    errors = [grad_check[n]["rel_error"] for n in names]
    colors = ["#27ae60" if grad_check[n]["pass"] else "#e74c3c" for n in names]
    bars   = ax3.bar(names, errors, color=colors, edgecolor="white", lw=1.2)
    ax3.axhline(y=1e-4, color="black", ls="--", lw=1.2, alpha=0.6,
                label="Pass threshold (1e-4)")
    ax3.set_yscale("log")
    ax3.set_title("Gradient Check — Relative Errors", fontweight="bold", fontsize=10)
    ax3.set_ylabel("Relative Error (log scale)")
    ax3.legend(fontsize=8); ax3.grid(True, axis="y", alpha=0.3, which="both")
    for bar, err in zip(bars, errors):
        ax3.text(bar.get_x() + bar.get_width() / 2, err * 2,
                 f"{err:.1e}", ha="center", fontsize=8)

    # ── Panel 4: W2 analytical vs numerical element comparison ────────────────
    w2_data  = grad_check["W2"]
    g_anal   = w2_data["analytical"].flatten()
    g_num    = w2_data["numerical"].flatten()
    idx      = range(len(g_anal))
    ax4.bar([i - 0.2 for i in idx], g_anal, width=0.38,
            color="#e74c3c", label="Analytical", alpha=0.8)
    ax4.bar([i + 0.2 for i in idx], g_num,  width=0.38,
            color="#3498db", label="Numerical",  alpha=0.8)
    ax4.set_title("W2 Gradient: Analytical vs Numerical", fontweight="bold", fontsize=10)
    ax4.set_xlabel("Parameter index"); ax4.set_ylabel("Gradient value")
    ax4.legend(fontsize=9); ax4.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS, "03_backprop_gradcheck.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  [VIZ] Figure 1 saved → {path}")
    plt.close(fig)


def figure2_gd_variants_and_lr(gd_results: dict, lr_results: dict) -> None:
    """
    Figure 2 (1×2): GD variant convergence + LR effect comparison.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 1 — Topic 3: GD Variants & Learning Rate Effects  │  Fig 2",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: GD Variants (loss vs gradient steps) ────────────────────────
    for name, data in gd_results.items():
        steps = range(1, len(data["loss_per_step"]) + 1)
        # Smooth noisy curves with a running window for visibility
        raw   = np.array(data["loss_per_step"])
        if name == "Stochastic (1)":
            # Apply moving average to smooth SGD noise
            k = 20
            smooth = np.convolve(raw, np.ones(k) / k, mode="valid")
            ax1.plot(range(k, len(raw) + 1), smooth,
                     color=data["color"], lw=2.0, label=f"{name} (smoothed)")
            ax1.plot(steps, raw, color=data["color"], lw=0.4, alpha=0.25)
        else:
            ax1.plot(steps, raw, color=data["color"], lw=2.2,
                     label=f"{name} ({data['final_acc']*100:.0f}%)")

    ax1.set_title("GD Variants: Loss vs Gradient Steps", fontweight="bold", fontsize=11)
    ax1.set_xlabel("Gradient Steps"); ax1.set_ylabel("BCE Loss")
    ax1.legend(fontsize=9, loc="upper right"); ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # ── Panel 2: Learning rate effects (loss vs epoch) ───────────────────────
    for label, data in lr_results.items():
        losses  = np.clip(data["losses"], 0, 5)     # cap at 5 for diverged runs
        epochs  = range(1, len(losses) + 1)
        ax2.plot(epochs, losses, color=data["color"], lw=2.2,
                 label=label, alpha=0.9)

    ax2.set_title("Learning Rate Effects on Convergence", fontweight="bold", fontsize=11)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("BCE Loss (capped at 5)")
    ax2.legend(fontsize=9, loc="upper right"); ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 5.2)

    plt.tight_layout()
    path = os.path.join(RESULTS, "03_gd_variants_lr.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [VIZ] Figure 2 saved → {path}")
    plt.close(fig)


def figure3_loss_surface(surface_data: dict) -> None:
    """
    Figure 3 (1×2): 2D loss surface contour + loss-vs-step for trajectories.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Phase 1 — Topic 3: Loss Surface  f(w₁,w₂) = (w₁-1)² + 4(w₂-1)²  │  Fig 3",
        fontsize=12, fontweight="bold")

    W1 = surface_data["W1"];  W2 = surface_data["W2"];  Z = surface_data["Z"]

    # ── Panel 1: Contour + trajectories ──────────────────────────────────────
    levels = np.logspace(-2, 2, 30)
    ax1.contourf(W1, W2, Z, levels=levels, cmap="Blues", alpha=0.65)
    ax1.contour( W1, W2, Z, levels=levels, colors="white", linewidths=0.4, alpha=0.5)

    for td in surface_data["trajectories"]:
        traj = td["traj"]
        ax1.plot(traj[:, 0], traj[:, 1], "-o", color=td["color"],
                 lw=2.2, ms=4, label=td["label"], alpha=0.9)
        # Mark start and end
        ax1.plot(traj[0, 0],  traj[0, 1],  "s", color=td["color"], ms=9, zorder=5)
        ax1.plot(traj[-1, 0], traj[-1, 1], "*", color=td["color"], ms=12, zorder=5)

    # Mark the global minimum
    ax1.plot(1, 1, "k*", ms=16, zorder=10, label="Minimum (1,1)")
    ax1.set_title("GD Trajectories on Loss Surface", fontweight="bold", fontsize=11)
    ax1.set_xlabel("w₁"); ax1.set_ylabel("w₂")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.set_xlim(-3.5, 4.5); ax1.set_ylim(-0.5, 4.5)

    # ── Panel 2: Loss-vs-step for each trajectory ─────────────────────────────
    for td in surface_data["trajectories"]:
        steps = range(len(td["loss"]))
        ax2.semilogy(steps, td["loss"], "-o", color=td["color"],
                     lw=2.2, ms=4, label=td["label"], alpha=0.9)

    ax2.axhline(y=0, color="black", ls="--", lw=1, alpha=0.4,
                label="Optimal loss = 0")
    ax2.set_title("Convergence: Loss vs GD Step", fontweight="bold", fontsize=11)
    ax2.set_xlabel("GD Step"); ax2.set_ylabel("f(w₁,w₂)  (log scale)")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    path = os.path.join(RESULTS, "03_loss_surface.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [VIZ] Figure 3 saved → {path}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\n" + "▓" * 65)
    print("  DEEP LEARNING MASTERY REPOSITORY")
    print("  Phase 1 — Topic 3: Gradient Descent & Backpropagation")
    print("▓" * 65)

    # ── Shared dataset: make_moons ─────────────────────────────────────────────
    X, y = make_moons(n_samples=500, noise=0.2, random_state=SEED)
    X_tr, _, y_tr, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)

    # Column convention for ManualMLP:  (n_features, N)
    X_col = X_tr.T.astype(np.float64)              # (2, 400)
    Y_col = y_tr.reshape(1, -1).astype(np.float64) # (1, 400)

    # ── A: Manual backprop ────────────────────────────────────────────────────
    mlp = run_manual_backprop(X_col, Y_col)

    # ── B: Gradient check — use a tiny model for speed ────────────────────────
    print("\n" + "=" * 65)
    print("SECTION B — Numerical Gradient Check")
    print("=" * 65)
    small = ManualMLP(n_input=2, n_hidden=4, lr=0.1)
    # Use only 8 samples: gradient check is O(n_params × N) forward passes
    Xs, Ys = X_col[:, :8], Y_col[:, :8]
    grad_check = numerical_gradient_check(small, Xs, Ys, eps=1e-5)

    # ── C: GD variants ────────────────────────────────────────────────────────
    gd_results = compare_gd_variants(X_tr, y_tr.astype(np.float32))

    # ── D: Learning rate effects ──────────────────────────────────────────────
    lr_results = compare_learning_rates(X_tr, y_tr.astype(np.float32))

    # ── E: Loss surface ────────────────────────────────────────────────────────
    surface_data = loss_surface_and_trajectories()

    # ── F: Autograd demo ─────────────────────────────────────────────────────
    autograd_demo()

    # ── G: Visualizations ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SECTION G — Generating Visualizations")
    print("=" * 65)
    figure1_backprop_and_checks(mlp, grad_check)
    figure2_gd_variants_and_lr(gd_results, lr_results)
    figure3_loss_surface(surface_data)

    print("\n" + "▓" * 65)
    print("  ✓  All 7 sections complete.")
    print(f"  ✓  3 figures saved to ./{RESULTS}/")
    print("▓" * 65 + "\n")


if __name__ == "__main__":
    main()
