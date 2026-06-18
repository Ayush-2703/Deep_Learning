"""
Phase 1 — Topic 1: Perceptron & Multilayer Perceptron (MLP)
============================================================
Repository : deep-learning-mastery/phase-1-foundations/01-perceptron-and-mlp/
File       : implementation.py
Framework  : PyTorch 2.x | NumPy | scikit-learn | matplotlib
Python     : 3.10+

Run:
    pip install torch numpy scikit-learn matplotlib seaborn
    python implementation.py

What this file demonstrates
────────────────────────────
  SECTION A │ NumPy Perceptron from scratch — AND gate (linearly separable)
  SECTION B │ NumPy Perceptron failure proof — XOR gate (non-separable)
  SECTION C │ MLP from scratch (raw PyTorch tensors, no nn.Module) — XOR solver
  SECTION D │ Production MLP (nn.Module + Adam) — make_moons binary classification
  SECTION E │ Full training pipeline: DataLoader, train loop, eval loop
  SECTION F │ Evaluation: accuracy, classification report, confusion matrix
  SECTION G │ Visualizations: decision boundary, loss/accuracy curves, convergence
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
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")
matplotlib.use("Agg")   # non-interactive backend (safe for servers/CI)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SEED       = 42
LR         = 1e-3        # Adam learning rate for production MLP
EPOCHS     = 150         # training epochs for production MLP
BATCH_SIZE = 32          # mini-batch size
N_SAMPLES  = 1000        # make_moons dataset size
NOISE      = 0.25        # noise level in make_moons

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS    = "results"
os.makedirs(RESULTS, exist_ok=True)

print(f"[CONFIG] Device     : {DEVICE}")
print(f"[CONFIG] PyTorch    : {torch.__version__}")
print(f"[CONFIG] Random seed: {SEED}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION A — PERCEPTRON FROM SCRATCH (NumPy)
# Implements the Rosenblatt (1958) online learning rule
# ═════════════════════════════════════════════════════════════════════════════

class Perceptron:
    """
    Single-layer Perceptron with the Rosenblatt learning rule.

    Decision function:
        z   = wᵀx + b
        ŷ  = H(z)  =  1 if z ≥ 0 else 0

    Weight update (only on misclassification):
        Δw = η · (y - ŷ) · x
        Δb = η · (y - ŷ)

    Parameters
    ----------
    learning_rate : float   — step size η (default 0.1)
    n_epochs      : int     — maximum training iterations (default 100)
    """

    def __init__(self, learning_rate: float = 0.1, n_epochs: int = 100):
        self.lr          = learning_rate
        self.n_epochs    = n_epochs
        self.weights_    = None      # shape: (n_features,) — set in fit()
        self.bias_       = 0.0
        self.errors_     = []        # misclassification count per epoch

    # ── Activation ───────────────────────────────────────────────────────────

    @staticmethod
    def _heaviside(z: np.ndarray) -> np.ndarray:
        """Heaviside step function: H(z) = 1 if z ≥ 0 else 0."""
        return np.where(z >= 0.0, 1, 0).astype(int)

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """
        Train the Perceptron on labeled data using online (sample-by-sample)
        weight updates.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,), values in {0, 1}

        Returns
        -------
        self  (for method chaining)
        """
        n_samples, n_features = X.shape

        # Initialize weights to zero — zero-init is standard for Perceptron
        self.weights_ = np.zeros(n_features, dtype=np.float64)
        self.bias_    = 0.0
        self.errors_  = []

        for epoch in range(self.n_epochs):
            n_errors = 0

            # Online learning: iterate one sample at a time
            for xi, yi in zip(X, y):
                # Forward: compute prediction
                z     = np.dot(xi, self.weights_) + self.bias_  # scalar
                y_hat = self._heaviside(np.array([z]))[0]        # 0 or 1

                # Compute signed error: +1, 0, or -1
                error = int(yi) - int(y_hat)

                # Update only if misclassified (error ≠ 0)
                if error != 0:
                    self.weights_ += self.lr * error * xi   # Δw = η·δ·x
                    self.bias_    += self.lr * error        # Δb = η·δ
                    n_errors      += 1

            self.errors_.append(n_errors)

            # Early stopping: perfect classification achieved
            if n_errors == 0:
                print(f"  [Perceptron] Converged at epoch {epoch + 1}/{self.n_epochs}")
                break
        else:
            print(f"  [Perceptron] Reached max epochs ({self.n_epochs}) without convergence")

        return self

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        z   = X @ w + b      ← vectorized over all samples
        ŷ  = H(z)
        """
        z = X @ self.weights_ + self.bias_   # shape: (n_samples,)
        return self._heaviside(z)             # shape: (n_samples,)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return fraction of correctly classified samples."""
        return float(np.mean(self.predict(X) == y))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION A: Demo — AND Gate (linearly separable → converges)
# ─────────────────────────────────────────────────────────────────────────────

def demo_perceptron_and() -> Perceptron:
    """Train a Perceptron on the AND logic gate and verify 100% accuracy."""
    print("\n" + "="*65)
    print("SECTION A — Perceptron on AND Gate (linearly separable)")
    print("="*65)

    # AND truth table
    X_and = np.array([[0., 0.],
                       [0., 1.],
                       [1., 0.],
                       [1., 1.]])
    y_and = np.array([0, 0, 0, 1])    # 1 only when BOTH inputs are 1

    p = Perceptron(learning_rate=0.1, n_epochs=100)
    p.fit(X_and, y_and)

    print(f"\n  Weights : {p.weights_}")
    print(f"  Bias    : {p.bias_:.4f}")
    print(f"  Accuracy: {p.accuracy(X_and, y_and)*100:.1f}%\n")

    print("  Input → Prediction | True Label | Status")
    print("  " + "-"*42)
    for xi, yi in zip(X_and, y_and):
        pred = p.predict(xi.reshape(1, -1))[0]
        mark = "✓" if pred == yi else "✗"
        print(f"  {xi.astype(int).tolist()} → pred={pred}  | true={yi}  | {mark}")

    return p


# ─────────────────────────────────────────────────────────────────────────────
# SECTION B: Demo — XOR Gate (NOT linearly separable → fails)
# ─────────────────────────────────────────────────────────────────────────────

def demo_perceptron_xor() -> Perceptron:
    """
    Attempt to train a Perceptron on XOR.
    Will NOT converge because XOR is not linearly separable.
    Motivates the need for an MLP.
    """
    print("\n" + "="*65)
    print("SECTION B — Perceptron on XOR Gate (NOT linearly separable)")
    print("="*65)

    X_xor = np.array([[0., 0.],
                       [0., 1.],
                       [1., 0.],
                       [1., 1.]])
    y_xor = np.array([0, 1, 1, 0])    # XOR: 1 when inputs differ

    p = Perceptron(learning_rate=0.1, n_epochs=1000)
    p.fit(X_xor, y_xor)

    acc = p.accuracy(X_xor, y_xor)
    print(f"\n  Best accuracy on XOR: {acc*100:.1f}%")
    print("  → Cannot achieve 100%. Single hyperplane cannot separate XOR classes.")
    print("  → Solution: Add hidden layers → MLP (Section C)\n")

    return p


# ═════════════════════════════════════════════════════════════════════════════
# SECTION C — MLP FROM SCRATCH (Raw PyTorch Tensors)
# Exposes the raw math: no nn.Module abstractions
# Architecture: Input(2) → Hidden(4, ReLU) → Output(1, Sigmoid)
# ═════════════════════════════════════════════════════════════════════════════

class MLPScratch:
    """
    2-layer MLP using raw PyTorch tensors — no nn.Module.
    Written to make the math of forward/backward passes explicit.

    Architecture:
        a⁰ = x                              (input, shape: N×2)
        z¹ = a⁰ W¹ + b¹                    (hidden pre-activation, N×n_hidden)
        a¹ = ReLU(z¹)                       (hidden activation, N×n_hidden)
        z² = a¹ W² + b²                     (output pre-activation, N×1)
        ŷ  = Sigmoid(z²)                    (predicted probability, N×1)

    Optimizer: vanilla SGD (manual gradient steps)
    Loss: Binary Cross-Entropy
    """

    def __init__(self, n_input: int, n_hidden: int, n_output: int,
                 lr: float = 0.5):
        self.lr = lr
        self.loss_history: list[float] = []

        # ── Xavier initialization ───────────────────────────────────────────
        # Scale = sqrt(2 / fan_in) — keeps activation variance stable at init
        std1 = np.sqrt(2.0 / n_input)
        std2 = np.sqrt(2.0 / n_hidden)

        # Layer 1 parameters — shape: (n_input × n_hidden) and (1 × n_hidden)
        self.W1 = torch.randn(n_input,  n_hidden, dtype=torch.float64) * std1
        self.b1 = torch.zeros(1,        n_hidden, dtype=torch.float64)

        # Layer 2 parameters — shape: (n_hidden × n_output) and (1 × n_output)
        self.W2 = torch.randn(n_hidden, n_output, dtype=torch.float64) * std2
        self.b2 = torch.zeros(1,        n_output, dtype=torch.float64)

        # Enable autograd tracking so PyTorch can compute ∂L/∂(W,b)
        for p in [self.W1, self.b1, self.W2, self.b2]:
            p.requires_grad_(True)

    # ── Activations ──────────────────────────────────────────────────────────

    @staticmethod
    def relu(z: torch.Tensor) -> torch.Tensor:
        """ReLU: max(0, z). Gradient = 1 if z>0 else 0."""
        return torch.clamp(z, min=0.0)

    @staticmethod
    def sigmoid(z: torch.Tensor) -> torch.Tensor:
        """Sigmoid: 1/(1+e^{-z}). Maps any real to (0,1)."""
        return 1.0 / (1.0 + torch.exp(-z))

    # ── Forward Pass ─────────────────────────────────────────────────────────

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
          X  : (N, n_input)
          z1 : (N, n_hidden)   — hidden pre-activations
          a1 : (N, n_hidden)   — hidden activations
          z2 : (N, n_output)   — output pre-activations
          a2 : (N, n_output)   — output probabilities ŷ ∈ (0,1)
        """
        self.z1 = X  @ self.W1 + self.b1    # (N,2)×(2,4) + (1,4) → (N,4)
        self.a1 = self.relu(self.z1)         # (N,4) element-wise
        self.z2 = self.a1 @ self.W2 + self.b2  # (N,4)×(4,1) + (1,1) → (N,1)
        self.a2 = self.sigmoid(self.z2)      # (N,1) probabilities
        return self.a2

    # ── Loss ─────────────────────────────────────────────────────────────────

    @staticmethod
    def bce(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Binary Cross-Entropy:
            L = -(1/N) Σ [y log(ŷ) + (1-y) log(1-ŷ)]
        eps prevents log(0) which would be -inf
        """
        eps  = 1e-9
        loss = -(y_true * torch.log(y_pred + eps) +
                 (1 - y_true) * torch.log(1 - y_pred + eps))
        return loss.mean()   # scalar

    # ── Parameter Update ─────────────────────────────────────────────────────

    def step(self):
        """
        Manual SGD step: θ ← θ - η · ∂L/∂θ
        Must be inside no_grad() to avoid autograd tracking the update itself.
        """
        with torch.no_grad():
            for param in [self.W1, self.b1, self.W2, self.b2]:
                param -= self.lr * param.grad     # gradient descent
                param.grad.zero_()               # reset gradient accumulator

    # ── Training Loop ─────────────────────────────────────────────────────────

    def train_model(self, X: torch.Tensor, y: torch.Tensor,
                    n_epochs: int = 10000, print_every: int = 2000) -> None:
        """
        Full training loop:
          1. Forward pass  → compute ŷ
          2. Loss          → scalar L
          3. backward()    → populate .grad for each param
          4. step()        → update params, zero grads
        """
        print(f"\n  Training MLPScratch ({n_epochs} epochs)...")

        for epoch in range(1, n_epochs + 1):
            y_pred = self.forward(X)             # (4,1) — forward pass
            loss   = self.bce(y_pred, y)         # scalar — loss
            loss.backward()                      # populate .grad tensors
            self.step()                          # update params & zero grads

            self.loss_history.append(loss.item())

            if epoch % print_every == 0:
                acc = ((y_pred.detach() >= 0.5).float() == y).float().mean()
                print(f"    Epoch {epoch:6d} | Loss: {loss.item():.6f} | "
                      f"Acc: {acc.item()*100:.1f}%")

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Threshold probabilities at 0.5 to get class labels."""
        with torch.no_grad():
            probs = self.forward(X)
            return (probs >= 0.5).float()        # (N,1) binary labels


def demo_mlp_scratch_xor() -> MLPScratch:
    """Solve XOR with MLPScratch — shows MLP overcomes Perceptron limitation."""
    print("\n" + "="*65)
    print("SECTION C — MLP from Scratch: XOR Solver")
    print("="*65)

    X_xor = torch.tensor([[0., 0.],
                           [0., 1.],
                           [1., 0.],
                           [1., 1.]], dtype=torch.float64)
    y_xor = torch.tensor([[0.], [1.], [1.], [0.]], dtype=torch.float64)

    mlp = MLPScratch(n_input=2, n_hidden=4, n_output=1, lr=0.5)
    mlp.train_model(X_xor, y_xor, n_epochs=10000, print_every=2000)

    preds = mlp.predict(X_xor)
    acc   = (preds == y_xor).float().mean().item()
    print(f"\n  Final XOR Accuracy: {acc*100:.1f}%\n")

    print("  Input → Pred | True | Status")
    print("  " + "-"*32)
    for i in range(len(X_xor)):
        xi  = X_xor[i].int().tolist()
        yi  = int(y_xor[i].item())
        p   = int(preds[i].item())
        mrk = "✓" if p == yi else "✗"
        print(f"  {xi} → pred={p}  | true={yi}  | {mrk}")

    return mlp


# ═════════════════════════════════════════════════════════════════════════════
# SECTION D — PRODUCTION MLP (nn.Module)
# Uses PyTorch's standard building blocks — same computation, cleaner API
# ═════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """
    Configurable Multilayer Perceptron for binary classification.

    Architecture:
        Input → [Linear → ReLU] × len(hidden_sizes) → Linear → Sigmoid

    Parameters
    ----------
    input_size   : int        — number of input features
    hidden_sizes : list[int]  — width of each hidden layer, e.g. [64, 64, 32]
    output_size  : int        — number of output neurons (1 for binary)
    """

    def __init__(self, input_size: int,
                 hidden_sizes: list,
                 output_size: int = 1):
        super(MLP, self).__init__()

        layers = []
        in_size = input_size

        # Build hidden layers dynamically
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))   # Wˡx + bˡ
            layers.append(nn.ReLU())                # max(0, z)
            in_size = h

        # Output layer
        layers.append(nn.Linear(in_size, output_size))
        layers.append(nn.Sigmoid())   # squeeze to (0,1) for probability

        self.net = nn.Sequential(*layers)

        # Weight initialization — Xavier Uniform is the PyTorch default for
        # Linear layers but we make it explicit for clarity
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for Linear layers, zeros for biases."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x       : (batch_size, input_size)
        returns : (batch_size, output_size)  — probabilities ŷ ∈ (0,1)
        """
        return self.net(x)

    def count_parameters(self) -> int:
        """Count total trainable parameters: Σₗ nˡ(nˡ⁻¹ + 1)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict_classes(self, x: torch.Tensor) -> torch.Tensor:
        """Return binary class labels by thresholding probabilities at 0.5."""
        with torch.no_grad():
            return (self.forward(x) >= 0.5).long()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION E — DATA PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def build_data_pipeline(n_samples: int = 1000, noise: float = 0.25,
                        test_size: float = 0.2, batch_size: int = 32):
    """
    Generate the make_moons dataset and wrap in PyTorch DataLoaders.

    make_moons creates two interleaving half-circles — a classic non-linearly
    separable binary classification problem that requires an MLP to solve.

    Steps:
      1. Generate data with sklearn
      2. Train/val split with stratification
      3. Standardize features (Z-score) on train statistics
      4. Convert to float32 PyTorch tensors
      5. Wrap in TensorDataset → DataLoader

    Returns
    -------
    train_loader, val_loader : DataLoader objects
    X_tr, X_va              : feature tensors (float32)
    y_tr, y_va              : label tensors  (float32, shape N×1)
    scaler                  : fitted StandardScaler for inverse transform
    """
    print("\n" + "="*65)
    print("SECTION E — Data Pipeline: make_moons")
    print("="*65)

    # ── 1. Generate ──────────────────────────────────────────────────────────
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=SEED)
    # X: (n_samples, 2), y: (n_samples,) ∈ {0, 1}

    # ── 2. Split ─────────────────────────────────────────────────────────────
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y,
        test_size=test_size,
        random_state=SEED,
        stratify=y      # ensures equal class ratio in both splits
    )

    # ── 3. Standardize ───────────────────────────────────────────────────────
    # IMPORTANT: fit only on train, transform both — prevents data leakage
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)   # μ_train=0, σ_train=1
    X_va   = scaler.transform(X_va)       # same stats applied to val

    # ── 4. Tensors ───────────────────────────────────────────────────────────
    # float32 is the standard dtype for GPU/PyTorch operations
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)  # (N,) → (N,1)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32).unsqueeze(1)

    # ── 5. DataLoaders ───────────────────────────────────────────────────────
    train_ds     = TensorDataset(X_tr_t, y_tr_t)
    val_ds       = TensorDataset(X_va_t, y_va_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    print(f"\n  Total samples : {n_samples}  (noise={noise})")
    print(f"  Train         : {len(X_tr_t)} samples")
    print(f"  Val           : {len(X_va_t)} samples")
    print(f"  Feature shape : {X_tr_t.shape}  (standardized)")
    print(f"  Label shape   : {y_tr_t.shape}  (unsqueezed to col vector)")
    print(f"  Batches/epoch : {len(train_loader)}")

    return (train_loader, val_loader,
            X_tr_t, X_va_t,
            y_tr_t, y_va_t,
            scaler)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION E — TRAINING ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def train_epoch(model:     nn.Module,
                loader:    DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device:    torch.device) -> tuple:
    """
    Run one full epoch of training.

    Training mode activates dropout / batchnorm differently from eval mode.
    We explicitly call model.train() here to be safe even if called repeatedly.

    Returns
    -------
    avg_loss : float — mean BCE loss over all batches
    accuracy : float — fraction of correct predictions
    """
    model.train()
    total_loss    = 0.0
    total_correct = 0
    total_n       = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)   # move to GPU if available
        y_batch = y_batch.to(device)

        # ── Forward ──────────────────────────────────────────────────────────
        optimizer.zero_grad()           # clear stale gradients from prev step
        y_pred = model(X_batch)         # (batch, 1) — probabilities

        # ── Loss ─────────────────────────────────────────────────────────────
        loss = criterion(y_pred, y_batch)   # scalar BCE

        # ── Backward ─────────────────────────────────────────────────────────
        loss.backward()                 # populate .grad for all parameters

        # ── Optimizer Step ───────────────────────────────────────────────────
        optimizer.step()                # Adam update: w ← w - lr*m̂/√v̂

        # ── Accumulate Metrics ───────────────────────────────────────────────
        total_loss    += loss.item() * X_batch.size(0)      # un-normalize
        preds          = (y_pred >= 0.5).float()
        total_correct += (preds == y_batch).sum().item()
        total_n       += X_batch.size(0)

    return total_loss / total_n, total_correct / total_n


@torch.no_grad()   # decorator: disables autograd for entire function
def evaluate_epoch(model:     nn.Module,
                   loader:    DataLoader,
                   criterion: nn.Module,
                   device:    torch.device) -> tuple:
    """
    Evaluate the model on a DataLoader without computing gradients.

    @torch.no_grad() is equivalent to wrapping the whole body in
    `with torch.no_grad():` — it reduces memory usage and speeds up inference.
    """
    model.eval()    # sets layers like Dropout/BatchNorm to inference mode
    total_loss    = 0.0
    total_correct = 0
    total_n       = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred         = model(X_batch)
        loss           = criterion(y_pred, y_batch)
        total_loss    += loss.item() * X_batch.size(0)
        preds          = (y_pred >= 0.5).float()
        total_correct += (preds == y_batch).sum().item()
        total_n       += X_batch.size(0)

    return total_loss / total_n, total_correct / total_n


def train_model(model:        nn.Module,
                train_loader: DataLoader,
                val_loader:   DataLoader,
                n_epochs:     int   = 150,
                lr:           float = 1e-3) -> dict:
    """
    Complete training pipeline.
    Uses Adam optimizer and BCELoss.
    Tracks best model state by validation accuracy.

    Returns
    -------
    history : dict of lists with keys:
        'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """
    print("\n" + "="*65)
    print("SECTION D — Production MLP Training: make_moons")
    print("="*65)
    print(f"\n  Architecture : {model}")
    print(f"  Parameters   : {model.count_parameters():,}")
    print(f"  Optimizer    : Adam  (lr={lr}, weight_decay=1e-4)")
    print(f"  Loss         : BCELoss")
    print(f"  Device       : {DEVICE}\n")

    model     = model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    history  = {"train_loss": [], "val_loss": [],
                "train_acc":  [], "val_acc":  []}
    best_acc = 0.0
    best_wts = None

    # ── Header ───────────────────────────────────────────────────────────────
    print(f"  {'Epoch':>6} │ {'T-Loss':>8} │ {'T-Acc':>7} │ "
          f"{'V-Loss':>8} │ {'V-Acc':>7}")
    print("  " + "─"*48)

    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion,
                                      optimizer, DEVICE)
        va_loss, va_acc = evaluate_epoch(model, val_loader, criterion, DEVICE)

        history["train_loss"].append(tr_loss)
        history["val_loss"  ].append(va_loss)
        history["train_acc" ].append(tr_acc)
        history["val_acc"   ].append(va_acc)

        # Save best model weights
        if va_acc > best_acc:
            best_acc = va_acc
            # Deep copy state dict to CPU to avoid CUDA memory issues
            best_wts = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Print every 25 epochs
        if epoch % 25 == 0 or epoch == 1:
            print(f"  {epoch:>6} │ {tr_loss:>8.4f} │ {tr_acc*100:>6.2f}% │ "
                  f"{va_loss:>8.4f} │ {va_acc*100:>6.2f}%")

    # Restore the best model found during training
    model.load_state_dict(best_wts)
    print(f"\n  ✓ Training complete. Best Val Accuracy: {best_acc*100:.2f}%")

    return history


# ═════════════════════════════════════════════════════════════════════════════
# SECTION F — EVALUATION REPORT
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_final(model: nn.Module,
                   X_val: torch.Tensor,
                   y_val: torch.Tensor) -> None:
    """
    Print a full evaluation report on the validation set:
    accuracy, per-class precision/recall/F1, confusion matrix.
    """
    model.eval()
    model = model.to(DEVICE)

    probs  = model(X_val.to(DEVICE)).cpu().numpy().flatten()
    preds  = (probs >= 0.5).astype(int)
    labels = y_val.cpu().numpy().flatten().astype(int)

    print("\n" + "="*65)
    print("SECTION F — Final Evaluation (Validation Set)")
    print("="*65)
    print(f"\n  Overall Accuracy: {accuracy_score(labels, preds)*100:.2f}%\n")
    print("  Classification Report:")
    print(classification_report(labels, preds,
                                target_names=["Class 0 (Moon A)",
                                              "Class 1 (Moon B)"]))

    cm = confusion_matrix(labels, preds)
    print("  Confusion Matrix:")
    print(f"    Predicted →  Class 0   Class 1")
    print(f"    Actual Class 0:  {cm[0,0]:5d}     {cm[0,1]:5d}")
    print(f"    Actual Class 1:  {cm[1,0]:5d}     {cm[1,1]:5d}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION G — VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def plot_decision_boundary(model: nn.Module,
                           X: torch.Tensor, y: torch.Tensor,
                           title: str, ax: plt.Axes) -> None:
    """
    Visualize the decision boundary of a 2D binary classifier.

    Creates a dense mesh grid over the feature space, runs inference on
    every point, and colors the background by predicted class.
    """
    margin = 0.6
    x1_min = X[:, 0].min().item() - margin
    x1_max = X[:, 0].max().item() + margin
    x2_min = X[:, 1].min().item() - margin
    x2_max = X[:, 1].max().item() + margin

    # Build a 300×300 mesh grid
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 300),
                            np.linspace(x2_min, x2_max, 300))
    grid = torch.tensor(
        np.c_[xx1.ravel(), xx2.ravel()],
        dtype=torch.float32
    ).to(DEVICE)

    model.eval()
    with torch.no_grad():
        Z = model(grid).cpu().numpy().reshape(xx1.shape)

    # Color regions by predicted probability
    ax.contourf(xx1, xx2, Z, alpha=0.35, cmap="RdBu", levels=50)
    ax.contour(xx1, xx2, Z, levels=[0.5], colors="black",
               linewidths=2.0, linestyles="--")

    colors = ["#e74c3c", "#2980b9"]
    y_flat = y.squeeze().cpu().numpy().astype(int)
    for cls, col in enumerate(colors):
        mask = y_flat == cls
        ax.scatter(X[mask, 0].cpu(), X[mask, 1].cpu(),
                   c=col, s=25, alpha=0.75,
                   edgecolors="white", linewidths=0.4,
                   label=f"Class {cls}")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")


def build_visualization(p_and:   Perceptron,
                         p_xor:   Perceptron,
                         scratch: MLPScratch,
                         mlp:     MLP,
                         history: dict,
                         X_tr:    torch.Tensor,
                         y_tr:    torch.Tensor,
                         X_va:    torch.Tensor,
                         y_va:    torch.Tensor) -> None:
    """Create a 2×3 dashboard of all results."""
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        "Phase 1 — Topic 1: Perceptron & MLP  │  Results Dashboard",
        fontsize=14, fontweight="bold", y=1.00
    )
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    # ── Panel 1: Perceptron AND convergence ──────────────────────────────────
    ax1.plot(p_and.errors_, color="#e67e22", linewidth=2, marker="o", ms=4)
    ax1.set_title("Perceptron — AND Gate Convergence", fontweight="bold", fontsize=10)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Misclassifications")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Perceptron XOR failure ──────────────────────────────────────
    ax2.plot(p_xor.errors_[:100], color="#e74c3c", linewidth=1.5, alpha=0.8)
    ax2.set_title("Perceptron — XOR Failure (first 100 epochs)", fontweight="bold", fontsize=10)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Misclassifications")
    ax2.axhline(y=0, color="green", linestyle="--", alpha=0.6, label="Convergence target")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: MLPScratch XOR loss ─────────────────────────────────────────
    ax3.plot(scratch.loss_history, color="#8e44ad", linewidth=1.2, alpha=0.9)
    ax3.set_title("MLP Scratch — XOR Training Loss", fontweight="bold", fontsize=10)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("BCE Loss")
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Production MLP loss curves ──────────────────────────────────
    epochs = range(1, len(history["train_loss"]) + 1)
    ax4.plot(epochs, history["train_loss"], label="Train", color="#e74c3c", lw=2)
    ax4.plot(epochs, history["val_loss"],   label="Val",   color="#2980b9", lw=2, ls="--")
    ax4.set_title("Production MLP — Loss Curves", fontweight="bold", fontsize=10)
    ax4.set_xlabel("Epoch"); ax4.set_ylabel("BCE Loss")
    ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3)

    # ── Panel 5: Production MLP accuracy curves ───────────────────────────────
    ax5.plot(epochs, [a*100 for a in history["train_acc"]],
             label="Train", color="#e74c3c", lw=2)
    ax5.plot(epochs, [a*100 for a in history["val_acc"]],
             label="Val",   color="#2980b9", lw=2, ls="--")
    ax5.set_title("Production MLP — Accuracy Curves", fontweight="bold", fontsize=10)
    ax5.set_xlabel("Epoch"); ax5.set_ylabel("Accuracy (%)")
    ax5.legend(fontsize=9); ax5.grid(True, alpha=0.3)

    # ── Panel 6: Decision boundary on validation set ─────────────────────────
    plot_decision_boundary(mlp, X_va, y_va,
                           "Production MLP — Decision Boundary (Val)", ax6)

    plt.tight_layout()
    save_path = os.path.join(RESULTS, "01_perceptron_mlp_dashboard.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  [VIZ] Dashboard saved → {save_path}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "▓"*65)
    print("  DEEP LEARNING MASTERY REPOSITORY")
    print("  Phase 1 — Topic 1: Perceptron & Multilayer Perceptron (MLP)")
    print("▓"*65)

    # ── A: AND gate ───────────────────────────────────────────────────────────
    p_and = demo_perceptron_and()

    # ── B: XOR failure ───────────────────────────────────────────────────────
    p_xor = demo_perceptron_xor()

    # ── C: MLP scratch on XOR ────────────────────────────────────────────────
    mlp_scratch = demo_mlp_scratch_xor()

    # ── D+E: Data pipeline ───────────────────────────────────────────────────
    (train_loader, val_loader,
     X_tr, X_va,
     y_tr, y_va, scaler) = build_data_pipeline(
        n_samples=N_SAMPLES,
        noise=NOISE,
        test_size=0.2,
        batch_size=BATCH_SIZE
    )

    # ── D: Build production MLP ──────────────────────────────────────────────
    mlp = MLP(input_size=2, hidden_sizes=[64, 64, 32], output_size=1)

    # ── E: Train ─────────────────────────────────────────────────────────────
    history = train_model(
        mlp, train_loader, val_loader,
        n_epochs=EPOCHS, lr=LR
    )

    # ── F: Evaluate ──────────────────────────────────────────────────────────
    evaluate_final(mlp, X_va, y_va)

    # ── G: Visualize ─────────────────────────────────────────────────────────
    build_visualization(p_and, p_xor, mlp_scratch, mlp,
                        history, X_tr, y_tr, X_va, y_va)

    print("\n" + "▓"*65)
    print("  ✓  All sections complete.")
    print(f"  ✓  Results saved to ./{RESULTS}/")
    print("▓"*65 + "\n")


if __name__ == "__main__":
    main()
