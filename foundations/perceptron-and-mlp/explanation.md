# Code Explanation: Perceptron & Multilayer Perceptron (MLP)

**Phase 1 — Topic 1 | `implementation.py` walkthrough**

This document explains every design decision in the code — why each function was chosen,
what each tensor shape means, and the mathematical reasoning behind each operation.

---

## Table of Contents

1. [Imports & Global Configuration](#1-imports--global-configuration)
2. [Section A — Perceptron Class](#2-section-a--perceptron-class)
3. [Section B — XOR Failure Demo](#3-section-b--xor-failure-demo)
4. [Section C — MLPScratch (Raw Tensors)](#4-section-c--mlpscratch-raw-tensors)
5. [Section D — Production MLP (nn.Module)](#5-section-d--production-mlp-nnmodule)
6. [Section E — Data Pipeline](#6-section-e--data-pipeline)
7. [Section E — Training Engine](#7-section-e--training-engine)
8. [Section F — Evaluation](#8-section-f--evaluation)
9. [Section G — Visualization](#9-section-g--visualization)
10. [Common Pitfalls and Why the Code Avoids Them](#10-common-pitfalls-and-why-the-code-avoids-them)

---

## 1. Imports & Global Configuration

```python
import warnings
warnings.filterwarnings("ignore")
```
**Why:** scikit-learn occasionally raises `ConvergenceWarning` when the Perceptron doesn't
converge (which is intentional in Section B). Suppressing these keeps the output clean.
In production you would remove this and handle warnings explicitly.

```python
import matplotlib
matplotlib.use("Agg")
```
**Why:** The default Matplotlib backend (`TkAgg`) tries to open a GUI window. On servers,
CI/CD environments, or remote machines, there is no display — this line switches to a
non-interactive backend (`Agg`) that writes to file instead of opening a window.
Must be called **before** `import matplotlib.pyplot`.

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
**Why:** A single line that makes all code device-agnostic. When we later write
`tensor.to(DEVICE)` or `model.to(DEVICE)`, the tensor/model automatically moves to GPU
if available, or stays on CPU otherwise. This pattern is the PyTorch standard.

```python
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```
**Why:** We set three separate seeds because PyTorch and NumPy maintain independent
random number generators. Without all three, `np.random.randn()` and
`torch.randn()` would still be non-deterministic relative to each other.
`manual_seed_all` seeds all CUDA devices (important for multi-GPU setups).

---

## 2. Section A — Perceptron Class

### Class-Level Design

```python
class Perceptron:
    def __init__(self, learning_rate: float = 0.1, n_epochs: int = 100):
        self.weights_    = None      # set in fit()
        self.bias_       = 0.0
        self.errors_     = []
```

**Why `weights_ = None` (not initialized here)?**  
We don't know the number of features until `fit()` is called.  
The trailing underscore `_` on `weights_` and `errors_` is a scikit-learn convention:
attributes that are set during `fit()` (not at construction) are suffixed with `_`.

**Why store `errors_`?**  
Tracking misclassification count per epoch lets us (a) detect early convergence and
(b) plot the convergence curve in Section G.

### The Heaviside Step Function

```python
@staticmethod
def _heaviside(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0.0, 1, 0).astype(int)
```

**Why `np.where` instead of `if z >= 0`?**  
`np.where` is vectorized — it processes entire arrays at once (C-speed). Using a Python
`if` statement would require a Python loop, which is ~100× slower.

**Why threshold at `z >= 0` (not `z > 0`)?**  
The original Perceptron uses `z ≥ 0` (inclusive). The choice at exactly z=0 is a
convention; using strict `>` would make the boundary asymmetric.

**Why `@staticmethod`?**  
The method doesn't use `self` (it has no instance state). Marking it `@staticmethod`
prevents Python from passing `self` implicitly and makes the call slightly faster.

### The `fit` Method

```python
n_samples, n_features = X.shape
self.weights_ = np.zeros(n_features, dtype=np.float64)
```

**Why initialize weights to zero?**  
For the Perceptron, zero initialization is fine because the step function's non-linearity
breaks the symmetry even with zero weights (any non-zero input immediately creates a
gradient). This is **not** safe for MLPs (see Section C). For zero-init in MLP:
all neurons compute the same gradient → they learn identical features → wasted capacity.

**Why `float64`?**  
The AND/XOR datasets are tiny (4 samples). `float64` precision avoids any rounding
edge cases in the exact threshold computation `z >= 0`. For large neural networks,
`float32` is preferred for memory and speed.

```python
for xi, yi in zip(X, y):
    z     = np.dot(xi, self.weights_) + self.bias_
    y_hat = self._heaviside(np.array([z]))[0]
```

**Why iterate sample-by-sample instead of batch?**  
The Perceptron uses **online learning** (also called stochastic learning): each sample
is processed one at a time, and weights are updated after every single sample.
This is a fundamental property of the original Perceptron algorithm.
Batch updates would give different convergence behavior.

**Why `np.array([z])` before calling `_heaviside`?**  
`z` is a scalar after `np.dot`. `_heaviside` calls `np.where`, which expects array-like
input. Wrapping in `np.array([z])` ensures this. The `[0]` extracts the scalar back.

```python
error = int(yi) - int(y_hat)
if error != 0:
    self.weights_ += self.lr * error * xi
    self.bias_    += self.lr * error
    n_errors      += 1
```

**Why check `if error != 0` before updating?**  
The Perceptron update rule is: `Δw = η(y - ŷ)x`. When `y == ŷ`, `error = 0`,
so `Δw = 0` — no update occurs. Checking explicitly skips the array operations
and increments `n_errors` only on actual mistakes (needed for convergence detection).

**Why add `n_errors` only when updating (not always)?**  
`n_errors` tracks **actual weight updates**, which equals the number of misclassifications.
This count is what the convergence theorem bounds: `T ≤ (R/γ)²`.

### The `predict` Method

```python
def predict(self, X: np.ndarray) -> np.ndarray:
    z = X @ self.weights_ + self.bias_   # (n_samples,)
    return self._heaviside(z)
```

**Why `X @ self.weights_` instead of a loop?**  
Matrix-vector multiplication `X @ w` is vectorized over all samples simultaneously.
`X` is `(n_samples, n_features)`, `w` is `(n_features,)`, so `X @ w` gives
`(n_samples,)` — one score per sample, computed in parallel.

**Why not call `fit`'s inner loop here?**  
`predict` must be pure inference — no weight updates. Separating `fit` and `predict`
is the standard ML API design (used by scikit-learn, PyTorch, TensorFlow, etc.)

---

## 3. Section B — XOR Failure Demo

```python
p = Perceptron(learning_rate=0.1, n_epochs=1000)
p.fit(X_xor, y_xor)
```

**Why 1000 epochs here vs 100 for AND?**  
For XOR, the Perceptron will never converge (no valid solution exists). We run more
epochs to clearly show that errors never reach zero — not just that it hasn't converged yet.

**Why only 4 training samples?**  
XOR is a complete truth table (all 2-bit combinations). These 4 samples fully define
the problem. More samples would just repeat these 4 patterns.

---

## 4. Section C — MLPScratch (Raw Tensors)

This section intentionally avoids `nn.Module` to show the underlying mechanics.

### Initialization

```python
std1 = np.sqrt(2.0 / n_input)
self.W1 = torch.randn(n_input, n_hidden, dtype=torch.float64) * std1
```

**Why `sqrt(2 / fan_in)` (He/Kaiming initialization)?**  
For ReLU activations, He initialization is the theoretically correct choice. The factor
`2/fan_in` compensates for the fact that ReLU zeros out roughly half of the neurons
(the negative half), effectively halving the variance. Without this scaling:
- Too small init → signals shrink layer by layer → **vanishing gradients**
- Too large init → signals explode → **exploding gradients**

**Why `torch.float64` here (not float32)?**  
The XOR dataset has only 4 samples. With `float32`, numerical precision issues can cause
the optimizer to get stuck. `float64` gives us 15-digit precision, making convergence
reliable for this tiny example. Production code uses `float32` (see MLP class).

```python
self.b1 = torch.zeros(1, n_hidden, dtype=torch.float64)
```

**Why shape `(1, n_hidden)` for bias, not `(n_hidden,)`?**  
During `forward`, the computation is:
```python
self.z1 = X @ self.W1 + self.b1
# X:  (N, n_input)
# W1: (n_input, n_hidden)
# X@W1: (N, n_hidden)
# b1:   (1, n_hidden)  ← broadcasts to (N, n_hidden)
```
Shape `(1, n_hidden)` allows PyTorch broadcasting to expand `b1` across all N samples
automatically. Shape `(n_hidden,)` would also work (1D broadcasting), but `(1, n_hidden)`
makes the broadcasting intent explicit.

```python
for p in [self.W1, self.b1, self.W2, self.b2]:
    p.requires_grad_(True)
```

**Why `requires_grad_(True)` (note the trailing underscore)?**  
`requires_grad_(True)` is an **in-place** operation (the underscore means in-place in
PyTorch). It tells the autograd engine to record all operations on this tensor in a
computation graph, enabling `loss.backward()` to compute `∂L/∂p` for each parameter.
Without this, calling `loss.backward()` would raise a RuntimeError.

### Forward Pass

```python
self.z1 = X  @ self.W1 + self.b1   # (N,2)×(2,4) + (1,4) → (N,4)
self.a1 = self.relu(self.z1)        # (N,4) element-wise
self.z2 = self.a1 @ self.W2 + self.b2  # (N,4)×(4,1) + (1,1) → (N,1)
self.a2 = self.sigmoid(self.z2)     # (N,1) probabilities
```

**Why store `z1`, `a1`, `z2` as instance attributes?**  
In this scratch implementation, we store intermediate values to allow manual inspection.
In `MLPScratch`, PyTorch's autograd handles gradient computation automatically, but
storing them makes the computation graph explicit for educational purposes.

**Why `ReLU` for hidden, `Sigmoid` for output?**  
- **ReLU in hidden layer:** Efficient non-linear activation; gradients don't saturate
  for positive values (unlike sigmoid/tanh, where large inputs cause near-zero gradients).
- **Sigmoid in output:** We need a probability ∈ (0, 1) for binary BCE loss.
  Sigmoid maps any real number to this range.

**Why `torch.clamp(z, min=0.0)` for ReLU instead of `torch.relu(z)` or `torch.max(z, 0)`?**  
All three are equivalent. `torch.clamp` is used here to make the mathematical operation
explicit: we are clamping the minimum value to 0, which is exactly max(0, z).

### Loss Function

```python
eps = 1e-9
loss = -(y_true * torch.log(y_pred + eps) +
         (1 - y_true) * torch.log(1 - y_pred + eps))
```

**Why `eps = 1e-9`?**  
`torch.log(0.0)` returns `-inf`. If the model is very confident and wrong (e.g., predicts
ŷ ≈ 0.0 when true y = 1), then `log(ŷ + 0) = log(0) = -∞`. Adding `eps` prevents
the loss from becoming infinite, which would cause NaN gradients and crash training.

**Why Manual BCE instead of `nn.BCELoss()`?**  
To show the exact mathematical formula in executable code. The behavior is identical.

### Parameter Update

```python
def step(self):
    with torch.no_grad():
        for param in [self.W1, self.b1, self.W2, self.b2]:
            param -= self.lr * param.grad
            param.grad.zero_()
```

**Why `torch.no_grad()` inside the update step?**  
Without `no_grad()`, `param -= self.lr * param.grad` would itself be recorded in the
computation graph. This is wasteful (we don't want to backprop through the update itself)
and can cause errors. `no_grad()` tells PyTorch: "don't track this operation."

**Why `param.grad.zero_()`?**  
PyTorch **accumulates** gradients by default — each `loss.backward()` call **adds** to
`param.grad`, not replaces it. If we don't zero gradients after each update, the gradient
from step 2 would be `∇step1 + ∇step2`, giving an incorrect update direction.
The trailing `_` means in-place zeroing (no new tensor allocated).

### Training Loop Structure

```python
y_pred = self.forward(X)    # 1. Forward
loss   = self.bce(...)      # 2. Loss
loss.backward()             # 3. Backward — populates .grad
self.step()                 # 4. Update + zero grads
```

**Why this exact order?**  
This is the universal training loop structure in PyTorch. Breaking the order causes errors:
- `backward()` before `forward()`: no computation graph exists yet → error
- Update before `backward()`: gradients are all zero → model never learns
- Forgetting `zero_()`: gradients accumulate → wrong update direction

---

## 5. Section D — Production MLP (nn.Module)

### Class Design

```python
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=1):
        super(MLP, self).__init__()
```

**Why inherit from `nn.Module`?**  
`nn.Module` provides the infrastructure for:
- `parameters()` iterator (for optimizers)
- `state_dict()` / `load_state_dict()` (saving/loading models)
- `.to(device)` (moving all parameters to GPU)
- `train()` / `eval()` mode switching
- Automatic registration of sub-modules

Without `nn.Module`, we would need to implement all of this manually (as in `MLPScratch`).

**Why call `super(MLP, self).__init__()`?**  
`nn.Module.__init__()` initializes the internal `_parameters`, `_modules`, and
`_buffers` dicts that `nn.Module` needs to track registered parameters. Without it,
operations like `model.parameters()` would fail silently or crash.

```python
for h in hidden_sizes:
    layers.append(nn.Linear(in_size, h))
    layers.append(nn.ReLU())
    in_size = h
```

**Why build layers dynamically in a list?**  
This makes the MLP configurable: `MLP(2, [64, 64, 32], 1)` creates a 3-hidden-layer
MLP, while `MLP(2, [128], 1)` creates a 1-hidden-layer MLP. The same code handles
both — no copy-pasting `nn.Linear` calls.

**Why `in_size = h` at the end of each loop iteration?**  
The output dimension of one layer becomes the input dimension of the next.
After the loop, `in_size` holds the width of the last hidden layer, which is
the correct input size for the output layer.

```python
self.net = nn.Sequential(*layers)
```

**Why `nn.Sequential` instead of registering layers individually?**  
`nn.Sequential` chains modules so that `self.net(x)` automatically applies them in order.
The `*layers` unpacks the list into positional arguments. This is more compact than:
```python
self.layer1 = layers[0]
self.layer2 = layers[1]
# ...
```
and also registers all modules under `self.net`, so `model.parameters()` finds them.

### Weight Initialization

```python
def _init_weights(self):
    for module in self.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
```

**Why `xavier_uniform_` instead of the PyTorch default (Kaiming uniform)?**  
PyTorch's `nn.Linear` default is Kaiming uniform (He init), which is optimal for ReLU.
We override with Xavier (Glorot) here because Xavier is the most commonly seen init in
literature and works well for sigmoid/tanh outputs too. Both work fine for this problem.

Xavier uniform initializes weights from a uniform distribution:
```
W ~ Uniform(-√(6/(fan_in + fan_out)), +√(6/(fan_in + fan_out)))
```

**Why `zeros_` for biases?**  
Zero-bias initialization is standard. Biases don't suffer from the symmetry-breaking
problem (their gradients differ from weight gradients), and zero init is simple and works.

### Forward Method

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)
```

**Why is `forward` so simple here?**  
`nn.Sequential` handles the entire chain. Behind the scenes, `self.net(x)` calls
`forward()` on each sub-module in order: `Linear → ReLU → Linear → ReLU → Linear → Sigmoid`.

**Why define `forward` at all — why not just `__call__`?**  
PyTorch's `nn.Module.__call__` calls `self.forward(x)` internally, but also runs hooks
(for debugging/profiling). Overriding `forward` is the correct pattern; overriding
`__call__` directly would bypass the hook machinery.

### Parameter Counter

```python
def count_parameters(self) -> int:
    return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

**Why `p.numel()`?**  
`numel()` returns the total number of elements in a tensor (product of all dimensions).
For a weight matrix `W ∈ ℝ^(64 × 2)`, `W.numel() = 128`. Summing over all parameters
gives the total parameter count.

**Why `if p.requires_grad`?**  
This filters to only **trainable** parameters. Frozen layers (common in transfer learning)
have `requires_grad = False` and shouldn't be counted in the trainable parameter budget.

---

## 6. Section E — Data Pipeline

### Why `make_moons`?

```python
X, y = make_moons(n_samples=n_samples, noise=noise, random_state=SEED)
```

`make_moons` creates two interleaving half-circles — classes are entangled and
cannot be separated by any straight line (not linearly separable). This makes it
a perfect testbed for MLP's ability to learn curved decision boundaries.
It's also 2D, so we can visualize the decision boundary directly.

### Train/Val Split

```python
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=test_size, random_state=SEED, stratify=y
)
```

**Why `stratify=y`?**  
Without stratification, by random chance, the split might produce an imbalanced
validation set (e.g., 70% class 0, 30% class 1). `stratify=y` ensures both splits
preserve the original class ratio (approximately 50/50 for `make_moons`).
This prevents misleadingly high accuracy from class imbalance.

### StandardScaler

```python
scaler = StandardScaler()
X_tr   = scaler.fit_transform(X_tr)   # μ and σ computed on train
X_va   = scaler.transform(X_va)       # same μ and σ applied
```

**Why fit only on training data?**  
This is the **data leakage** rule: the validation set must be treated as unseen data.
If we fit the scaler on all data (including validation), the scaler would have access to
validation statistics during training — this is cheating and inflates validation metrics.

**Why standardize at all?**  
Without standardization, features with large absolute values (e.g., range 0–10000) will
dominate the dot product `Wx + b`, while small-scale features contribute little.
Standardization makes all features contribute equally at initialization, and allows
the optimizer to move efficiently in all directions without needing feature-specific
learning rates.

### Label Shape: `.unsqueeze(1)`

```python
y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)  # (N,) → (N,1)
```

**Why unsqueeze?**  
`torch.tensor(y_tr)` creates shape `(N,)` — a 1D tensor. Our model outputs shape `(N, 1)`
(a 2D column vector). PyTorch's `nn.BCELoss` computes the loss element-wise between
`y_pred` and `y_true` — **their shapes must match exactly**.
`unsqueeze(1)` inserts a dimension at position 1: `(N,)` → `(N, 1)`.

If we skip this, we get the silent **broadcasting bug**: `BCELoss` would broadcast
`(N,)` against `(N, 1)` to `(N, N)`, computing N² loss values instead of N, and the
average would still be finite — so training might appear to work but be completely wrong.

### DataLoader

```python
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, ...)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, ...)
```

**Why `shuffle=True` for train, `shuffle=False` for val?**  
- Training: Shuffling ensures each mini-batch contains a random mix of samples.
  Without shuffling, early batches might contain only class 0 and late batches only
  class 1, causing biased gradient updates and poor convergence.
- Validation: Order doesn't matter for evaluation (we're not updating weights), and
  keeping consistent order makes results reproducible and comparable across experiments.

**Why mini-batches at all?**  
- **Full batch (all data at once):** Exact gradient, but slow per update and high memory.
- **Stochastic (one sample):** Very noisy gradients, but many updates per epoch.
- **Mini-batch:** Best of both — parallelizable on GPU, reasonably low variance.
  `batch_size=32` is a common default; 32–256 are typical values.

---

## 7. Section E — Training Engine

### `train_epoch` Function

```python
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
```

**Why `model.train()`?**  
Calling `.train()` sets the model to training mode. This affects layers that behave
differently during training vs. inference:
- **Dropout:** Active during training (randomly zeros neurons), disabled during eval
- **BatchNorm:** Uses batch statistics during training, running statistics during eval
This model has neither (yet), but calling `.train()` is still correct practice and
ensures correctness when the architecture changes.

```python
optimizer.zero_grad()    # BEFORE forward pass
y_pred = model(X_batch)
loss   = criterion(y_pred, y_batch)
loss.backward()
optimizer.step()
```

**Why `zero_grad()` before, not after, the forward pass?**  
Gradients must be zero before `backward()` adds new gradients to them. Placing
`zero_grad()` before the forward pass (rather than after `optimizer.step()`) is the
safest pattern — it explicitly clears any stale gradients from previous iterations,
even if a previous `backward()` raised an exception and `step()` was never called.

**Why `loss.item()` when accumulating?**  
`loss` is a tensor tracked by autograd. Calling `.item()` detaches it to a plain
Python float. Without `.item()`, `total_loss` would accumulate a chain of tensor
operations in memory — a serious memory leak over many batches.

```python
total_loss += loss.item() * X_batch.size(0)
```

**Why multiply by batch size?**  
`criterion(y_pred, y_batch)` returns the **mean** loss over the batch.
To compute the epoch-level average correctly, we un-normalize by multiplying by batch size,
then divide by `total_n` at the end. Without this, the last (possibly smaller) batch
would be weighted equally to full batches — giving a slightly wrong epoch average.

### `evaluate_epoch` Function

```python
@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
```

**Why both `@torch.no_grad()` and `model.eval()`? Are they redundant?**  
No — they do different things:
- `model.eval()`: Changes the **behavior** of specific layers (Dropout off, BatchNorm
  uses running stats).
- `@torch.no_grad()`: Disables gradient **tracking** (saves memory, speeds up inference).
Both are needed. `model.eval()` without `no_grad()` still builds a computation graph
(wasted memory). `no_grad()` without `model.eval()` still applies dropout.

### `train_model` Function

```python
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
```

**Why Adam instead of SGD?**  
Adam (Adaptive Moment Estimation) maintains per-parameter adaptive learning rates.
For a 2-feature MLP on `make_moons`, SGD with a fixed learning rate would require
more epochs and careful tuning. Adam converges faster with minimal tuning.
Covered in depth in Topic 5 (Optimizers).

**Why `weight_decay=1e-4`?**  
`weight_decay` adds an L2 regularization penalty: `L_total = L_data + λ‖w‖²`.
This penalizes large weights, discouraging overfitting.
`1e-4` is a mild regularization — light enough to not hurt convergence but enough
to keep weights from growing arbitrarily large.
(Covered in depth in Topic 5 — Regularization).

```python
best_wts = {k: v.cpu().clone() for k, v in model.state_dict().items()}
```

**Why `.cpu().clone()`?**  
- `.cpu()`: Moves the tensor from GPU to CPU for storage. Storing GPU tensors while
  continuing to train on GPU can exhaust GPU memory.
- `.clone()`: Creates a deep copy. Without `.clone()`, we'd store a reference to the
  original tensor, which continues to change during training — defeating the purpose
  of saving the "best" state.

---

## 8. Section F — Evaluation

```python
@torch.no_grad()
def evaluate_final(model, X_val, y_val):
    probs  = model(X_val.to(DEVICE)).cpu().numpy().flatten()
    preds  = (probs >= 0.5).astype(int)
    labels = y_val.cpu().numpy().flatten().astype(int)
```

**Why `.cpu().numpy()` instead of using PyTorch for metrics?**  
scikit-learn's `accuracy_score`, `classification_report`, and `confusion_matrix` expect
NumPy arrays, not PyTorch tensors. `.cpu()` moves data from GPU to CPU first (if on
CUDA); `.numpy()` converts to NumPy without copying memory (shared memory when on CPU).

**Why `.flatten()`?**  
Both `probs` and `labels` would be shape `(N, 1)` (column vectors). scikit-learn
metrics expect 1D arrays of shape `(N,)`. `.flatten()` collapses dimensions.

**Why threshold at 0.5?**  
The decision threshold of 0.5 is the Bayes-optimal threshold when both classes have
equal prior probability and equal misclassification costs. For imbalanced datasets or
asymmetric costs (e.g., medical diagnosis), a different threshold might be better.
(Covered in Topic 4 — Loss Functions.)

---

## 9. Section G — Visualization

### Decision Boundary Plot

```python
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 300),
                        np.linspace(x2_min, x2_max, 300))
grid = torch.tensor(np.c_[xx1.ravel(), xx2.ravel()], dtype=torch.float32)
```

**Why 300×300 mesh?**  
This creates 90,000 points covering the 2D feature space. We run inference on all of
them to get a predicted probability at each point, which we then color as a background.
300 gives smooth-looking boundaries; lower values (e.g., 50) would give blocky outlines.

**Why `np.c_[xx1.ravel(), xx2.ravel()]`?**  
`xx1.ravel()` flattens the 300×300 grid to a 90,000-element array of x₁ values.
`np.c_[...]` horizontally stacks two arrays, creating shape `(90000, 2)` —
one row per grid point with both feature values. This is the input format our model expects.

```python
Z = model(grid).cpu().numpy().reshape(xx1.shape)
ax.contourf(xx1, xx2, Z, alpha=0.35, cmap="RdBu", levels=50)
ax.contour(xx1, xx2, Z, levels=[0.5], colors="black", linewidths=2.0)
```

**Why two contour calls?**  
- `contourf` fills regions with color gradient (showing model confidence, not just class)
- `contour` at level 0.5 draws the **decision boundary** as a crisp black line

**Why `alpha=0.35` for the fill?**  
Partial transparency lets the data points (plotted on top) remain clearly visible
through the background color. Fully opaque would obscure the scatter plot.

---

## 10. Common Pitfalls and Why the Code Avoids Them

| Pitfall | Where It Would Occur | How the Code Avoids It |
|---------|---------------------|------------------------|
| Data leakage | StandardScaler | `fit_transform` on train only, `transform` on val |
| Gradient accumulation | Training loop | `optimizer.zero_grad()` before each forward pass |
| Wrong label shape | BCELoss | `.unsqueeze(1)` → `(N,1)` to match model output |
| GPU→CPU memory leak | Best weight saving | `.cpu().clone()` when saving best state dict |
| log(0) = -inf | Manual BCE | `eps = 1e-9` added inside log |
| Class imbalance in split | train_test_split | `stratify=y` parameter |
| Mode mismatch (train/eval) | Inference | `model.eval()` + `@torch.no_grad()` always paired |
| Zero-init in MLP | MLPScratch | Xavier initialization scales weights by `sqrt(2/fan_in)` |
| Tensor not on same device | Loss computation | `X_batch.to(device)` before every model call |
| Late matplotlib import order | Backend selection | `matplotlib.use("Agg")` before `import pyplot` |

---

