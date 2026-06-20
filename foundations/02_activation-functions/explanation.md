# Code Explanation: Activation Functions

**Topic: `implementation.py` walkthrough**

Every non-obvious line is explained below — the *why* behind numerical tricks,
design choices, tensor operations, and experiment setup.

---

## Table of Contents

1. [Section A — NumPy Implementations](#1-section-a--numpy-implementations)
2. [Section B — PyTorch Factory Pattern](#2-section-b--pytorch-factory-pattern)
3. [Section C — Gradient Flow Experiment](#3-section-c--gradient-flow-experiment)
4. [Section D — Training Comparison](#4-section-d--training-comparison)
5. [Section E — Dead Neuron Analysis (Hooks)](#5-section-e--dead-neuron-analysis-hooks)
6. [Section F — Softmax Numerical Stability](#6-section-f--softmax-numerical-stability)
7. [Section G — Visualization](#7-section-g--visualization)
8. [Live Results Interpretation](#8-live-results-interpretation)

---

## 1. Section A — NumPy Implementations

### Base Class Design

```python
class ActivationFn:
    name:  str = "base"
    color: str = "#000000"

    def forward(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def derivative(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError
```

**Why a class per activation (not plain functions)?**
Each activation carries three pieces of data together: its computation (`forward`),
its derivative (`derivative`), its display name, and its plot color. Grouping them
in a class keeps related things together and allows us to iterate over
`ALL_NUMPY_ACTS` polymorphically in the visualization loop without
`if/elif` chains.

**Why `name` and `color` as class attributes (not instance)?**
They are the same for every instance of `SigmoidFn`. Class attributes avoid
allocating a new string per object. The visualization loop reads `fn.color`
without caring whether it's a class or instance attribute — Python resolves either.

---

### Stable Sigmoid

```python
def forward(self, z):
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-np.abs(z))),
        np.exp(z) / (1.0 + np.exp(z))
    )
```

**Why two branches instead of the naive `1/(1+exp(-z))`?**

For large negative `z` (e.g., z = -1000):
```
Naïve:  exp(-(-1000)) = exp(1000) → inf
        1 / (1 + inf) = 0.0       ← actually correct, but...

In float32: exp(89) = inf, so even z=-90 causes overflow for the intermediate
computation, and inf/inf → NaN if it appears in a denominator elsewhere.
```

The stable trick uses the identity:
```
σ(z) = 1/(1+e^{-z}) = e^z/(1+e^z)
```

- For `z ≥ 0`: use `1/(1+exp(-z))` — argument of exp is ≤ 0 (never overflows)
- For `z < 0`: use `exp(z)/(1+exp(z))` — argument of exp is < 0 (never overflows)

**Why `np.abs(z)` in the positive branch?**
`np.where` evaluates **both** branches for every element before selecting.
Without `np.abs(z)`, elements where `z < 0` would still compute `exp(-z)` in
the positive branch, causing potential overflow. Using `np.abs(z)` ensures
the exponent is always non-positive in the first branch.

---

### ELU and `np.expm1`

```python
return np.where(z > 0, z, self.alpha * np.expm1(z))
```

**Why `np.expm1(z)` instead of `np.exp(z) - 1`?**

For very small `z` (e.g., z = 1e-8):
```
np.exp(1e-8) - 1  →  (1 + 1e-8 + ...) - 1  →  1e-8  + rounding errors
np.expm1(1e-8)    →  1.0000000050000001e-08  (exact Taylor series for small z)
```

`expm1` is specifically designed to compute `e^z - 1` accurately for small z,
avoiding catastrophic cancellation. The ELU derivative uses this:
```python
def derivative(self, z):
    return np.where(z > 0, 1.0, self.forward(z) + self.alpha)
```

The identity `ELU'(z) = ELU(z) + α` for `z ≤ 0` avoids recomputing `exp(z)`:
```
ELU(z) = α(e^z - 1)
ELU'(z) = αe^z = α(e^z - 1) + α = ELU(z) + α
```
This is a simple algebraic trick that reuses already-computed values.

---

### SELU Constants

```python
LAMBDA = 1.0507009873554804934193349852946
ALPHA  = 1.6732632423543772848170429916717
```

**Why these specific values?**
Klambauer et al. (2017) derived them by solving a system of two equations that
enforce the **self-normalizing** fixed-point conditions:

```
E[SELU(z)]   = 0    when z ~ N(0, 1)
Var[SELU(z)] = 1    when z ~ N(0, 1)
```

These are integrals over the Gaussian distribution, solved numerically to
high precision. Any other (λ, α) pair would break the normalization property.
The 34-digit precision isn't necessary in float64, but it's the convention
in all implementations to copy these constants verbatim.

---

### GELU Approximation

```python
_K = np.sqrt(2.0 / np.pi)   # constant precomputed once at class level

def forward(self, z):
    return 0.5 * z * (1.0 + np.tanh(self._K * (z + 0.044715 * z ** 3)))
```

**Why approximate instead of exact?**
The exact GELU uses `erf(z/√2)`, which is slower than `tanh` on most hardware
and was slower on the GPU hardware available in 2016 when BERT was developed.
The tanh approximation matches the exact to within ~0.0002 for all z.

**Why precompute `_K` at the class level?**
`np.sqrt(2.0/np.pi)` is computed once at class definition time, not inside
every `forward()` call. Over millions of training steps, this saves a tiny but
real amount of time.

**Why `z ** 3` instead of `pow(z, 3)` or `np.power(z, 3)`?**
The `**` operator for NumPy arrays calls `np.power` internally and is the
idiomatic, readable form. All three are equivalent in performance.

---

### Mish Overflow Protection

```python
sp = np.log1p(np.exp(np.clip(z, -500, 20)))
```

**Why `np.clip(z, -500, 20)`?**
- `np.exp(20) ≈ 4.8 × 10^8` — large but finite in float64
- `np.exp(21) ≈ 1.3 × 10^9` — still fine
- `np.exp(710) → inf` in float64 (overflow)

Clipping to max 20 keeps `exp(z)` well within float64 range for the
visualization domain (z ∈ [-4, 4] in our plots). The lower clip (-500) prevents
`exp(very_negative)` from underflowing to 0 (which would make `log1p(0) = 0`,
slightly inaccurate but not disastrous).

**Why `np.log1p(x)` instead of `np.log(1 + x)`?**
Same reason as `expm1`: for tiny `x`, `log(1 + x)` has catastrophic cancellation.
`log1p` uses a Taylor series internally for accurate results near x=0.

---

## 2. Section B — PyTorch Factory Pattern

```python
TORCH_ACT_FACTORY = {
    "Sigmoid":    lambda: nn.Sigmoid(),
    "Tanh":       lambda: nn.Tanh(),
    "ReLU":       lambda: nn.ReLU(),
    ...
    "Mish":       lambda: MishModule(),
}
```

**Why lambdas instead of module instances?**

```python
# WRONG — shared instance:
BAD = {"ReLU": nn.ReLU()}
model_A = nn.Sequential(nn.Linear(2,64), BAD["ReLU"])
model_B = nn.Sequential(nn.Linear(2,64), BAD["ReLU"])
# model_A and model_B share the SAME nn.ReLU object!
# If either is deleted or modified, both break.

# RIGHT — factory:
GOOD = {"ReLU": lambda: nn.ReLU()}
model_A = nn.Sequential(nn.Linear(2,64), GOOD["ReLU"]())  # fresh instance
model_B = nn.Sequential(nn.Linear(2,64), GOOD["ReLU"]())  # different instance
```

`nn.Module` objects have state (registered parameters, hooks, training/eval
mode flags). Sharing one instance between two models means calling
`model_A.train()` also sets `model_B` to train mode — a subtle, hard-to-debug bug.

**Why not just call `nn.ReLU()` directly each time?**
The factory pattern decouples the *name* of an activation from the *construction*
of its module. Our training loop can iterate over
`for name, factory in TORCH_ACT_FACTORY.items()` without knowing in advance
which activations exist — adding a new one just requires one dictionary entry.

**Why a custom `MishModule` instead of `nn.Mish()`?**
PyTorch added `nn.Mish` in version 1.9. The custom implementation works
on all versions ≥ 1.7 and makes the formula explicit for educational value:
```python
class MishModule(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
        # F.softplus(x) = log(1 + e^x), numerically stable in PyTorch
```

---

## 3. Section C — Gradient Flow Experiment

### Weight Initialization Strategy

```python
init_std = 1.0 / np.sqrt(n_hidden)
...
nn.init.normal_(m.weight, mean=0.0, std=init_std)
```

**Why this specific `init_std`?**

With `n_hidden=64` hidden units and input `x ~ N(0,1)`, if we initialize
`W ~ N(0, 1/n_hidden)`, then the pre-activation:
```
z = Wx + b,   where W ~ N(0, 1/64),  x ~ N(0,1)
Var[z] = n_hidden · (1/64)  · Var[x] = 1 · 1 = 1
```
So `z ~ N(0, 1)` at initialization — the activations are not saturated yet.

For sigmoid, `σ'(z)` at z ~ N(0,1) averages around 0.2. After 15 layers:
`(0.2)^15 ≈ 3 × 10^{-11}` — dramatic vanishing gradient, visible in the log plot.

For ReLU, `ReLU'(z) = 1` for positive z (50% of neurons). There is no
multiplicative decay — gradient signal flows backward unattenuated.

### Why `reversed(norms)` in visualization

```python
y_vals = list(reversed(norms))  # input layer on left
```

When we iterate `model.modules()` and collect Linear layer gradients, we get
them in **forward-pass order**: output layer first (index 0 in the list),
input layer last. In the gradient flow plot, convention is to show the input
layer on the left (shallower side), so we reverse.

### Why `model.modules()` Not `model.children()`

```python
for m in model.modules():
    if isinstance(m, nn.Linear) and m.weight.grad is not None:
```

`model.children()` returns only **direct** sub-modules (the immediate children
in the module tree). `model.modules()` returns all sub-modules recursively,
including nested ones. Since `nn.Sequential` wraps all layers, `children()` would
return the `Sequential` itself (not the individual `Linear` layers inside it).
`modules()` descends into the `Sequential` and finds each `Linear`.

---

## 4. Section D — Training Comparison

### Activation-Specific Initialization

```python
def _init_model(model, act_name):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if act_name == "SELU":
                nn.init.kaiming_normal_(m.weight, mode="fan_in",
                                        nonlinearity="linear")
            else:
                nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
```

**Why Kaiming/LeCun normal for SELU and not Xavier?**

SELU's self-normalization guarantee requires that the input to each SELU layer
has variance 1. Klambauer et al. proved this holds if and only if weights are
initialized with LeCun normal:
```
W ~ N(0, 1/fan_in)
```

PyTorch's `kaiming_normal_` with `mode="fan_in"` and `nonlinearity="linear"` gives:
```
std = sqrt(1/fan_in)   ← exactly LeCun normal
```

Using `xavier_uniform_` with SELU would break the self-normalization property.
The model would still train, but wouldn't have the normalization benefit
and could behave erratically on deeper networks.

### Why Only 100 Epochs for the Comparison

With Adam (lr=1e-3) and 800 training samples, 100 epochs gives each model
enough time to converge while keeping the runtime manageable. The key insight
from the live results was visible within 50 epochs — all ReLU-family activations
converged well above 99%, while vanilla Sigmoid struggled at 75% due to
vanishing gradients through the 2-hidden-layer network.

### Tracking Loss Correctly Across Variable-Size Batches

```python
tl += loss.item() * len(Xb)   # un-normalized loss × batch size
...
history["train_loss"].append(tl / tn)   # divide by total samples
```

**Why not just `tl += loss.item()`?**

`nn.BCELoss()` returns the *mean* loss over the batch. The last batch of an
epoch may be smaller (e.g., 800 % 32 = 0, but in general it won't divide evenly).
If we average means directly:
```
epoch_loss = (mean_batch_1 + mean_batch_2 + ... + mean_last_batch) / n_batches
```
The smaller last batch gets weighted equally to full batches — incorrect.
The correct approach: accumulate total loss (mean × size), then divide by
total samples.

---

## 5. Section E — Dead Neuron Analysis (Hooks)

### Why PyTorch Forward Hooks

```python
def make_hook(store: list):
    def hook(module, inp, out):
        store.append(out.detach().cpu())
    return hook
```

**What is a forward hook?**
A hook is a callback function registered on an `nn.Module` that PyTorch calls
**automatically** after every `forward()` call of that module. It receives:
- `module`: the module itself
- `inp`: tuple of inputs
- `out`: the module's output tensor

This lets us inspect internal activations without modifying the model's
`forward()` method — essential for analysis of pre-built or third-party models.

**Why `make_hook(store)` (closure) instead of a global list?**

```python
# WRONG — all hooks append to the same global list, mixing layers:
outputs = []
def hook(module, inp, out):
    outputs.append(out.detach().cpu())

# RIGHT — each layer has its own bucket via closure:
for layer in activation_layers:
    bucket = []
    layer_outputs.append(bucket)
    hooks.append(layer.register_forward_hook(make_hook(bucket)))
```

`make_hook(store)` creates a **closure** — each call to `make_hook` captures
its own `store` reference. When the hook fires, it appends to that specific layer's
bucket, not a shared global. Without this, we couldn't separate layer 1's
activations from layer 2's.

**Why `out.detach().cpu()`?**

- `.detach()`: Removes the tensor from the computation graph. Without this,
  the hook stores a tensor with gradient history attached — keeping the entire
  computation graph in memory, causing an OOM on deep networks.
- `.cpu()`: Moves from GPU to CPU for storage. Keeping many intermediate
  activation tensors on GPU would exhaust VRAM.

### Registering and Removing Hooks

```python
hooks.append(layer.register_forward_hook(make_hook(bucket)))
...
for h in hooks:
    h.remove()
```

**Why must we remove hooks after use?**

`register_forward_hook` returns a `RemovableHook` object. The hook remains
registered on the module permanently until explicitly removed. If we forget
to call `h.remove()`:
1. Every subsequent forward pass (even during normal training) would append
   to the bucket — wasting memory
2. If the model is evaluated repeatedly, the buckets would grow without bound
3. Hooks are garbage-collected only when the module itself is deleted

This is a common memory leak pattern in PyTorch debugging code.

### Dead Neuron Detection Logic

```python
combined  = torch.cat(bucket, dim=0)   # (N_total, n_hidden)
dead_mask = (combined == 0.0).all(dim=0)
n_dead    = dead_mask.sum().item()
```

**Shape trace:**
```
Each hook call appends: (batch_size, n_hidden)  e.g. (32, 128)
After torch.cat along dim=0:  (N_total, 128)    e.g. (800, 128)
(combined == 0.0):            (800, 128) — True where output is exactly 0
.all(dim=0):                  (128,)     — True for neuron j if ALL 800 samples gave 0
.sum():                       scalar     — count of dead neurons
```

**Why `== 0.0` exactly for ReLU?**
ReLU output for `z ≤ 0` is **exactly** 0.0 (integer clamp, no floating-point
approximation). Leaky ReLU output for `z < 0` is `0.01 * z`, which is a small
negative float — never exactly 0 (unless z=0 exactly, which almost never
happens with random data).

This means `== 0.0` correctly captures:
- ReLU dead neurons: ✓ (always output 0)
- Leaky ReLU always-negative neurons: ✗ (output α·z ≠ 0)

Exactly the semantics we want.

---

## 6. Section F — Softmax Numerical Stability

### The Overflow Chain

```python
exp_naive = np.exp(z)        # z = [1000, 1001, 1002] → [inf, inf, inf]
naive     = exp_naive / exp_naive.sum()   # inf/inf → NaN
```

**Why does `inf/inf = NaN`?**
IEEE 754 floating-point standard defines:
```
inf / inf = NaN  (indeterminate form)
inf + finite = inf
inf - inf = NaN
```
The softmax denominator `inf + inf + inf = inf`. Then `inf/inf = NaN`.

**Why the stable trick works:**

```python
z_shift   = z - z.max()           # [-2, -1, 0] — max is 0
exp_stable = np.exp(z_shift)       # [0.135, 0.368, 1.0] — all finite!
stable     = exp_stable / exp_stable.sum()
```

Mathematically: subtracting `max(z)` from all logits doesn't change the
softmax output (the constant cancels in numerator and denominator), but it
guarantees the maximum exponent argument is 0, so `exp(0) = 1` — no overflow.

### PyTorch's `log_softmax` vs Manual `log(softmax(x))`

```python
wrong = torch.log(torch.softmax(logits, dim=0))
right = F.log_softmax(logits, dim=0)
```

**Why do both give the same numerical answer here but `wrong` is still wrong?**

In this example `logits = [1000, 1001, 1002]`, PyTorch's `torch.softmax`
uses the stable trick internally, so the intermediate result is valid.
`torch.log` of valid probabilities gives correct log-probabilities.

However, if `logits` were moderate values, the numerics of the two approaches
diverge because:
```
log(softmax(z)_k) = log(e^{z_k} / Σⱼ e^{z_j})
                  = z_k - log(Σⱼ e^{z_j})
```

`log_softmax` computes this in one fused operation using LogSumExp:
```
log_softmax(z)_k = z_k - log(Σⱼ exp(z_j - max(z))) - max(z)
```

This avoids the intermediate step of computing softmax probabilities (which
round to 0 for large negative logits) and then taking their log (which would
give `-inf`). The fused operation retains more precision.

---

## 7. Section G — Visualization

### Why Three Separate Figures

```
Figure 1 (02_activation_overview.png):
  - Function shapes + derivatives (requires all numpy activations)
  - Gradient flow experiment (requires grad_results)
  - Dead neuron bar chart (requires dead_results)

Figure 2 (02_training_comparison.png):
  - Train/val loss and accuracy per activation

Figure 3 (02_decision_boundaries.png):
  - 3×3 grid of decision boundaries
```

All three depend on different computation results and have different optimal
aspect ratios. Separate figures allow saving at the right size and presenting
independently without one figure becoming unreadably crowded.

### Gradient Flow: Log Scale

```python
ax.semilogy(x_vals, y_vals, ...)
```

**Why `semilogy` (log y-axis)?**
The gradient norms for sigmoid span 12 orders of magnitude across 15 layers
(from ~`2.6e-01` at the output to `2.8e-10` at the input). On a linear scale,
the input-layer gradient would appear as exactly zero — invisible. A log scale
makes the exponential decay clearly visible as a steep straight line, while
ReLU appears relatively flat.

### Decision Boundary Grid

```python
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 250),
                        np.linspace(x2_min, x2_max, 250))
grid = torch.tensor(np.c_[xx1.ravel(), xx2.ravel()], dtype=torch.float32)
```

**Why build the mesh grid once outside the loop?**
All 9 models are evaluated on the same 2D space. Building `xx1, xx2` once (250×250
= 62,500 points) and reusing it avoids 8 redundant grid constructions.

**Shape trace:**
```
xx1, xx2:    each (250, 250) — one grid value per point
xx1.ravel(): (62500,) — flattened to 1D
np.c_[...]:  (62500, 2) — stacked to 2-column feature matrix
grid:        torch tensor (62500, 2)
model(grid): (62500, 1) — predicted probability at each point
.reshape(xx1.shape): (250, 250) — back to grid shape for contourf
```

**Why `contourf` AND `contour`?**

```python
ax.contourf(xx1, xx2, Z, alpha=0.3, cmap="RdBu", levels=50)  # filled colors
ax.contour(xx1, xx2, Z, levels=[0.5], colors="black", linewidths=1.8)  # boundary line
```

- `contourf`: fills the background with color gradient showing model confidence
  (red → class 0, blue → class 1). `levels=50` creates a smooth gradient.
- `contour` at `levels=[0.5]`: draws exactly one contour — the decision
  boundary where P(class=1) = 0.5. The black line makes the boundary crisp
  and visible against the color fill.

---

## 8. Live Results Interpretation

### Gradient Flow Results (Section C)

```
Sigmoid    | input-layer norm: 2.82e-10 | output: 2.62e-01 | ratio: 1.1e-09
Tanh       | input-layer norm: 1.78e-01 | output: 9.63e-02 | ratio: 1.8e+00
ReLU       | input-layer norm: 3.78e-03 | output: 3.16e-03 | ratio: 1.2e+00
ELU        | input-layer norm: 1.42e-01 | output: 6.48e-02 | ratio: 2.2e+00
```

- **Sigmoid**: gradient at input is 10^9× smaller than at output — extreme vanishing.
  Training layer 1 while keeping layer 15 fixed is essentially impossible.
- **Tanh**: slight vanishing (ratio 1.8), far better than sigmoid. Gradient decreases
  but not catastrophically over 15 layers.
- **ReLU**: near-flat (ratio 1.2). Gradient is roughly the same at every layer —
  the ideal behaviour.
- **ELU**: even better than Tanh (ratio 2.2), smooth and close to ReLU.

### Training Accuracy (Section D)

```
Sigmoid:    75.00% ← Struggling — vanishing gradient limits learning
Tanh:       99.50% ← Excellent
ReLU:       99.00% ← Excellent
SiLU/Swish: 99.50% ← Best final loss (0.01553)
GELU:       99.50% ← Best alternatives on NLP tasks
```

**Why does Sigmoid underperform so dramatically?**
The network has 2 hidden layers. With sigmoid, gradients at layer 1 are already
`(0.25)^2 = 0.0625` of the output gradient — 16× smaller. The model barely
updates layer 1 weights, effectively learning only from the output layer.

### Dead Neurons (Section E)

```
ReLU (lr=0.5 — high):    3/256 dead (1.2%)  ← Some dead
ReLU (lr=0.001 — proper): 9/256 dead (3.5%)  ← Slightly more!
Leaky ReLU (lr=0.5):      0/256 dead (0.0%)  ← Zero dead
```

**Surprising: why does proper LR have MORE dead neurons than high LR?**
With `lr=0.001` and SGD (not Adam), the model trains slowly. After only 30 epochs,
some neurons may have negative pre-activations that haven't had enough gradient
signal yet to recover — they are "temporarily dead" rather than permanently dead.
With `lr=0.5`, larger updates can rescue some borderline neurons. This shows that
"dead neurons" in practice is a more nuanced phenomenon than the theoretical
definition — a short experiment with SGD is not fully representative of
Adam-trained models in production.

---

## Quick Reference: Pitfalls This Code Avoids

| Pitfall | Location | Fix Applied |
|---|---|---|
| Shared activation instances | TORCH_ACT_FACTORY | Lambda factory pattern |
| `exp(-z)` overflow in sigmoid | `SigmoidFn.forward` | Two-branch stable implementation |
| `exp(z) - 1` precision loss | `ELUFn.forward` | `np.expm1(z)` |
| Log softmax via log(softmax) | Section F demo | `F.log_softmax` |
| Hook memory leak | Section E | `h.remove()` after use |
| Gradient tracking in hooks | Section E | `.detach().cpu()` on stored tensors |
| Wrong epoch-average loss | Section D | Accumulate `loss × batch_size`, divide by `total_n` |
| Mesh grid rebuilt per model | Section G (Fig 3) | Built once outside loop |
| Linear scale hides 10^9 range | Section G (Fig 1) | `ax.semilogy()` |
| SELU with wrong init | `_init_model` | `kaiming_normal_` with `fan_in` |

---

*Previous: [Topic 1 — Perceptron & MLP](../01-perceptron-and-mlp/explanation.md)*
*Next: [Topic 3 — Gradient Descent & Backpropagation](../03-gradient-descent-and-backprop/explanation.md)*
