# Theory: Regularization, Optimizers, Batch Normalization & Early Stopping

**Deep Learning Mastery Repository**

---

## Table of Contents
1. [Regularization Overview](#1-regularization-overview)
2. [L1 and L2 Regularization](#2-l1-and-l2-regularization)
3. [Dropout](#3-dropout)
4. [Batch Normalization](#4-batch-normalization)
5. [Optimizers](#5-optimizers)
6. [Early Stopping](#6-early-stopping)

---

## 1. Regularization Overview

Regularization refers to any technique that reduces overfitting by adding a penalty
or constraint on the model's complexity beyond fitting the training data.

**The regularised objective:**
```
L_reg(θ) = L_data(θ) + λ · R(θ)

where:
  L_data = empirical risk (e.g. BCE, MSE)
  R(θ)   = regularisation term (penalises complexity)
  λ       = regularisation strength (hyperparameter)
```

The fundamental tension: reducing `L_data` wants more complex weights (overfit);
`R(θ)` pushes weights toward zero/simplicity (generalise).

---

## 2. L1 and L2 Regularization

### 2.1 L2 Regularization (Weight Decay / Ridge)

```
R(θ) = Σᵢ wᵢ²  =  ‖w‖²₂

L_reg = L_data + (λ/2)‖w‖²₂

Gradient:  ∂L_reg/∂w = ∂L_data/∂w + λw

Update rule (L2 GD):
  w ← w − η(∂L_data/∂w + λw)
    = w(1 − ηλ) − η·∂L_data/∂w    ← shrinks w by factor (1−ηλ) each step
```

**Why (1−ηλ) factor = "weight decay":**
Every gradient step multiplies weights by `(1−ηλ)` before adding the gradient.
This exponentially decays weights toward zero over time unless the data gradient
actively prevents it. Hence the name "weight decay."

**Effect:** L2 prefers **small but non-zero** weights. All weights shrink but
rarely reach exactly zero.

**Geometric interpretation:**
L2 constrains parameters to lie within an L2-ball `‖w‖₂ ≤ C`.
The optimal regularised solution is found at the boundary of this sphere.

### 2.2 L1 Regularization (Lasso)

```
R(θ) = Σᵢ |wᵢ|  =  ‖w‖₁

Gradient:  ∂R/∂wᵢ = sign(wᵢ)   (subgradient at wᵢ=0)

Update rule (L1 subgradient):
  w ← w − η(∂L_data/∂w + λ·sign(w))
```

**Effect:** L1 pushes weights toward **exactly zero** → **sparse** models.
Many weights become exactly zero, performing implicit feature selection.

**Geometric interpretation:**
L1 constrains to an L1-ball (diamond shape in 2D). Corners of the diamond lie
on the coordinate axes — if the loss function's contours touch the diamond at
a corner, the solution is sparse (one weight = 0).

### 2.3 L1 vs L2 Comparison

```
Property              L1 (Lasso)              L2 (Ridge)
──────────────────────────────────────────────────────────
Sparsity              YES — exact zeros        No (small non-zero)
Feature selection     Implicit                 No
Scale invariance      No                       Yes
Differentiable        No (at 0)                Yes everywhere
Unique solution       Not always               Always (convex + strictly conv.)
Best for              Many irrelevant features Few irrelevant features
```

### 2.4 Elastic Net

Combines both:
```
R(θ) = α‖w‖₁ + (1−α)‖w‖²₂   where α ∈ [0,1]
```
Gets sparsity of L1 with stability of L2.

---

## 3. Dropout

### 3.1 Mechanism

During training, each neuron is independently **zeroed out** with probability `p`:

```
For each forward pass:
  mask_j ~ Bernoulli(1-p)      j = 1, ..., n_hidden
  ã_j    = mask_j · a_j / (1-p)   ← scale by 1/(1-p) to maintain expected value
```

**Why scale by `1/(1-p)` during training?**
Without scaling, the expected value of a unit's output changes between training
(`E[ã] = (1-p)a`) and inference (`E[a] = a`). Scaling by `1/(1-p)` makes
`E[ã] = a`, keeping magnitudes consistent. This is called **inverted dropout**.

### 3.2 Training vs Inference

```
Training:   dropout mask is sampled fresh every forward pass (stochastic)
Inference:  dropout is DISABLED — use full network with no masking
            This is why model.eval() is essential before inference
```

### 3.3 Why Dropout Works

**Ensemble interpretation:**
With n neurons and dropout rate p, each forward pass samples a different
sub-network from the `2^n` possible ones. Training with dropout ≈ averaging
over exponentially many sub-networks at inference.

**Co-adaptation prevention:**
Without dropout, neurons can co-adapt: neuron A learns to fix the errors of
neuron B. This creates fragile representations. Dropout forces each neuron
to be independently useful.

**Noise injection:**
Dropout adds multiplicative Bernoulli noise, acting as a regulariser.
This is related to Bayesian approximate inference (Gal & Ghahramani, 2016).

### 3.4 When and Where to Use

```
Placement:
  ✓ After hidden layer activations
  ✓ Before fully-connected layers (common in VGG, etc.)
  ✗ Not after batch normalisation layers (they conflict)
  ✗ Not in convolutional layers (use spatial dropout instead)
  ✗ Not in output layer

Dropout rates:
  p=0.5   → default for fully-connected layers (Srivastava et al.)
  p=0.1–0.3 → for input layer or convolutional features
  p=0.0   → for small datasets (dropout may hurt more than help)
```

---

## 4. Batch Normalization

### 4.1 The Problem It Solves: Internal Covariate Shift

As weights update during training, the distribution of inputs to each layer
changes — a phenomenon called **internal covariate shift**. This forces
each layer to continuously adapt to a shifting input distribution, slowing
convergence and requiring careful initialisation.

### 4.2 The Algorithm (Ioffe & Szegedy, 2015)

For a mini-batch B = {x₁, ..., xₘ}:

```
Step 1 — Compute batch statistics:
  μB = (1/m) Σᵢ xᵢ            (batch mean)
  σ²B = (1/m) Σᵢ (xᵢ − μB)²   (batch variance)

Step 2 — Normalise:
  x̂ᵢ = (xᵢ − μB) / √(σ²B + ε)  (zero mean, unit variance, ε=1e-5)

Step 3 — Scale and shift (learnable):
  yᵢ = γ x̂ᵢ + β               γ, β are learned parameters
```

**Why learnable γ (scale) and β (shift)?**
Pure normalisation would force `x̂ᵢ` to have mean=0, std=1 always. But the
optimal activation statistics for a particular layer may not be (0, 1). The
learnable `γ` and `β` allow the network to learn the optimal scale and shift,
including recovering the identity transform `γ=σ, β=μ` if needed.

### 4.3 Training vs Inference

```
Training:  use batch statistics μB, σ²B (computed per mini-batch)
Inference: use RUNNING statistics μ_run, σ²_run (exponential moving average
           accumulated during training)

Running stats update (each training step):
  μ_run ← momentum · μ_run + (1−momentum) · μB
  σ²_run ← momentum · σ²_run + (1−momentum) · σ²B

This is why model.train() and model.eval() matter for BatchNorm!
```

### 4.4 Effects and Benefits

```
1. Allows higher learning rates (gradients are better conditioned)
2. Reduces sensitivity to weight initialisation
3. Acts as mild regulariser (batch statistics add noise similar to dropout)
4. Significantly speeds up training (2-14× empirically)
5. Reduces need for Dropout in some architectures
```

### 4.5 Placement

```
Standard:   Linear → BatchNorm → Activation  (original paper)
Alternative: Linear → Activation → BatchNorm  (some argue works better)

In ResNets: Pre-activation: BN → ReLU → Conv (He et al., 2016)
```

### 4.6 Limitations

```
✗ Performance degrades with small batch sizes (< 8–16)
  → Alternative: Layer Normalization (used in Transformers)
✗ Not suitable for online learning (batch stats meaningless for batch=1)
✗ Complications in RNNs (variable sequence lengths)
  → Alternative: Layer Norm for RNNs and Transformers
```

---

## 5. Optimizers

### 5.1 Vanilla SGD

```
g_t    = ∇θ L(θ_t)
θ_{t+1} = θ_t − η · g_t
```

Pros: Simple, memory efficient, theoretically well-understood.
Cons: Same LR for all parameters; sensitive to LR choice; no momentum.

### 5.2 SGD with Momentum

```
v_t    = β·v_{t-1} + g_t               β = momentum coeff. (usually 0.9)
θ_{t+1} = θ_t − η·v_t
```

**Intuition:** Accumulates an exponentially decaying moving average of gradients.
In flat directions (gradients consistent), velocity builds up → large steps.
In noisy/oscillating directions, gradients cancel → small effective steps.

**Nesterov Momentum (NAG):**
```
θ_look = θ_t − β·v_t                   (look-ahead position)
g_look  = ∇θ L(θ_look)                  (gradient at look-ahead)
v_t     = β·v_{t-1} + g_look
θ_{t+1} = θ_t − η·v_t
```
NAG corrects the direction of momentum by computing the gradient at the
look-ahead position, leading to better convergence.

### 5.3 AdaGrad

```
G_t    = Σᵢ₌₁ᵗ gᵢ²          (cumulative sum of squared gradients, per parameter)
θ_{t+1} = θ_t − (η / √(G_t + ε)) · g_t
```

**Effect:** Parameters with large historical gradients (frequent features) get
small effective LR; sparse/rare features get large LR — adaptive per-parameter rates.
**Problem:** G_t grows monotonically → LR shrinks to 0 over time → training stalls.

### 5.4 RMSprop

```
v_t    = β·v_{t-1} + (1−β)·g_t²       (exponential moving average of g²)
θ_{t+1} = θ_t − (η / √(v_t + ε)) · g_t

Typical: β=0.9, η=1e-3, ε=1e-8
```

**Fix for AdaGrad:** Uses EMA instead of cumulative sum → LR doesn't shrink to 0.
Widely used in RNNs. Proposed by Hinton in his Coursera lecture (unpublished!).

### 5.5 Adam (Adaptive Moment Estimation)

```
m_t = β₁·m_{t-1} + (1−β₁)·g_t         (1st moment — mean of gradients)
v_t = β₂·v_{t-1} + (1−β₂)·g_t²        (2nd moment — uncentred variance)

Bias correction (important at early steps when m,v initialised at 0):
  m̂_t = m_t / (1 − β₁ᵗ)
  v̂_t = v_t / (1 − β₂ᵗ)

Update:
  θ_{t+1} = θ_t − η · m̂_t / (√v̂_t + ε)

Typical defaults: β₁=0.9, β₂=0.999, ε=1e-8, η=1e-3
```

**Why bias correction?**
At t=1: `m_1 = (1−β₁)·g_1`. With β₁=0.9, `m_1 = 0.1·g_1` — severely
underestimates the true gradient mean. Dividing by `(1−β₁ᵗ) = 0.1` corrects
this to `m̂_1 = g_1`. As t grows, `β₁ᵗ → 0` and correction factor → 1.

**Adam = Momentum + RMSprop:**
- `m_t` captures gradient direction and momentum (like SGD+momentum)
- `v_t` captures gradient magnitude per parameter (like RMSprop)
- Division `m̂/√v̂` = normalised step in gradient direction ≈ unit gradient

**Adam Variants:**
```
AdamW:   Decouples weight decay from gradient update (Loshchilov & Hutter, 2019)
         θ_{t+1} = θ_t(1−ηλ) − η·m̂_t/√(v̂_t+ε)   ← L2 applied directly to weights
         Better than Adam+L2 regularisation in transformers.

AMSGrad: Uses max of past v̂_t to ensure non-increasing LR (convergence guarantee).

Adadelta: Similar to RMSprop but without manual LR specification.
```

### 5.6 Optimizer Comparison

```
                  Momentum   Adaptive LR   Memory        Typical Use
────────────────────────────────────────────────────────────────────────
SGD               No         No            O(1)          Convex problems
SGD+Momentum      Yes        No            O(d)          CNNs (fine-tuned)
AdaGrad           No         Yes           O(d)          Sparse features
RMSprop           Yes        Yes           O(d)          RNNs
Adam              Yes        Yes           O(2d)         Default for DNNs
AdamW             Yes        Yes           O(2d)         Transformers
```

---

## 6. Early Stopping

### 6.1 Algorithm

```
best_val_loss = ∞
patience_counter = 0

For each epoch:
  train_epoch()
  val_loss = evaluate()

  if val_loss < best_val_loss − min_delta:
    best_val_loss    = val_loss
    best_model_state = copy(model.state_dict())
    patience_counter = 0
  else:
    patience_counter += 1

  if patience_counter >= patience:
    restore(best_model_state)
    STOP training

Typical: patience=10–20 epochs, min_delta=1e-4
```

### 6.2 Why Early Stopping Works as Regularisation

**Theorem (Bousquet & Bottou, 2008):**
For SGD on a quadratic loss, early stopping is equivalent to L2 regularisation
with `λ ≈ 1/(η·T)` where T = number of steps.

**Intuition:** The model starts from random initialisation and gradually
moves through parameter space. Early stopping halts it before it can overfit
— the number of steps T plays the role of the regularisation budget.

### 6.3 Key Hyperparameters

```
patience    : how many epochs of non-improvement before stopping (10–50)
min_delta   : minimum improvement to count as "improved" (1e-4 to 1e-3)
monitor     : 'val_loss' (most common) or 'val_accuracy'
restore_best: should we restore the best weights? (YES — always)
```

---

## Key Equations Summary

| Technique | Key Formula | Effect |
|---|---|---|
| L2 regularisation | L + λ‖w‖² → w(1−ηλ) | Small weights, no sparsity |
| L1 regularisation | L + λ‖w‖₁ → w−ηλ sign(w) | Sparse weights (feature selection) |
| Dropout | ã = mask·a/(1−p) | Ensemble of sub-networks |
| Batch Norm | ŷ = γ(x−μ)/σ + β | Normalised activations |
| Adam update | θ ← θ − η·m̂/√(v̂+ε) | Adaptive per-parameter LR |
| Early stopping | Stop at T* = argmin val_loss | Implicit regularisation |
