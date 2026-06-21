# Theory: Gradient Descent & Backpropagation

**Deep Learning Mastery Repository**

---

## Table of Contents

1. [The Optimization Problem](#1-the-optimization-problem)
2. [Gradient Descent](#2-gradient-descent)
3. [Gradient Descent Variants](#3-gradient-descent-variants)
4. [The Backpropagation Algorithm](#4-the-backpropagation-algorithm)
5. [Chain Rule — The Mathematical Foundation](#5-chain-rule--the-mathematical-foundation)
6. [Full Backprop Derivation for a 2-Layer MLP](#6-full-backprop-derivation-for-a-2-layer-mlp)
7. [Computational Graphs and PyTorch Autograd](#7-computational-graphs-and-pytorch-autograd)
8. [Learning Rate: Analysis and Convergence](#8-learning-rate-analysis-and-convergence)
9. [Gradient Verification](#9-gradient-verification)
10. [Common Failure Modes](#10-common-failure-modes)

---

## 1. The Optimization Problem

### Framing Training as Minimization

Neural network training is fundamentally a non-convex optimization problem over
the parameter space:

```
Find:   θ* = argmin L(θ)
             θ ∈ ℝᵈ

where:
  θ  = all parameters (weights and biases) of the network, flattened to a vector
  d  = total parameter count (can be billions for LLMs)
  L  = loss function, aggregated over the training set:
          L(θ) = (1/N) Σᵢ₌₁ᴺ ℓ(f(xᵢ; θ), yᵢ)
```

### The Loss Landscape

The loss `L(θ)` defines a hypersurface in ℝ^(d+1). Its key features:

| Feature | Description | Implication for Training |
|---|---|---|
| **Global minimum** | Lowest point; f'(θ*)=0, f''(θ*)>0 | Optimal solution — unreachable in practice |
| **Local minima** | Low-loss basin with zero gradient | GD may converge here; often acceptable |
| **Saddle points** | Zero gradient, not a minimum | GD can stall; common in deep nets |
| **Plateaus** | Near-zero gradient over a region | Slow training; hard to detect |
| **Cliffs** | Abrupt large gradient | Instability; need gradient clipping |

**Key insight for deep learning:** Recent theoretical work (Goodfellow et al., 2015;
Choromanska et al., 2015) shows that in high-dimensional spaces, most critical points
with low loss values are saddle points (not local minima), and local minima generally
have loss values close to the global minimum. This is why gradient descent works
surprisingly well despite the non-convexity.

---

## 2. Gradient Descent

### Core Algorithm

The gradient `∇θ L` points in the direction of **steepest ascent** at θ.
Gradient descent moves in the **opposite** direction:

```
θₜ₊₁ = θₜ − η · ∇θ L(θₜ)

where:
  η   = learning rate (step size), η > 0
  ∇θ L = gradient vector ∂L/∂θ ∈ ℝᵈ (same shape as θ)
```

**Why the gradient direction?**

The first-order Taylor expansion of L around θ:
```
L(θ + Δθ) ≈ L(θ) + (∇θ L)ᵀ Δθ
```

To minimize `L(θ + Δθ)` with the constraint `‖Δθ‖ = c` (fixed step size),
the optimal Δθ is in the direction of `-∇θ L`:

```
min_{‖Δθ‖=c}  (∇θ L)ᵀ Δθ  =  -c · ‖∇θ L‖

achieved when  Δθ = -c · ∇θ L / ‖∇θ L‖
```

So Δθ = -η ∇θ L is the locally optimal update.

### Convergence for Convex Functions

For a strongly convex, L-smooth function (both conditions rarely hold in deep learning):

```
Convergence rate:
  L(θₜ) - L(θ*) ≤ (1 - μ/L)ᵗ · [L(θ₀) - L(θ*)]

where:
  μ = strong convexity constant (minimum curvature)
  L = Lipschitz constant of gradient (maximum curvature)
  κ = L/μ = condition number (measures how elongated the loss surface is)

For a quadratic f(θ) = θᵀAθ:
  μ = λ_min(A),   L = λ_max(A),   κ = λ_max/λ_min
```

**Practical implication:** An elongated loss landscape (large κ) causes slow
convergence with gradient descent because the gradient points diagonally,
forcing a zig-zagging path. Adaptive optimizers (Adam) solve this — covered in Topic 5.

---

## 3. Gradient Descent Variants

The key distinction is **how much data** is used to estimate the gradient per update.

### 3.1 Full-Batch (Batch) Gradient Descent

```
∇θ L  =  (1/N) Σᵢ₌₁ᴺ ∇θ ℓ(xᵢ, yᵢ; θ)

θ ← θ - η · ∇θ L

Properties:
  ✓ Exact gradient at each step
  ✓ Stable, smooth convergence curve
  ✗ One gradient step per full pass over data
  ✗ Cannot fit large datasets in memory
  ✗ Cannot take advantage of GPU parallelism efficiently
```

### 3.2 Stochastic Gradient Descent (SGD)

```
Sample one training example (xₜ, yₜ) uniformly at random

∇θ L̃  =  ∇θ ℓ(xₜ, yₜ; θ)   ← noisy estimate of true gradient

θ ← θ - η · ∇θ L̃

Properties:
  ✓ N gradient steps per epoch (one per sample)
  ✓ Noise helps escape local minima and saddle points
  ✓ Works with streaming data (online learning)
  ✗ Very noisy loss curve
  ✗ Cannot parallelize on GPU (batch size 1)
  ✗ Memory bandwidth dominated (poor GPU utilization)
```

**The noise in SGD is not purely harmful.** The gradient noise acts as implicit
regularization — it biases the optimizer toward wider minima that generalize better
(Hochreiter & Schmidhuber, 1997; Keskar et al., 2017).

### 3.3 Mini-Batch Gradient Descent (Standard in Practice)

```
Sample a mini-batch B = {(x₁,y₁), ..., (xB,yB)} uniformly from training set

∇θ L̃  =  (1/B) Σᵢ₌₁ᴮ ∇θ ℓ(xᵢ, yᵢ; θ)

θ ← θ - η · ∇θ L̃

Properties:
  ✓ Gradient steps per epoch: N/B  (N=1000, B=32 → 31 steps/epoch)
  ✓ Parallelizable on GPU: matrix multiply (B × d) is BLAS-optimized
  ✓ Noise is reduced but preserved (B << N)
  ✓ Batch size B is a tunable hyperparameter
```

### Gradient Noise Comparison

```
               Full-Batch GD        Mini-Batch SGD       Stochastic SGD
Gradient noise:     None            O(1/√B)              O(1/√1) = O(1)
Steps/epoch:         1               N/B                    N
GPU efficiency:    Good              Best                   Poor
```

**Batch size effects (empirical findings):**
- `B=1–32`: High noise, acts as regularizer, slower wall-clock time
- `B=32–256`: Sweet spot for most tasks
- `B=256–4096`: Lower noise, trains faster, but may generalize worse (sharp minima)
- `B >> 4096`: Approaches full-batch behavior; requires linear LR scaling rule

**Linear LR scaling rule (Goyal et al., 2017):**
```
If batch size scales by k, scale learning rate by k:
  B → kB  ⟹  η → kη
```

This keeps the expected update magnitude constant across batch sizes.

---

## 4. The Backpropagation Algorithm

Backpropagation is an efficient algorithm for computing `∇θ L` by applying the chain
rule of calculus in reverse order through the computation graph.

### History and Importance

Backpropagation was first described for neural networks by Rumelhart, Hinton & Williams
(1986), though earlier independent derivations exist (Werbos 1974, Parker 1985).
It reduced gradient computation from O(d²) (finite differences for d parameters)
to O(d) — making deep learning computationally feasible.

### Core Idea

**Forward pass:** Compute outputs layer by layer, **caching** all intermediate values.

**Backward pass:** Walk backward through the computation graph, applying the chain rule
to accumulate gradients.

### The "Delta" Notation

For each layer l, define the **error signal** (delta):

```
δˡ = ∂L/∂zˡ   ∈ ℝ^(nˡ)     (gradient of loss w.r.t. pre-activation)
```

This is the key quantity that flows backward. Once we have δˡ for every layer,
the parameter gradients follow immediately:

```
∂L/∂Wˡ = δˡ (aˡ⁻¹)ᵀ
∂L/∂bˡ = δˡ
```

---

## 5. Chain Rule — The Mathematical Foundation

### Univariate Chain Rule

For compositions of scalar functions:
```
y = f(g(x))
dy/dx = (df/dg) · (dg/dx) = f'(g(x)) · g'(x)
```

### Multivariate Chain Rule

When z depends on x through multiple intermediate variables y₁, y₂, ..., yₖ:
```
z = f(y₁(x), y₂(x), ..., yₖ(x))

∂z/∂x = Σᵢ (∂z/∂yᵢ)(∂yᵢ/∂x)
```

### Vector Chain Rule (Jacobian Form)

For vector-to-vector functions y = f(x) where x ∈ ℝⁿ, y ∈ ℝᵐ:
```
Jacobian: J = ∂y/∂x ∈ ℝ^(m×n),   where Jᵢⱼ = ∂yᵢ/∂xⱼ

If z = g(y) and y = f(x), then:
∂z/∂x = (∂z/∂y) · J = (∂z/∂y) · (∂y/∂x)
```

For element-wise functions f (where yᵢ = f(xᵢ) independently):
```
J = diag(f'(x₁), f'(x₂), ..., f'(xₙ))

∂z/∂x = (∂z/∂y) ⊙ f'(x)     ← Hadamard (element-wise) product
```

This is the form used in backprop for activation functions.

---

## 6. Full Backprop Derivation for a 2-Layer MLP

### Network Definition (Batch Form, Column Convention)

```
Input:    X   ∈ ℝ^(n⁰ × N)          n⁰ = features, N = batch size

Layer 1:  Z¹  = W¹X + b¹ 1ᵀ         W¹ ∈ ℝ^(n¹ × n⁰),  b¹ ∈ ℝ^(n¹ × 1)
          A¹  = ReLU(Z¹)             Z¹, A¹ ∈ ℝ^(n¹ × N)

Layer 2:  Z²  = W²A¹ + b² 1ᵀ        W² ∈ ℝ^(1 × n¹),   b² ∈ ℝ^(1 × 1)
          Â   = σ(Z²)                 Z², Â ∈ ℝ^(1 × N)

Loss:     L   = -(1/N) Σᵢ [yᵢ log(âᵢ) + (1-yᵢ) log(1-âᵢ)]   ∈ ℝ
```

### Forward Pass (with Caching)

```
Step 1:  Z¹  ← W¹X + b¹             ← save Z¹ (needed for ReLU backward)
Step 2:  A¹  ← ReLU(Z¹)             ← save A¹ (needed for W² gradient)
Step 3:  Z²  ← W²A¹ + b²            ← save Z² (needed for sigmoid backward)
Step 4:  Â   ← σ(Z²)                 ← save Â  (needed for loss)
Step 5:  L   ← BCE(Â, Y)
```

### Backward Pass (Layer by Layer)

**Step B1 — Output layer: ∂L/∂Z²**

```
∂L/∂Â = -(Y/Â) + (1-Y)/(1-Â)          [derivative of BCE w.r.t. Â]

∂Â/∂Z² = Â ⊙ (1 - Â)                  [sigmoid derivative: σ'(z) = σ(z)(1-σ(z))]

∂L/∂Z² = ∂L/∂Â ⊙ ∂Â/∂Z²              [element-wise chain rule]
         = [-(Y/Â) + (1-Y)/(1-Â)] ⊙ [Â(1-Â)]

Expanding:
         = -Y(1-Â) + (1-Y)Â
         = -Y + YÂ + Â - YÂ
         = Â - Y                        ← BEAUTIFUL SIMPLIFICATION

Define:  δ² = (1/N)(Â - Y)  ∈ ℝ^(1 × N)
              ↑ 1/N for correct averaging in next steps
```

**Step B2 — Output layer parameter gradients**

```
∂L/∂W² = δ² (A¹)ᵀ                    ∈ ℝ^(1 × n¹)
∂L/∂b² = δ² 1  =  Σᵢ δ²ᵢ            ∈ ℝ^(1 × 1)    [sum over batch]
```

**Step B3 — Propagate error to hidden layer**

```
∂L/∂A¹ = (W²)ᵀ δ²                    ∈ ℝ^(n¹ × N)   [backward through W²]
δ¹      = ∂L/∂Z¹ = ∂L/∂A¹ ⊙ ReLU'(Z¹)  ∈ ℝ^(n¹ × N)   [backward through ReLU]

         where ReLU'(z) = 1 if z > 0 else 0
```

**Step B4 — Hidden layer parameter gradients**

```
∂L/∂W¹ = δ¹ Xᵀ                       ∈ ℝ^(n¹ × n⁰)
∂L/∂b¹ = δ¹ 1  =  Σᵢ δ¹ᵢ            ∈ ℝ^(n¹ × 1)    [sum over batch]
```

### Complete Summary Table

| Gradient | Formula | Shape |
|---|---|---|
| δ² (output delta) | (1/N)(Â − Y) | (1, N) |
| ∂L/∂W² | δ²(A¹)ᵀ | (1, n¹) |
| ∂L/∂b² | Σᵢ δ²ᵢ | (1, 1) |
| ∂L/∂A¹ | (W²)ᵀδ² | (n¹, N) |
| δ¹ (hidden delta) | ∂L/∂A¹ ⊙ ReLU'(Z¹) | (n¹, N) |
| ∂L/∂W¹ | δ¹Xᵀ | (n¹, n⁰) |
| ∂L/∂b¹ | Σᵢ δ¹ᵢ | (n¹, 1) |

### General Recursive Backprop Formula

For an L-layer network, the backward pass follows the recurrence:

```
δᴸ = (1/N)(Â - Y)                         [output layer, BCE+sigmoid]

For l = L-1, L-2, ..., 1:
    dAˡ = (Wˡ⁺¹)ᵀ δˡ⁺¹                   [error through weights]
    δˡ  = dAˡ ⊙ fˡ'(Zˡ)                  [error through activation]

For all l:
    ∂L/∂Wˡ = δˡ (Aˡ⁻¹)ᵀ
    ∂L/∂bˡ = Σᵢ δˡᵢ (sum over batch)
```

### Why Caching is Essential

The backward pass reuses:
- `Z¹` → to compute ReLU'(Z¹) (which neurons were active)
- `A¹` → to compute ∂L/∂W²
- `A⁰ = X` → to compute ∂L/∂W¹

Without caching, we would need to recompute the forward pass for every parameter
during backprop — O(L) extra forward passes per backward pass. Caching reduces
backprop to a single backward sweep: O(d) time, same as forward pass.

**Memory cost:** We must store all activations for the entire batch simultaneously.
This is why GPU memory limits batch size — larger batches need more cached activation memory.

---

## 7. Computational Graphs and PyTorch Autograd

### Computational Graphs

Every neural network computation can be represented as a **directed acyclic graph (DAG)**:
- **Nodes** = tensors (inputs, parameters, intermediates, outputs)
- **Edges** = operations (matrix multiply, add, activation, loss)

```
Example: z = (x + w) * σ(w)

     x ────┐
           [+]──→ a ───[*]──→ z
     w ────┘       ↗
               [σ]
     w ────────┘

Forward: x=2, w=1 → a=3, σ(1)=0.731 → z=3*0.731=2.193
Backward: ∂z/∂w = a·σ'(w) + σ(w) = 3·0.197 + 0.731 = 1.322
```

### PyTorch Autograd: Dynamic Computation Graphs

PyTorch builds the computation graph **dynamically** (during the forward pass):
- Each `Tensor` with `requires_grad=True` tracks operations in its `.grad_fn`
- After `loss.backward()`, gradients are computed in **reverse topological order**

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1      # Builds graph: [Power, Mul, Add, Add]
y.backward()                  # Walks graph backward: dy/dx = 2x + 3 = 7
print(x.grad)                 # tensor(7.)
```

**retain_graph=True:**
By default, after `backward()`, PyTorch frees the computation graph (saves memory).
If you need multiple backward passes (e.g., second-order methods, GAN training),
use `loss.backward(retain_graph=True)`.

**Gradient Accumulation:**
PyTorch **adds** gradients to `.grad` rather than replacing. This is intentional
(for gradient accumulation over micro-batches), but requires calling
`optimizer.zero_grad()` or `param.grad.zero_()` before each backward pass.

---

## 8. Learning Rate: Analysis and Convergence

### Effect on Convergence

For a quadratic loss `f(θ) = θᵀAθ` with eigenvalues λ_min, λ_max:

```
The GD update is:  θₜ₊₁ = (I - ηA)θₜ

Convergence requires spectral radius ρ(I - ηA) < 1:
  max |1 - η·λᵢ| < 1  for all eigenvalues λᵢ

  ⟹  0 < η < 2/λ_max

Optimal η (minimizes convergence rate):
  η* = 2/(λ_min + λ_max)
  ρ* = (κ-1)/(κ+1)   where κ = λ_max/λ_min  (condition number)
```

### Geometric Behavior for Different Learning Rates

```
f(w₁, w₂) = (w₁-1)² + 4(w₂-1)²
  Curvature along w₁: 2    (∂²f/∂w₁² = 2)
  Curvature along w₂: 8    (∂²f/∂w₂² = 8)
  Condition number:   κ = 8/2 = 4
  Convergence limit:  η < 2/8 = 0.25

η = 0.05   → both directions converge slowly
η = 0.12   → near-optimal convergence
η = 0.24   → stable but oscillates in w₂ (high-curvature direction)
η = 0.30   → diverges in w₂ (exceeds stability limit)
```

### Learning Rate Schedules (Preview)

Fixed learning rates are suboptimal. Common schedules:

| Schedule | Formula | When to Use |
|---|---|---|
| Step decay | η = η₀ × γ^(t//s) | Simple, works well |
| Cosine annealing | η = η_min + ½(η_max-η_min)(1+cos(πt/T)) | Default for vision |
| Warmup + decay | Linear warmup then cosine | Transformers |
| Cyclical LR | Oscillates between bounds | Faster convergence |

---

## 9. Gradient Verification

### Why Verify Gradients?

Manual backprop implementations frequently have subtle bugs:
- Off-by-one errors in shape transposition
- Missing 1/N normalization factors
- Incorrect Hadamard vs matrix multiplication
- Sign errors in the BCE derivative

Gradient checking provides a mathematical guarantee that your backprop is correct.

### Numerical Gradient (Central Finite Differences)

```
∂L/∂θᵢ ≈ [L(θ + εeᵢ) - L(θ - εeᵢ)] / (2ε)

where:
  eᵢ  = unit vector in dimension i
  ε   = small perturbation (typically 1e-5 to 1e-7)

Error:  O(ε²)  ← second-order accurate (central differs from forward by O(ε²) vs O(ε))
```

### Relative Error Criterion

```
Relative error (Karpathy criterion):

  e_rel = ‖∇_anal - ∇_num‖₂ / (‖∇_anal‖₂ + ‖∇_num‖₂ + ε)

Interpretation:
  e_rel < 1e-7   → Excellent (floating-point precision limit)
  e_rel < 1e-5   → Good (acceptable for most implementations)
  e_rel < 1e-3   → Suspicious (check implementation)
  e_rel > 1e-2   → Bug likely present
```

### When to Run Gradient Checks

- After implementing a new layer's backward pass
- Before major training runs
- **Not during training** (gradient checking is O(d) forward passes — very slow)
- Use a small batch (N=5-10) and small model for speed

---

## 10. Common Failure Modes

### 10.1 Exploding Gradients

**Symptom:** Loss becomes NaN or Inf within first few epochs.

**Cause:** Gradient norms grow exponentially through deep layers.

**Solutions:**
```
1. Gradient clipping:   clip ‖∇θ‖₂ ≤ threshold (typically 1.0 or 5.0)
                        θ.grad.data = θ.grad.data * (clip_val / max(‖g‖, clip_val))
2. Lower learning rate
3. Better initialization
4. Batch Normalization (stabilizes activations)
```

### 10.2 Vanishing Gradients

**Symptom:** Early layers have near-zero gradients; only output layers update.

**Cause:** Chain rule multiplies many values < 1 (sigmoid derivatives ≤ 0.25).

**Solutions:**
```
1. ReLU / GELU activations (gradient = 1 for positive inputs)
2. Residual connections (skip connections add gradient highway)
3. Batch Normalization (rescales activations)
4. Proper initialization (He/Xavier)
```

### 10.3 Learning Rate Too High

**Symptom:** Loss oscillates or diverges immediately.

**Diagnosis:** Plot loss per step — if it bounces, LR is too high.

**Fix:** Reduce η by 10× and retrain. Use learning rate finder (LR range test).

### 10.4 Gradient Accumulation Bug

**Symptom:** Training appears to work but is slower than expected; gradients grow.

**Cause:** Forgetting `optimizer.zero_grad()` causes gradients to accumulate.

**Fix:** Always call `optimizer.zero_grad()` before each `loss.backward()`.

### 10.5 Incorrect Loss Reduction

**Symptom:** Loss values are unexpectedly large; gradients scale with batch size.

**Cause:** Using `reduction='sum'` instead of `reduction='mean'` in the loss.

**Fix:** `nn.BCELoss(reduction='mean')` (the default). Or manually divide by N.

---

## Key Equations Summary

| Concept | Formula |
|---|---|
| GD update | θ ← θ − η∇L(θ) |
| Mini-batch gradient | ∇θ L̃ = (1/B) Σᵢ∈B ∇θ ℓ(xᵢ,yᵢ) |
| Output delta (BCE+σ) | δ² = (1/N)(Â−Y) |
| Weight gradient | ∂L/∂Wˡ = δˡ(Aˡ⁻¹)ᵀ |
| Bias gradient | ∂L/∂bˡ = Σᵢ δˡᵢ |
| Error propagation | δˡ = (Wˡ⁺¹)ᵀδˡ⁺¹ ⊙ fˡ'(Zˡ) |
| Numerical gradient | ∂L/∂θᵢ ≈ [L(θ+εeᵢ)−L(θ−εeᵢ)]/(2ε) |
| Relative error | ‖∇anal−∇num‖/(‖∇anal‖+‖∇num‖) |
| Convergence limit | η < 2/λ_max |

---

*Previous: [Topic 2 — Activation Functions](../02-activation-functions/theory.md)*
*Next:
