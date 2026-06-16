# Theory: Activation Functions

**Phase 1 — Topic 2 | Deep Learning Mastery Repository**

---

## Table of Contents

1. [Why Activation Functions Exist](#1-why-activation-functions-exist)
2. [Desirable Properties](#2-desirable-properties)
3. [Classic Activations](#3-classic-activations)
4. [Modern Activations](#4-modern-activations)
5. [Output Layer Activations](#5-output-layer-activations)
6. [The Vanishing Gradient Problem](#6-the-vanishing-gradient-problem)
7. [The Dead Neuron Problem](#7-the-dead-neuron-problem)
8. [Comparative Analysis](#8-comparative-analysis)
9. [Practical Selection Guide](#9-practical-selection-guide)

---

## 1. Why Activation Functions Exist

### The Linearity Collapse Theorem

**Claim:** A deep neural network with no activation functions (or only linear activations) is equivalent to a single linear layer — regardless of depth.

**Proof:**
```
Without activations, each layer computes:
    aˡ = Wˡ aˡ⁻¹ + bˡ

Composing two layers:
    a² = W²(W¹x + b¹) + b²
       = W²W¹x + W²b¹ + b²
       = W_eff x + b_eff

By induction for L layers:
    aᴸ = (Wᴸ ··· W²W¹) x + b_eff
       =  W_combined x + b_combined

∴ A depth-L linear network = A depth-1 linear network.
```

**Consequence:** No matter how many linear layers we stack, the model can only represent linear transformations. The decision boundary is always a hyperplane. XOR, circles, spirals — impossible.

**Activation functions break this collapse** by introducing non-linearity after each linear transformation:

```
aˡ = f(Wˡ aˡ⁻¹ + bˡ)    ← f is non-linear
```

Now compositions of layers cannot be collapsed into a single layer.

### What Non-Linearity Buys Us

| Without Activation | With Activation |
|---|---|
| Linear decision boundary only | Arbitrary curved boundaries |
| Depth is redundant | Depth adds representational power |
| Cannot solve XOR | Can solve any continuous function (UAT) |
| Features: raw inputs | Features: hierarchical abstractions |

---

## 2. Desirable Properties

A good activation function ideally satisfies:

| Property | Description | Why It Matters |
|---|---|---|
| **Non-linearity** | f(az+b) ≠ af(z)+b | Prevents linearity collapse |
| **Differentiable** | f'(z) exists almost everywhere | Required for backpropagation |
| **Non-saturating** | f'(z) ≠ 0 for large \|z\| | Prevents vanishing gradients |
| **Zero-centered** | E[f(z)] ≈ 0 | Faster convergence (symmetric gradients) |
| **Monotonic** | f'(z) ≥ 0 always | Simpler loss landscape |
| **Computationally cheap** | Avoids exp/log when possible | Efficiency in deep nets |
| **Bounded output** | f(z) ∈ [a, b] | Prevents activation explosion |

No single activation satisfies all properties — every choice involves trade-offs.

---

## 3. Classic Activations

### 3.1 Heaviside Step Function

```
         ⎧ 1   if z ≥ 0
H(z)  =  ⎨
         ⎩ 0   if z < 0

H'(z) = 0   everywhere (except z=0 where it's undefined)
```

**Range:** {0, 1}

**Properties:**
- Non-differentiable at z = 0
- Gradient = 0 everywhere → **cannot be trained with gradient descent**
- Historical: used in the original Perceptron (Topic 1)
- Not used in modern networks

---

### 3.2 Sigmoid (Logistic)

```
                   1
σ(z)  =  ──────────────
               1 + e⁻ᶻ

         e⁻ᶻ          1          1
σ'(z) = ───────── = ─────── · ─────── = σ(z)(1 − σ(z))
         (1+e⁻ᶻ)²    1+e⁻ᶻ    1+e⁻ᶻ
```

**Range:** (0, 1)

**Maximum derivative:** σ'(0) = 0.25 (at z = 0)

**Numerically stable implementation:**
```python
# Naïve: exp(-z) overflows for large negative z
σ(z) = 1 / (1 + exp(-z))             # overflow when z → -∞

# Stable: use identity 1/(1+e^{-z}) = e^z/(1+e^z)
σ(z) = {  1/(1+exp(-z))   if z ≥ 0
        {  exp(z)/(1+exp(z))  if z < 0
```

**Pros:**
- Smooth and differentiable everywhere
- Output interpretable as probability ∈ (0, 1)
- Historical importance; still used in output layers for binary classification

**Cons:**
- **Vanishing gradient:** σ'(z) ≤ 0.25. In a 10-layer network: (0.25)^10 ≈ 10^{-6}. Gradients disappear.
- **Not zero-centered:** Output always positive (0 to 1) → gradients for weights always have same sign → zig-zag optimization

---

### 3.3 Hyperbolic Tangent (Tanh)

```
          eᶻ − e⁻ᶻ
tanh(z) = ─────────   =   2σ(2z) − 1   (related to sigmoid)
          eᶻ + e⁻ᶻ

tanh'(z) = 1 − tanh²(z)
```

**Range:** (−1, 1)

**Maximum derivative:** tanh'(0) = 1.0

**Properties:**
- Zero-centered output (unlike sigmoid) → faster convergence in hidden layers
- Still suffers vanishing gradient for large |z|, but gradient is 4× larger than sigmoid at z=0
- Preferred over sigmoid for hidden layers when saturation is tolerable
- tanh is a rescaled, shifted sigmoid: tanh(z) = 2σ(2z) - 1

**Gradient comparison at z=2:**
```
σ'(2)    = 0.105
tanh'(2) = 0.071
```
Both saturate, but tanh's larger max gradient helps slightly in early training.

---

## 4. Modern Activations

### 4.1 Rectified Linear Unit (ReLU)

```
ReLU(z) = max(0, z) = { z   if z > 0
                       { 0   if z ≤ 0

ReLU'(z) = { 1   if z > 0
           { 0   if z ≤ 0
```

**Range:** [0, ∞)

**Introduced:** LeCun et al. (2010), popularized by Krizhevsky (AlexNet, 2012)

**Why ReLU revolutionized deep learning:**

```
With sigmoid in 10-layer net:
  ∂L/∂W¹ ≈ (0.25)^10 · ∂L/∂aᴸ  ≈ 10⁻⁶ · (upstream gradient)
  → Learning stops

With ReLU in 10-layer net:
  ∂L/∂W¹ = ∏ᵢ ReLU'(zⁱ) · ∂L/∂aᴸ  = ∏ᵢ {0,1} · (upstream gradient)
  → Gradient = 1 for all active neurons → no vanishing!
```

**Pros:**
- No vanishing gradient for positive inputs
- Computationally cheap: `max(0, z)` — no exponential
- Sparse activation: ~50% of neurons are zero → efficient computation
- Empirically trains deep networks where sigmoid/tanh fail

**Cons:**
- **Dead neurons:** If z ≤ 0 for all training inputs, gradient = 0 always → neuron never updates
- Not zero-centered (always ≥ 0)
- Unbounded output (can cause large activations)

---

### 4.2 Leaky ReLU

```
LeakyReLU(z) = max(αz, z) = { z    if z > 0
                             { αz   if z ≤ 0

                              α ∈ (0, 1), typically 0.01

LeakyReLU'(z) = { 1   if z > 0
               { α   if z ≤ 0
```

**Range:** (−∞, ∞)

**Fix for dead neurons:** The gradient for z ≤ 0 is α (not 0), so neurons can always recover.

**Mathematical guarantee:** `|LeakyReLU'(z)| = max(α, 1) · indicator`, so gradient is always at least α.

---

### 4.3 Parametric ReLU (PReLU)

```
PReLU(z) = max(αz, z)    where α is a LEARNED parameter
```

α is initialized to 0.25 and updated by backpropagation. Each neuron (or each layer) can have its own α. Subsumes both ReLU (α=0) and Leaky ReLU (α=fixed).

---

### 4.4 Exponential Linear Unit (ELU)

```
ELU(z, α) = { z               if z > 0
            { α(eᶻ − 1)      if z ≤ 0

ELU'(z, α) = { 1               if z > 0
             { α·eᶻ = ELU+α   if z ≤ 0
```

**Range:** (−α, ∞), typically (−1, ∞) with α=1

**Key property:** Negative saturation at −α, but smooth exponential transition (not hard kink).

**Advantages over ReLU:**
- Mean activation closer to zero (negative values bring mean toward 0)
- Smooth at z = 0 (ELU'(0) = α from left, 1 from right — still not smooth unless α=1)
- No dead neurons
- Better robustness to noise

---

### 4.5 Scaled Exponential Linear Unit (SELU)

```
SELU(z) = λ · ELU(z, α)
        = λ · { z               if z > 0
              { α(eᶻ − 1)      if z ≤ 0

λ = 1.0507009873554804934193349852946
α = 1.6732632423543772848170429916717
```

**Self-normalizing property:**

These specific constants were derived by Klambauer et al. (2017) by solving the fixed-point equations:

```
E[SELU(z)] = 0    and    Var[SELU(z)] = 1

when z ~ N(0, 1)
```

The fixed-point constraint means that if inputs have mean=0 and variance=1, the SELU output also has mean≈0 and variance≈1 — the normalization is **built into the activation function**.

**When to use:** Deep fully-connected networks where Batch Normalization would be awkward (RNNs, small batch sizes). Requires `lecun_normal` weight initialization.

---

### 4.6 Gaussian Error Linear Unit (GELU)

```
GELU(z) = z · Φ(z)

where Φ(z) is the CDF of the standard normal distribution:
  Φ(z) = P(X ≤ z),  X ~ N(0,1)

Exact:        GELU(z) = z · ½[1 + erf(z/√2)]

Approximation (used in practice):
  GELU(z) ≈ 0.5z · [1 + tanh(√(2/π) · (z + 0.044715z³))]
```

**Intuition:** GELU stochastically gates the input by its probability of being positive under a standard normal. High positive values pass through completely; large negative values are gated to near-zero. Unlike ReLU, the gating is smooth and differentiable.

**Where used:** BERT, GPT-2, GPT-3, most modern Transformer architectures.

**Advantages:**
- Smooth, non-monotonic (has a slight dip for small negative values)
- Strong empirical performance on NLP benchmarks
- Avoids the sharp kink at z = 0 that can cause optimization instability

---

### 4.7 SiLU / Swish

```
SiLU(z) = z · σ(z) = z / (1 + e⁻ᶻ)

SiLU'(z) = σ(z) + z · σ(z)(1 − σ(z))
          = σ(z)(1 + z(1 − σ(z)))
```

**Range:** Approximately (−0.28, ∞)

**Proposed by:** Google Brain (Ramachandran et al., 2017)

**Properties:**
- Non-monotonic: has a minimum around z ≈ −1.28
- Self-gated: multiplies input by its own sigmoid
- Smooth everywhere (unlike ReLU's kink)
- SiLU and Swish are the same function with β=1
- Used in EfficientNet, MobileNetV3

---

### 4.8 Mish

```
Mish(z) = z · tanh(softplus(z))
         = z · tanh(ln(1 + eᶻ))

Mish'(z) = tanh(sp) + z · sech²(sp) · σ(z)
           where sp = softplus(z)
```

**Range:** Approximately (−0.31, ∞)

**Properties:**
- Smooth, non-monotonic, unbounded above
- Self-regularizing (bounded below by ~−0.31)
- Slightly outperforms Swish on some vision tasks (YOLOv4)

---

## 5. Output Layer Activations

The output layer activation depends on the **task type**, not the architecture:

### 5.1 Binary Classification — Sigmoid

```
ŷ = σ(z) ∈ (0, 1)    → interpreted as P(class=1 | x)

Loss: Binary Cross-Entropy
  L = −[y log ŷ + (1−y) log(1−ŷ)]
```

Decision: predict class 1 if ŷ ≥ 0.5

### 5.2 Multi-class Classification — Softmax

```
           e^{zₖ}
ŷₖ  =  ──────────────     k = 1, 2, ..., K
          Σⱼ e^{zⱼ}

Properties:
  ŷₖ ∈ (0, 1)    for all k
  Σₖ ŷₖ = 1       (valid probability distribution)
```

**Numerically stable implementation:**
```
Naïve: exp(1000) = inf  →  inf/inf = NaN  (BROKEN)

Stable trick:
  z_shifted = z − max(z)         ← no change in value since max cancels
  ŷₖ = e^{z_shifted,k} / Σⱼ e^{z_shifted,j}

Why correct:
  e^{zₖ − max(z)} / Σⱼ e^{zⱼ − max(z)}
= (e^{−max(z)} · e^{zₖ}) / (e^{−max(z)} · Σⱼ e^{zⱼ})
= e^{zₖ} / Σⱼ e^{zⱼ}    ✓  (unchanged)
```

**Loss:** Categorical Cross-Entropy
```
L = −Σₖ yₖ log ŷₖ
```
Note: **Never** apply `nn.Softmax()` and then `nn.NLLLoss()`. Use `nn.CrossEntropyLoss()` directly, which combines `LogSoftmax + NLLLoss` in a numerically stable way.

### 5.3 Regression — Identity (No Activation)

```
ŷ = z ∈ (−∞, ∞)

Loss: Mean Squared Error (MSE) or Mean Absolute Error (MAE)
  L = (1/N) Σᵢ (yᵢ − ŷᵢ)²
```

No activation needed — we want unbounded real output.

---

## 6. The Vanishing Gradient Problem

### Mathematical Derivation

For an L-layer network, the gradient at the first layer (via chain rule) is:

```
∂L       ∂L      ∂aᴸ   ∂aᴸ⁻¹         ∂a¹
──── = ───── · ─────── · ──────── · ··· ────────
∂W¹     ∂aᴸ    ∂aᴸ⁻¹    ∂aᴸ⁻²         ∂W¹

     = ∂L/∂aᴸ · ∏ₗ₌₂ᴸ f'(zˡ) · Wˡ · f'(z¹) · a⁰
```

For sigmoid, f'(z) ≤ 0.25 for all z. Therefore:

```
‖∂L/∂W¹‖ ≤ ‖∂L/∂aᴸ‖ · (0.25)^L · ∏ₗ ‖Wˡ‖

For L = 10:  (0.25)^10 ≈ 9.5 × 10⁻⁷
For L = 20:  (0.25)^20 ≈ 9.1 × 10⁻¹³
```

Early layers essentially stop learning. This was the core obstacle to deep learning before ReLU.

### Why ReLU Resists Vanishing Gradients

```
ReLU'(z) ∈ {0, 1}

For active neurons (z > 0):  ReLU'(z) = 1
Product of 1s across L layers = 1 (no decay!)

∂L/∂W¹ = ∂L/∂aᴸ · (1)^L_active · upstream_terms
        = ∂L/∂aᴸ · upstream_terms (for active path)
```

The gradient can flow unchanged through ReLU neurons, enabling training of networks with 100+ layers.

### Exploding Gradients

The opposite problem occurs when weights are large:

```
If ‖Wˡ‖ > 1 and f'(zˡ) > 1:
  ‖∂L/∂W¹‖ → ∞ as L increases

Solutions:
  1. Gradient clipping: clip ‖∂L/∂θ‖ ≤ threshold (standard in RNNs)
  2. Careful initialization (Xavier, He)
  3. Batch Normalization (Topic 5)
  4. Residual connections (ResNet — Phase 2)
```

---

## 7. The Dead Neuron Problem

### Definition

A ReLU neuron is **dead** if:

```
∀ x ∈ training set:   zⁱ = Wⁱ xⁱ + bⁱ ≤ 0

⟹  ReLU(zⁱ) = 0   and   ∂L/∂Wⁱ = 0   for all samples

⟹  Wⁱ never updates  ⟹  neuron stays dead forever
```

### Causes

```
1. Large learning rate:
   W ← W − η · ∂L/∂W  can overshoot, sending W to large negative values
   → zⁱ = Wⁱx + bⁱ becomes strongly negative for all x

2. Unfavorable initialization:
   If b is initialized very negatively, z = Wx + b < 0 from the start

3. Large negative weight updates during training
```

### Diagnosis

```python
# After training, check fraction of neurons with zero activation for all samples
activations = []  # shape: (n_samples, n_neurons)
dead_fraction = (activations == 0).all(axis=0).mean()
# Healthy: < 5%  |  Concerning: > 20%
```

### Solutions

| Solution | Mechanism | Trade-off |
|---|---|---|
| Leaky ReLU | α ≠ 0 → gradient always exists | Slight accuracy reduction in some tasks |
| PReLU | Learnable α | Extra parameters |
| ELU | Smooth negative saturation | Slower computation |
| Lower learning rate | Prevents drastic weight updates | Slower convergence |
| He initialization | Correct variance for ReLU | Must use right init |

---

## 8. Comparative Analysis

```
                 Vanishing   Dead      Zero-    Smooth   Computation
Activation       Gradient   Neurons  Centered   (C¹)     Cost
─────────────────────────────────────────────────────────────────────
Step             N/A (0)     N/A       No        No      O(1)
Sigmoid          SEVERE      No        No        Yes     O(exp)
Tanh             MODERATE    No        Yes       Yes     O(exp)
ReLU             None*       SEVERE    No        No      O(1)
Leaky ReLU       None*       None      No        No      O(1)
ELU              None*       None      ~Yes      Yes     O(exp,neg)
SELU             None*       None      ~Yes      Yes     O(exp,neg)
GELU             None*       None      ~Yes      Yes     O(erf/tanh)
SiLU/Swish       None*       None      ~Yes      Yes     O(exp)
Mish             None*       None      ~Yes      Yes     O(exp)

*No vanishing for active/positive neurons; saturation may still occur for
 very deep networks with all ReLU-family functions, but is far less severe.
```

### Output Range Comparison

```
z:       -4   -3   -2   -1    0    1    2    3    4

Step:     0    0    0    0    1    1    1    1    1
Sigmoid: .02  .05  .12  .27  .50  .73  .88  .95  .98
Tanh:    -.99 -.99 -.96 -.76  0   .76  .96  .99  .99
ReLU:     0    0    0    0    0    1    2    3    4
GELU:    ~0   ~0  -.05 -.16   0   .84  2.0  3.0  4.0
Swish:   -.07 -.14 -.24 -.27  0   .73  1.76 2.86 3.93
```

---

## 9. Practical Selection Guide

### Hidden Layers

```
Default choice:  ReLU
  → Fast, simple, works well for most tasks

If dead neurons are a problem:  Leaky ReLU or ELU
  → Small α (0.01-0.1) is usually sufficient

If training transformers / NLP:  GELU
  → Standard in BERT, GPT, T5, Llama

If training EfficientNet / modern CNN:  SiLU/Swish
  → Slightly better than ReLU on image classification

If no batch normalization:  SELU
  → Self-normalizing; requires lecun_normal init and AlphaDropout

Avoid for hidden layers:  Sigmoid, Tanh
  → Reserve for output layers or gating mechanisms (LSTM gates)
```

### Output Layer

```
Binary classification:      Sigmoid    → ŷ ∈ (0,1)
Multi-class classification:  Softmax   → probability vector
Regression:                 None/Identity → ŷ ∈ ℝ
Multi-label classification:  Sigmoid   → independent probabilities per class
```

### Architecture-Specific Defaults

| Architecture | Hidden | Output |
|---|---|---|
| MLP (tabular data) | ReLU or GELU | task-dependent |
| CNN | ReLU | task-dependent |
| ResNet | ReLU | Softmax |
| Transformer | GELU | Softmax/None |
| LSTM/GRU | Tanh + Sigmoid (gates) | task-dependent |
| GAN generator | ReLU / Tanh | Tanh (images) |
| GAN discriminator | Leaky ReLU | Sigmoid |

---

## Key Equations Summary

| Function | Formula | Derivative |
|---|---|---|
| Sigmoid | σ(z) = 1/(1+e⁻ᶻ) | σ(z)(1−σ(z)), max=0.25 |
| Tanh | (eᶻ−e⁻ᶻ)/(eᶻ+e⁻ᶻ) | 1−tanh²(z), max=1 |
| ReLU | max(0,z) | H(z) ∈ {0,1} |
| Leaky ReLU | max(αz,z) | α or 1 |
| ELU | z or α(eᶻ−1) | 1 or ELU+α |
| SELU | λ·ELU(z,α) | λ or λαeᶻ |
| GELU | z·Φ(z) | Φ(z)+z·φ(z) |
| SiLU | z·σ(z) | σ(z)(1+z(1−σ(z))) |
| Softmax | eᶻᵢ/Σeᶻⱼ | sᵢ(1−sᵢ) diag |

---

*Previous: [Topic 1 — Perceptron & MLP](../01-perceptron-and-mlp/theory.md)*
*Next: [Topic 3 — Gradient Descent & Backpropagation](../03-gradient-descent-and-backprop/theory.md)*
