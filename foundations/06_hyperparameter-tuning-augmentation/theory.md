# Theory: Hyperparameter Tuning & Data Augmentation

**Deep Learning Mastery Repository**

---

## Table of Contents
1. [The Hyperparameter Optimization Problem](#1-the-hyperparameter-optimization-problem)
2. [Key Hyperparameters](#2-key-hyperparameters)
3. [Search Strategies](#3-search-strategies)
4. [Learning Rate Schedules](#4-learning-rate-schedules)
5. [Data Augmentation](#5-data-augmentation)
6. [Cross-Validation for Hyperparameter Selection](#6-cross-validation-for-hyperparameter-selection)

---

## 1. The Hyperparameter Optimization Problem

### Hyperparameters vs Parameters

```
Parameters (θ):      Learned via gradient descent during training
                      e.g. weights W, biases b

Hyperparameters (λ): Set BEFORE training; control the learning PROCESS itself
                      e.g. learning rate, batch size, number of layers
```

### Bilevel Optimization Formulation

```
λ* = argmin_λ  L_val(θ*(λ))

subject to:    θ*(λ) = argmin_θ  L_train(θ; λ)
```

This is a **bilevel optimization** problem: the outer loop searches for the
best hyperparameters, while the inner loop trains a full model for each
candidate hyperparameter setting. The inner loop is itself a full (and
expensive) optimization, making hyperparameter tuning computationally costly.

---

## 2. Key Hyperparameters

### 2.1 Learning Rate (η)

The single most impactful hyperparameter. Covered in depth in Topic 3.

```
Too small:  Slow convergence, may get stuck in poor local minima
Too large:  Divergence, oscillation, NaN losses
Typical search range: [1e-5, 1e-1] on log scale
```

### 2.2 Batch Size (B)

```
Small B (8-32):    More gradient noise → regularization effect; slower per-epoch
Large B (256+):    Less noise → faster per-epoch; may need LR adjustment (Topic 3)
Memory constraint: B limited by GPU VRAM (activations scale linearly with B)
```

### 2.3 Number of Hidden Layers / Depth

```
Shallow (1-2 layers): Fast training, limited representational capacity
Deep (10+ layers):    Higher capacity, harder to train (vanishing gradients),
                       needs residual connections / normalization (Phase 2+)
```

### 2.4 Hidden Layer Width

```
Narrow:  Fewer parameters, may underfit (bottleneck effect)
Wide:    More parameters, slower per-step, may overfit without regularisation
```

### 2.5 Regularization Strength (λ, dropout p)

Covered in Topic 5. Typical search: λ ∈ [1e-5, 1e-1], p ∈ [0.1, 0.5].

### 2.6 Optimizer-Specific Hyperparameters

```
Adam:    β1 (momentum, default 0.9), β2 (variance, default 0.999), ε (1e-8)
SGD:     momentum coefficient (0.9 typical)
```

---

## 3. Search Strategies

### 3.1 Grid Search

```
Define a discrete grid for each hyperparameter:
  η ∈ {1e-4, 1e-3, 1e-2}
  B ∈ {16, 32, 64}

Total configurations: 3 × 3 = 9 (exhaustive cartesian product)

Pros: Simple, exhaustive, reproducible
Cons: Exponential blowup with more hyperparameters (curse of dimensionality)
      Wastes compute on unpromising regions
      Fixed grid may miss the optimal value between grid points
```

### 3.2 Random Search (Bergstra & Bengio, 2012)

```
Sample each hyperparameter independently from a distribution:
  η ~ LogUniform(1e-5, 1e-1)
  B ~ Choice([16, 32, 64, 128])

Run N random configurations (N << grid size)
```

**Why random search often beats grid search:**

```
Key insight: most hyperparameters have LOW effective dimensionality.
Often only 1-2 hyperparameters matter for a given problem.

Grid search:    wastes evaluations on the "unimportant" axis
Random search:  every evaluation explores a DIFFERENT value of EVERY parameter
                 → much better coverage of the "important" axis with the
                   same compute budget
```

```
Grid Search (9 points, 3×3):       Random Search (9 points):
  η                                  η
  │ ● ─ ● ─ ●                       │  ●      ●
  │ ● ─ ● ─ ●                       │      ●        ●
  │ ● ─ ● ─ ●                       │ ●        ●  ●
  └──────────── B                   └──────────── B
  Only 3 unique η values tested      9 unique η values tested!
```

### 3.3 Bayesian Optimization

```
Build a surrogate probabilistic model (typically Gaussian Process) of:
  f(λ) = L_val(θ*(λ))

Iteratively:
  1. Fit GP to all (λᵢ, f(λᵢ)) observed so far
  2. Use an acquisition function (e.g., Expected Improvement) to choose
     the next λ to evaluate — balances EXPLORATION vs EXPLOITATION
  3. Evaluate f(λ) at the chosen point, add to observations
  4. Repeat
```

**Acquisition function — Expected Improvement (EI):**
```
EI(λ) = E[max(f(λ_best) − f(λ), 0)]

Favors points that are either:
  - Likely to improve on the best observed value (exploitation)
  - Highly uncertain (exploration) — could be a hidden optimum
```

**Pros:** Sample-efficient (fewer evaluations needed than random/grid search)
**Cons:** Sequential (hard to parallelize); GP scales poorly beyond ~20 dimensions

### 3.4 Hyperband / Successive Halving

```
Idea: allocate small budgets to many configurations, then progressively
      eliminate the worst performers and allocate more budget to survivors.

Round 1: Train 81 configs for 1 epoch each       → keep top 27
Round 2: Train 27 configs for 3 epochs each       → keep top 9
Round 3: Train 9 configs for 9 epochs each        → keep top 3
Round 4: Train 3 configs for 27 epochs each       → keep top 1

Total compute ≈ much less than training all 81 configs for 27 epochs each.
```

This is an early-stopping-based bandit algorithm — poorly performing
configurations are killed early, freeing compute for promising ones.

### 3.5 Population-Based Training (PBT)

```
Train a population of models in parallel.
Periodically:
  - "Exploit": copy weights from better-performing members to worse ones
  - "Explore": perturb hyperparameters of copied members (mutate η, etc.)

This jointly optimizes weights AND hyperparameters DURING training,
avoiding the need for separate full training runs per configuration.
```

---

## 4. Learning Rate Schedules

A fixed learning rate is rarely optimal throughout training. Schedules adapt η over time.

### 4.1 Step Decay

```
η_t = η_0 · γ^⌊t/s⌋

e.g., η_0=0.1, γ=0.1, s=30:
  Epochs 0-29:  η=0.1
  Epochs 30-59: η=0.01
  Epochs 60-89: η=0.001
```

### 4.2 Exponential Decay

```
η_t = η_0 · e^(-kt)
```

Smooth continuous decay; k controls decay speed.

### 4.3 Cosine Annealing

```
η_t = η_min + ½(η_max − η_min)·(1 + cos(πt/T))

At t=0: η = η_max
At t=T: η = η_min
Smooth, asymmetric decay (slow start, fast middle, slow end)
```

### 4.4 Warmup + Decay (Transformer Standard)

```
Phase 1 (Warmup, t < t_warmup):
  η_t = η_max · (t / t_warmup)         ← linear ramp-up

Phase 2 (Decay, t ≥ t_warmup):
  η_t = η_max · schedule(t)             ← cosine, inverse sqrt, etc.
```

**Why warmup?** At initialization, gradients are often poorly scaled (especially
with Adam's adaptive estimates starting from 0). A sudden large LR step can cause
instability. Warmup lets the optimizer "settle in" before applying full LR.

### 4.5 Cyclical Learning Rates (CLR)

```
η oscillates between η_min and η_max in a triangular or sinusoidal pattern.

Benefit: periodically increasing η can help escape sharp local minima/saddle
points, while the low-η phases allow fine convergence.
```

### 4.6 ReduceLROnPlateau

```
Monitor validation loss.
If no improvement for `patience` epochs:
  η ← η × factor   (e.g., factor=0.1, multiply by 10×reduction)

Adaptive: only decays when training stalls, rather than on a fixed schedule.
```

---

## 5. Data Augmentation

### 5.1 Purpose

Data augmentation artificially increases the diversity of the training set by
applying label-preserving transformations, acting as a powerful regularizer
without requiring additional labeled data.

```
Original training set: N samples
Effective training set: N × (number of augmentation variants) — but generated
                         on-the-fly, not stored, avoiding memory blowup.
```

### 5.2 Augmentation as Implicit Regularization

```
Training with augmentation ≈ training on the original loss PLUS an expectation
over a distribution of label-preserving transformations T:

L_aug(θ) = E_{T~𝒯} [ L(f(T(x); θ), y) ]

This is a smoothed version of the original loss — encourages f to be invariant
to the transformations in 𝒯, which is usually a desirable property
(e.g., a cat is still a cat if flipped horizontally).
```

### 5.3 Common Augmentations (Tabular / Generic)

For tabular data (used in this topic's experiments):
```
Gaussian noise injection:  x' = x + ε,   ε ~ N(0, σ²)
Feature dropout:           randomly zero out a subset of input features
Mixup:                     x' = λx_i + (1-λ)x_j,  y' = λy_i + (1-λ)y_j
SMOTE (for class imbalance): synthesize new minority-class samples via
                              interpolation between nearest neighbours
```

### 5.4 Common Augmentations (Image — Preview for Phase 2)

```
Geometric:    rotation, flipping, cropping, scaling, translation
Color:        brightness, contrast, saturation, hue jitter
Noise:        Gaussian noise, blur
Erasing:      Random Erasing, Cutout — randomly mask rectangular regions
Mixing:       Mixup, CutMix — combine two images and their labels
Advanced:     AutoAugment, RandAugment — learned/randomized augmentation policies
```

### 5.5 Mixup In Depth

```
Given two random samples (xᵢ, yᵢ) and (xⱼ, yⱼ):
  λ ~ Beta(α, α)              typically α=0.2 to 0.4
  x̃ = λxᵢ + (1−λ)xⱼ
  ỹ = λyᵢ + (1−λ)yⱼ

Train on (x̃, ỹ) instead of original samples.
```

**Why Beta distribution?** Beta(α,α) is symmetric and concentrates mass near
0 and 1 for small α (mostly near-original samples, occasionally heavily-mixed
ones) — providing a tunable balance between aggressive and mild mixing.

**Effect:** Encourages locally linear behavior between class regions; acts as
a strong regularizer; smooths decision boundaries.

### 5.6 When NOT to Augment

```
✗ When variation would change the label (e.g., flipping a "6" digit → looks like "9")
✗ When augmented samples become unrealistic / out-of-distribution
✗ For tasks with strict invariances that conflict with the augmentation
  (e.g., do not flip text images horizontally — text becomes unreadable/wrong)
```

---

## 6. Cross-Validation for Hyperparameter Selection

### 6.1 K-Fold Cross-Validation

```
1. Split training data into K folds (typically K=5 or K=10)
2. For each fold k:
     Train on the other K-1 folds
     Validate on fold k
3. Average validation performance across all K folds

CV_score(λ) = (1/K) Σₖ L_val(θ*_k(λ); fold k)
```

**Why CV instead of a single train/val split?**
A single split's validation score has high variance — by chance, the
validation set might be unusually easy or hard. K-Fold CV averages over K
different splits, giving a more robust estimate of how a hyperparameter setting
will generalize, at the cost of K× more compute.

### 6.2 Stratified K-Fold

For classification, **stratified** K-fold ensures each fold preserves the
overall class distribution — critical for imbalanced datasets, preventing
folds where, e.g., a particular class is entirely absent from training or
validation.

### 6.3 Nested Cross-Validation

```
Outer loop: K-fold split for FINAL performance estimate
Inner loop: K-fold split (within each outer training set) for hyperparameter
            selection

This avoids "leaking" test set information into hyperparameter choices —
critical when reporting a paper's final benchmark numbers.
```

---

## Key Equations Summary

| Concept | Formula |
|---|---|
| Bilevel HPO | λ* = argmin_λ L_val(θ*(λ)) |
| Step decay | η_t = η₀γ^⌊t/s⌋ |
| Cosine annealing | η_t = η_min + ½(η_max−η_min)(1+cos(πt/T)) |
| Mixup | x̃=λxᵢ+(1−λ)xⱼ, ỹ=λyᵢ+(1−λ)yⱼ |
| K-Fold CV score | (1/K)Σₖ L_val(θ*_k(λ)) |
| Expected Improvement | E[max(f(λ_best)−f(λ), 0)] |

