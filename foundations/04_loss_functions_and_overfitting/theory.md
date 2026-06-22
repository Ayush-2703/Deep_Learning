# Theory: Loss Functions, Overfitting & Bias-Variance Trade-off

**Deep Learning Mastery Repository**

---

## Table of Contents
1. [Loss Functions](#1-loss-functions)
2. [Regression Losses](#2-regression-losses)
3. [Classification Losses](#3-classification-losses)
4. [Advanced Losses](#4-advanced-losses)
5. [Choosing a Loss Function](#5-choosing-a-loss-function)
6. [Overfitting and Underfitting](#6-overfitting-and-underfitting)
7. [Bias-Variance Trade-off](#7-bias-variance-trade-off)
8. [Diagnosing with Learning Curves](#8-diagnosing-with-learning-curves)

---

## 1. Loss Functions

A loss function `ℓ(ŷ, y)` measures the discrepancy between the model's
prediction `ŷ` and the true label `y`. The training objective is:

```
L(θ) = (1/N) Σᵢ ℓ(f(xᵢ; θ), yᵢ)   +   λ·R(θ)
        ↑ empirical risk                  ↑ regularisation (Topic 5)
```

**Requirements for a good loss function:**
- Differentiable w.r.t. model outputs (for gradient-based optimisation)
- Appropriate for the output distribution (regression vs classification)
- Numerically stable (no log(0), no inf)
- Sensitive enough to drive learning — but not to outliers if robustness matters

---

## 2. Regression Losses

### 2.1 Mean Squared Error (MSE / L2 Loss)

```
MSE(ŷ, y) = (1/N) Σᵢ (yᵢ − ŷᵢ)²

Gradient:  ∂MSE/∂ŷᵢ = (2/N)(ŷᵢ − yᵢ)
```

**Properties:**
- Differentiable everywhere
- Penalises large errors quadratically → **sensitive to outliers**
- Optimal under Gaussian noise assumption (MLE of Gaussian likelihood)
- Unique global minimum for convex models

**When to use:** Clean data, Gaussian noise, outliers already removed.

### 2.2 Mean Absolute Error (MAE / L1 Loss)

```
MAE(ŷ, y) = (1/N) Σᵢ |yᵢ − ŷᵢ|

Gradient:  ∂MAE/∂ŷᵢ = (1/N) · sign(ŷᵢ − yᵢ)   (undefined at 0)
```

**Properties:**
- Not differentiable at yᵢ = ŷᵢ (use subgradient or smooth approximation)
- **Robust to outliers** (linear penalty, not quadratic)
- Optimal under Laplacian noise assumption
- Gradient is constant magnitude — can oscillate near minimum

**When to use:** Data with outliers; median regression; robust estimation.

### 2.3 Huber Loss (Smooth L1)

```
          ⎧ ½(yᵢ−ŷᵢ)²              if |yᵢ−ŷᵢ| ≤ δ   (quadratic zone)
Hδ(ŷ,y) = ⎨
          ⎩ δ|yᵢ−ŷᵢ| − ½δ²        if |yᵢ−ŷᵢ| > δ   (linear zone)

Gradient:  ∂Hδ/∂ŷᵢ = { (ŷᵢ−yᵢ)          if |error| ≤ δ
                       { δ·sign(ŷᵢ−yᵢ)   if |error| > δ
```

**Properties:**
- **Combines MSE (near 0) and MAE (far from 0)**
- Differentiable everywhere — no subgradient needed
- δ is a hyperparameter: larger δ → more MSE-like; smaller → more MAE-like
- PyTorch: `nn.SmoothL1Loss()` uses δ=1 by default

**When to use:** Object detection (bounding box regression), regression with some outliers.

### 2.4 Mean Absolute Percentage Error (MAPE)

```
MAPE(ŷ, y) = (100/N) Σᵢ |yᵢ − ŷᵢ| / |yᵢ|
```

Undefined when `yᵢ = 0`. Use SMAPE or relative RMSE instead for zero-containing targets.

---

## 3. Classification Losses

### 3.1 Binary Cross-Entropy (BCE / Log Loss)

```
BCE(ŷ, y) = -(1/N) Σᵢ [yᵢ log ŷᵢ + (1−yᵢ) log(1−ŷᵢ)]

Gradient:  ∂BCE/∂ŷᵢ = -(yᵢ/ŷᵢ) + (1−yᵢ)/(1−ŷᵢ)
```

**Properties:**
- Derived from maximum likelihood estimation of Bernoulli distribution
- Output `ŷᵢ ∈ (0,1)` → apply sigmoid before BCE
- Gradient `∂BCE/∂zᵢ = ŷᵢ − yᵢ` (after composing with sigmoid)
- PyTorch: `nn.BCELoss()` expects sigmoid-activated input,
          `nn.BCEWithLogitsLoss()` = sigmoid + BCE (numerically stable, preferred)

**Why BCE instead of MSE for classification?**
```
MSE with sigmoid activation:
  Gradient: ∂MSE/∂zᵢ = (ŷᵢ−yᵢ)·σ'(zᵢ) = (ŷᵢ−yᵢ)·ŷᵢ(1−ŷᵢ)

When ŷᵢ ≈ 1 (confident wrong prediction): ŷᵢ(1−ŷᵢ) ≈ 0 → vanishing gradient!
BCE avoids this: gradient = ŷᵢ−yᵢ (no saturation term)
```

### 3.2 Categorical Cross-Entropy

```
CE(ŷ, y) = -(1/N) Σᵢ Σₖ yᵢₖ log ŷᵢₖ

For one-hot labels: CE(ŷ, y) = -(1/N) Σᵢ log ŷᵢ,yᵢ
                                            ↑ probability assigned to correct class
```

**Properties:**
- Generalises BCE to K classes
- Output `ŷ ∈ ΔK` (probability simplex) → apply softmax before CE
- PyTorch: `nn.CrossEntropyLoss()` = LogSoftmax + NLLLoss (numerically stable)
  DO NOT use `nn.Softmax + nn.NLLLoss` manually

**Label smoothing variant:**
```
y_smooth = (1 − ε)·y_one_hot + ε/K

Instead of 0s and 1s, targets are ε/K and (1−ε+ε/K)
Prevents overconfident predictions; ε = 0.1 is common (used in Vision Transformer)
```

### 3.3 Kullback-Leibler Divergence (KL Loss)

```
KL(P‖Q) = Σₓ P(x) log[P(x)/Q(x)]
         = Σₓ P(x) log P(x) − Σₓ P(x) log Q(x)
         = −H(P) + CE(P, Q)

Properties:
  KL ≥ 0   (Gibbs' inequality)
  KL = 0   iff P = Q
  NOT symmetric: KL(P‖Q) ≠ KL(Q‖P)
```

**When to use:** Variational Autoencoders (VAE), knowledge distillation,
distribution matching.

---

## 4. Advanced Losses

### 4.1 Focal Loss (Lin et al., 2017)

```
FL(ŷ, y) = −(1−ŷ)^γ · y·log(ŷ) − ŷ^γ·(1−y)·log(1−ŷ)

γ ≥ 0 is the focusing parameter.
γ = 0 → standard BCE.
γ = 2 → standard choice for object detection.
```

**Motivation:** Class imbalance problem.

In object detection, 99%+ of image regions are background (easy negatives).
Standard CE is dominated by these easy examples despite them contributing
little useful gradient. Focal loss down-weights easy examples:

```
When ŷ ≈ y (easy, confident correct prediction): (1−ŷ)^γ ≈ 0 → small loss
When ŷ ≈ 1-y (hard, wrong prediction):           (1−ŷ)^γ ≈ 1 → full loss
```

**Where used:** RetinaNet, YOLO variants, any extreme class imbalance.

### 4.2 Contrastive / Triplet Loss (Metric Learning)

```
Contrastive: L = y·d² + (1−y)·max(0, m−d)²
             d = ‖f(x₁) − f(x₂)‖₂

Triplet:     L = max(0, ‖f(a)−f(p)‖² − ‖f(a)−f(n)‖² + m)
             a=anchor, p=positive, n=negative, m=margin
```

**Where used:** Face recognition, image similarity, few-shot learning.

---

## 5. Choosing a Loss Function

```
Task                    → Loss Function                   → PyTorch
────────────────────────────────────────────────────────────────────────
Binary classification   → BCE with logits                 nn.BCEWithLogitsLoss()
Multi-class (hard)      → Categorical CE                  nn.CrossEntropyLoss()
Multi-class (soft)      → Label-smoothed CE               nn.CrossEntropyLoss(label_smoothing=0.1)
Multi-label             → BCE per label                   nn.BCEWithLogitsLoss()
Regression (clean)      → MSE                             nn.MSELoss()
Regression (outliers)   → Huber / MAE                     nn.SmoothL1Loss() / nn.L1Loss()
Imbalanced detection    → Focal loss                      (custom or torchvision)
Distribution match      → KL divergence                   nn.KLDivLoss()
Metric learning         → Triplet / Contrastive           nn.TripletMarginLoss()
```

---

## 6. Overfitting and Underfitting

### 6.1 Definitions

```
                    Training Loss    Validation Loss
Underfitting:           HIGH             HIGH        ← model too simple
Good fit:               LOW              LOW         ← ideal
Overfitting:            LOW              HIGH        ← model memorised training data
```

**Overfitting:** The model learns noise and idiosyncrasies of the training set
rather than the true underlying function. Performance degrades on unseen data.

**Underfitting:** The model lacks capacity or is poorly trained. Cannot capture
the true patterns even in the training set.

### 6.2 Root Causes

**Overfitting causes:**
- Model too complex (too many parameters relative to data)
- Training too long (early epochs = learning signal, late epochs = memorising noise)
- Insufficient data
- No regularisation

**Underfitting causes:**
- Model too simple (linear model for non-linear data)
- Learning rate too low or too high (poor optimisation)
- Too few epochs
- Features not informative

### 6.3 Model Complexity vs Generalisation

```
Training error:
  ↓ monotonically as model complexity increases (more params → fits training better)

Validation error:
  ↓ first (model gains capacity to capture patterns)
  ↑ eventually (overfitting: learns training noise)

                Low Complexity              Optimal              High Complexity
                    ↓                          ↓                       ↓
Error:           ─────────────────────────────⊙──────────────────────────────
Train error:    ─────────────────────────────⊙──────────────────────────────/
Val error:       \───────────────────────────⊙──────────────────────────────/
                  Underfitting zone          Sweet spot          Overfitting zone
```

---

## 7. Bias-Variance Trade-off

### 7.1 Mathematical Decomposition

For a regression problem, the expected prediction error can be decomposed as:

```
E[(y − ŷ)²] = Bias²(ŷ) + Variance(ŷ) + σ²_noise

where:
  Bias(ŷ)     = E[ŷ] − f(x)            (systematic error: model wrong on average)
  Variance(ŷ) = E[(ŷ − E[ŷ])²]         (sensitivity to training data fluctuations)
  σ²_noise    = irreducible noise in y  (inherent in the data — cannot be reduced)
```

### 7.2 Intuition

```
HIGH BIAS (Underfitting):
  - Model makes consistent errors in a particular direction
  - Predictions are stable but wrong
  - Example: Linear model on quadratic data
  - Fix: Increase model capacity, add features

HIGH VARIANCE (Overfitting):
  - Model predictions change drastically with small changes in training data
  - Predictions are unstable — different training sets give very different models
  - Example: 100-layer MLP on 100 samples
  - Fix: Regularisation, more data, simpler model

THE TRADE-OFF:
  Decreasing bias → increasing variance (more complex model → less stable)
  Decreasing variance → increasing bias (simpler model → more systematic error)
  Optimal point: minimise Bias² + Variance (not each individually)
```

### 7.3 Double-Descent Phenomenon

Modern deep learning violates the classical U-shaped bias-variance curve:

```
Classical view:
  ↑ Error
  │         *
  │      *     *
  │   *            *
  │ *                 *
  └──────────────────────→ Model complexity
         Classical optimal

Modern view (double descent):
  ↑ Error         Interpolation
  │         *     threshold
  │      *     *  │
  │   *            *    *
  │ *                  *  *  * ← overparameterised regime
  └──────────────────────────→ Model complexity
         A second descent occurs in the overparameterised regime!
```

Very large models (e.g., GPT-4) are massively overparameterised yet generalise
well. This is an active research area — explained partially by implicit
regularisation of SGD and the benign overfitting theory.

---

## 8. Diagnosing with Learning Curves

### 8.1 Training / Validation Loss Plot

```
Scenario 1 — Underfitting:
  Train loss: high and plateaus early
  Val loss:   high (similar to train)
  Gap:        small

Scenario 2 — Overfitting:
  Train loss: decreases smoothly to near 0
  Val loss:   decreases then INCREASES (diverges from train)
  Gap:        LARGE

Scenario 3 — Good fit:
  Train loss: decreases smoothly
  Val loss:   decreases and tracks train loss closely
  Gap:        small and stable
```

### 8.2 Training Set Size Curves

Plot error vs number of training samples N:

```
High bias (underfitting): Both train and val errors are high for all N
  Adding more data doesn't help — model needs more capacity

High variance (overfitting): Train error is low, val error is high; gap decreases as N grows
  Adding more data helps — model benefits from seeing more examples
```

### 8.3 Practical Diagnostic Checklist

```
Step 1: Check training loss
  → If it doesn't decrease: LR too small, vanishing gradient, or bug

Step 2: Compare train vs val loss
  → Large gap → overfitting → add regularisation (Topic 5)
  → Both high  → underfitting → increase capacity or train longer

Step 3: Check loss curve shape
  → Oscillating → LR too high
  → Plateauing early → LR too low or stuck in saddle point

Step 4: Consider epoch of best validation performance
  → Save model at best val loss (early stopping — Topic 5)
```

---

## Key Equations Summary

| Loss | Formula | Use case |
|---|---|---|
| MSE | (1/N)Σ(y−ŷ)² | Regression, clean data |
| MAE | (1/N)Σ\|y−ŷ\| | Robust regression |
| Huber | quadratic+linear hybrid | Regression + outliers |
| BCE | −[y log ŷ+(1−y)log(1−ŷ)] | Binary classification |
| CE | −Σₖ yₖ log ŷₖ | Multi-class |
| KL | Σ P log(P/Q) | Distribution matching |
| Focal | −(1−ŷ)^γ y log ŷ | Imbalanced detection |
| Bias² + Var | E[(y−ŷ)²] − σ²_noise | Error decomposition |
