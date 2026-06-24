# Code Explanation: Loss Functions, Overfitting & Bias-Variance

**Phase 1 — Topic 4 | `implementation.py` walkthrough**

---

## 1. Section A — Loss Function Implementations

### Huber Loss Branching

```python
def forward(self, pred, target):
    e   = torch.abs(pred - target)
    qdr = 0.5 * e ** 2
    lin = self.delta * e - 0.5 * self.delta ** 2
    return torch.mean(torch.where(e <= self.delta, qdr, lin))
```

**Why compute both `qdr` and `lin` for every element?**
`torch.where` is vectorized — it evaluates both branches for the entire tensor,
then selects element-wise based on the condition. This is faster on GPU than a
Python-level conditional because it avoids branching divergence and uses a single
fused kernel. The cost is computing both branches even where only one is used,
but for elementwise ops this is cheaper than control flow.

**Why does `lin` use `−0.5δ²` as an offset?**
This ensures continuity at `e = δ`: `qdr(δ) = 0.5δ²` and `lin(δ) = δ·δ − 0.5δ² = 0.5δ²`.
Without this offset, the loss would have a discontinuous jump at the transition point,
creating a non-smooth (and incorrect) loss surface.

### Focal Loss Implementation

```python
def forward(self, pred, target):
    p   = torch.sigmoid(pred)              # pred = logits
    bce = -(target * torch.log(p + self.eps) + (1-target)*torch.log(1-p+self.eps))
    focal_weight = (1-p)**self.gamma * target + p**self.gamma * (1-target)
    return torch.mean(focal_weight * bce)
```

**Why does `pred` represent logits, not probabilities?**
Computing sigmoid internally (rather than expecting pre-sigmoided input) lets us
control the `eps` clamp precisely and matches the convention of
`BCEWithLogitsLoss` — the numerically preferred pattern in PyTorch.

**Why two terms in `focal_weight`?**
The formula needs different down-weighting depending on the true class:
- When `target=1`: weight = `(1-p)^γ` — penalize less when p is already high (easy positive)
- When `target=0`: weight = `p^γ` — penalize less when p is already low (easy negative)
Combining via `target * term1 + (1-target) * term2` selects the correct branch
per-sample without a Python `if`, keeping the operation vectorized.

---

## 2. Section B — Overfitting Demonstration

### Why 5 Architectures Spanning 4 Orders of Magnitude

```python
architectures = {
    "Tiny  [2]":          [2],            # 9 params
    "Small [16,16]":      [16, 16],       # 337 params
    "Medium [64,64]":     [64, 64],       # 4,417 params
    "Large [256,256,256]":[256, 256, 256],# 132,609 params
    "Huge  [512]×5":      [512]*5,        # 1,052,673 params
}
```

The make_moons dataset has only 640 training samples. Sweeping from 9 to over
1 million parameters (more parameters than training samples by 1600×) creates
the classic "more capacity than data" regime where overfitting becomes visible.

**Why is the "Huge" model's gap negative in the live run?**
```
Huge  [512]×5  | Train L: 0.1872 | Val L: 0.1529 | Gap: -0.0343
```
This looks counter-intuitive — shouldn't massive overcapacity overfit?
With only 150 epochs and Adam's adaptive step sizes, a 1M-parameter network on
640 samples hasn't fully converged on the training set yet (train loss is
actually higher than Medium's). This illustrates an important nuance: overfitting
requires BOTH capacity AND sufficient training time/iterations to memorize noise.
Given enough additional epochs, this gap would likely flip positive.

### Why BCELoss (Not MSE) Drives This Experiment

Using BCE keeps every architecture directly comparable on the same loss scale
used for classification accuracy — MSE on probabilities would conflate
regression-style penalty with the classification task.

---

## 3. Section C — Bias-Variance Decomposition

### Bootstrap Resampling

```python
idx  = rng.integers(0, len(X_tr), size=len(X_tr))   # bootstrap sample
Xb, Yb = X_tr[idx], y_tr[idx]
```

**Why bootstrap (sample with replacement) instead of just reusing X_tr directly?**
The bias-variance decomposition formally requires sampling many **different**
training sets from the same underlying distribution. Since we only have one
finite dataset, bootstrap resampling approximates drawing new training sets:
each bootstrap sample has the same size as `X_tr` but with random repeats and
omissions (≈63% unique samples per bootstrap, by the well-known bootstrap formula
`1−1/e≈0.632`). This variability across `n_trials` bootstrap samples is what lets
us measure how much the model's predictions change due to training-data noise — exactly variance.

### The Decomposition Formula in Code

```python
mean_pred = all_preds.mean(axis=0)          # E[ŷ(x)] over trials, per test point
bias_sq   = np.mean((mean_pred - y_true)**2)   # (E[ŷ]-y)² averaged over test set
variance  = np.mean(all_preds.var(axis=0))     # Var(ŷ) averaged over test set
```

**Why `axis=0`?**
`all_preds` has shape `(n_trials, n_test)`. We want, for EACH test point,
the average prediction **across trials** (axis=0), then average that error
**across test points**. `.var(axis=0)` similarly computes variance across
trials for each test point, then we average across test points.

**Live result interpretation:**
```
Low-complexity [4]:    Bias²=0.0982  Variance=0.0036
High-complexity [128,128]: Bias²=0.0483  Variance=0.0059
```
The low-complexity model has 2× higher bias (systematically wrong — can't capture
the moon-shaped boundary well) but lower variance (consistent across bootstraps,
since it has so little capacity it converges to nearly the same simple function
every time). The high-complexity model has lower bias (can fit the true boundary)
at the cost of higher variance (different bootstraps yield slightly different
boundaries). The trade-off is visible numerically, exactly as theory predicts.

---

## 4. Section D — Learning Curves

### Random Subsampling Without Replacement

```python
idx  = np.random.default_rng(SEED).choice(len(X_tr), size=sz, replace=False)
```

**Why `replace=False` here but `replace=True` (bootstrap) in Section C?**
Different goals: Section C wants statistical resampling variability (bootstrap).
Section D wants a clean, non-redundant subset of increasing size to measure how
performance scales with **more unique information**, not resampling noise.
`replace=False` ensures no duplicate samples within each subset.

**Why re-seed with the same SEED every iteration?**
```python
idx = np.random.default_rng(SEED).choice(len(X_tr), size=sz, replace=False)
```
Using the same seed for every subset size means smaller subsets are *prefixes*
of larger ones in expectation (same RNG state → same initial draws), making the
learning curve progression more interpretable (we're adding data, not swapping it
out entirely).

### Batch Size Adaptation for Small Subsets

```python
ld = DataLoader(TensorDataset(Xsub, Ysub), batch_size=min(32, sz), shuffle=True)
```

**Why `min(32, sz)`?**
For `sz=30` (the smallest training size tested), a fixed `batch_size=32` would
either error (if DataLoader requires full batches) or silently use a single
batch of 30. Explicitly capping batch size avoids inconsistent behavior across
different training set sizes, ensuring all subsets get proper mini-batch training.

---

## 5. Visualization Notes

### Stacked Bar for Bias-Variance

```python
ax4.bar(x_bv, bias_vals, 0.35, color="#e74c3c", label="Bias²")
ax4.bar(x_bv, var_vals,  0.35, color="#3498db", label="Variance", bottom=bias_vals)
```

`bottom=bias_vals` stacks the variance bar on top of the bias bar, so the total
bar height visually represents `Bias² + Variance ≈ Total MSE` — directly
illustrating the additive decomposition from the theory.

### Fill Between for Generalization Gap

```python
ax.fill_between(data["sizes"], data["train"], data["val"], alpha=0.12, color="#e74c3c")
```

Shading the region between train and validation curves makes the generalization
gap immediately visible as a colored area rather than requiring the reader to
mentally compute the vertical distance between two lines at each point.

---

## Pitfalls Avoided

| Pitfall | Fix |
|---|---|
| f-string syntax error from nested brackets | Extract value to variable before formatting |
| Comparing models by epoch instead of fair budget | All architectures trained for same 150 epochs |
| Single train/test split hides variance | Bootstrap resampling (20 trials) for bias-variance |
| Inconsistent batch sizes across data sizes | `batch_size=min(32, sz)` |
| Mixing loss scales (BCE vs MSE) | BCE used uniformly for all classification comparisons |

*Previous: [Topic 3 — Gradient Descent & Backprop](../03-gradient-descent-backprop/explanation.md)*
*Next: [Topic 5 — Regularization & Optimizers](../05-regularization-optimizers-batchnorm/explanation.md)*
