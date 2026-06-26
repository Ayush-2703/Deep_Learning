# Code Explanation: Hyperparameter Tuning & Data Augmentation

**Phase 1 — Topic 6 | `implementation.py` walkthrough**

---

## 1. Section A — Grid Search vs Random Search

### Building the Grid via `itertools.product`

```python
for lr, h in itertools.product(lr_grid, hidden_grid):
```

**Why `itertools.product` instead of nested for-loops?**
`itertools.product(A, B)` generates the Cartesian product lazily (one tuple at
a time), which is more memory-efficient and more readable than:
```python
for lr in lr_grid:
    for h in hidden_grid:
        ...
```
Both are equivalent in this case, but `product` scales cleanly to N hyperparameters
without nesting N for-loops.

### Random Search: Log-Uniform Sampling

```python
lr = float(10 ** rng.uniform(-4, -2))
```

**Why sample in log-space rather than linear space?**
If we sampled `lr ~ Uniform(1e-4, 1e-2)` directly, the vast majority of samples
would land in `[5e-3, 1e-2]` (linear uniform spreads mass evenly across the
*absolute* range, which is dominated by the larger end). But learning rate's
*effect* on training is roughly multiplicative — `1e-4` vs `1e-3` is as
significant a change as `1e-3` vs `1e-2`. Sampling the *exponent* uniformly
(`rng.uniform(-4, -2)`) and then exponentiating gives **log-uniform** sampling,
which spreads search effort evenly across orders of magnitude — matching how
learning rate actually affects optimization.

### Why Random Search Used a Wider Hidden-Size Choice

```python
h = int(rng.choice([8, 16, 32, 64, 128, 256]))   # 6 choices vs grid's 3
```

This intentionally demonstrates one of random search's practical advantages:
since it doesn't need a fixed grid, we can search over a denser/wider candidate
set at the same total budget (9 evaluations), giving finer-grained coverage of
the hidden-size axis. The grid search's `hidden_grid` only has 3 values, so it can
never discover that `hidden=64` or `hidden=16` might be a strong setting.

### Live Result Interpretation

```
Best Grid:   lr=1e-02, hidden=8,  acc=96.9%
Best Random: lr=8.94e-03, hidden=8, acc=96.9%
```

Both methods found essentially the same optimal region (lr near 1e-2, small
hidden size) and tied at the best accuracy. This is expected for a simple 2D
problem with few important hyperparameters — random search's advantage compounds
in higher-dimensional search spaces (e.g., 5+ hyperparameters), which is not
fully visible in this 2-hyperparameter toy example, but the coverage plot (Fig 1,
Panel 1) still shows random search explores 9 unique `lr` values versus the
grid's mere 3.

---

## 2. Section B — Learning Rate Schedules

### Scheduler Construction Pattern

```python
if sched_type == "step":
    sched = StepLR(opt, step_size=30, gamma=0.3)
elif sched_type == "cosine":
    sched = CosineAnnealingLR(opt, T_max=n_epochs)
elif sched_type == "plateau":
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
```

**Why does `ReduceLROnPlateau` need different `.step()` call syntax?**
```python
if sched_type == "plateau":
    sched.step(epoch_loss)     # ← requires the monitored metric
elif sched is not None:
    sched.step()                # ← no argument needed
```
`StepLR` and `CosineAnnealingLR` are purely time-based — they only need to know
"what epoch are we on," which they track internally via their own `.step()` call
counter. `ReduceLROnPlateau` is metric-based — it needs the actual validation
(or training) loss value passed explicitly each call, since its decision to
reduce LR depends on whether that metric improved, not on elapsed epochs.

### Custom Warmup+Cosine via `LambdaLR`

```python
def warmup_cosine(epoch, warmup=10, total=n_epochs, base_lr=1e-2):
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1 + np.cos(np.pi * progress))

sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=warmup_cosine)
```

**Why does `warmup_cosine` return a multiplier, not an absolute LR?**
`LambdaLR` works by **multiplying** the base LR (set when constructing the
optimizer) by whatever the lambda function returns at each epoch:
```
actual_lr_at_epoch_t = base_lr × lr_lambda(t)
```
During warmup (`epoch < warmup`), the multiplier ramps linearly from
`1/warmup` to `1.0` — so the LR ramps from near-0 up to the full base LR.
After warmup, the multiplier follows a cosine decay from 1.0 down to 0.0 —
the LR decays smoothly to (near) zero by the end of training. This pattern is
exactly the warmup+cosine schedule described in theory.md §4.4, used by
virtually all modern Transformer training recipes.

**Why `(epoch + 1) / warmup` and not `epoch / warmup`?**
At `epoch=0`, `epoch/warmup = 0` would give a multiplier of exactly 0 — meaning
the very first epoch trains with LR=0 (no learning at all, wasted epoch).
Using `(epoch+1)/warmup` ensures the first epoch already has a small positive
LR (`1/warmup`), avoiding the wasted first step.

### Live Result Interpretation

```
Constant lr=1e-2     | final_loss=0.1378 | final_acc=95.0%
Step Decay           | final_loss=0.1540 | final_acc=96.9%   ← highest accuracy
ReduceLROnPlateau    | final_loss=0.1378 | final_acc=95.0%   ← identical to constant!
```

**Why did `ReduceLROnPlateau` behave identically to the constant schedule here?**
With `patience=8` and a well-behaved, monotonically-decreasing loss curve on this
simple problem, the loss likely never plateaus for 8 consecutive epochs without
improvement — so the scheduler simply never triggers a reduction, and the LR
stays at its initial value the whole time. This is a realistic outcome: adaptive
schedulers only intervene when their trigger condition is actually met, and for
an "easy" problem like 2D make_moons, a well-chosen constant LR may already be
near-optimal, leaving no plateau to detect.

---

## 3. Section C — Data Augmentation

### Why Feature Dropout ≠ `nn.Dropout`

```python
def feature_dropout_augment(X: torch.Tensor, p: float = 0.2) -> torch.Tensor:
    mask = (torch.rand_like(X) > p).float()
    return X * mask
```

This is conceptually similar to `nn.Dropout`, but applied to the **raw input
features** (a data augmentation), not to hidden activations (a regularization
layer). It's implemented manually here (rather than via `nn.Dropout(p)` on the
input) to make explicit that this is a *data-level* transformation applied
before the model sees the batch, distinct from architecture-level dropout
covered in Topic 5. Note also there's no `1/(1-p)` rescaling here — unlike
`nn.Dropout`, which rescales to preserve expected activation magnitude,
simple feature-level augmentation often omits this since we're not trying
to approximate an ensemble, just adding input noise/missingness.

### Mixup Implementation

```python
def mixup(X, y, alpha=0.4):
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(X.size(0))
    X_mixed = lam * X + (1 - lam) * X[perm]
    y_mixed = lam * y + (1 - lam) * y[perm]
    return X_mixed, y_mixed
```

**Why `torch.randperm(X.size(0))` instead of randomly sampling a second batch?**
`randperm` shuffles the *existing* batch indices, pairing each sample with
another random sample from the *same* batch. This avoids needing a second
DataLoader iterator and guarantees every original sample participates in
exactly one mixup pair per batch — efficient and simple.

**Why is `lam` sampled once per batch, not once per sample?**
```python
lam = float(np.random.beta(alpha, alpha))   # single scalar, applied to whole batch
```
The original Mixup paper (Zhang et al., 2018) samples one λ per batch for
implementation simplicity and to keep the mixing ratio interpretable across
the whole batch in one training step. Per-sample λ values would also work and
are used in some variants, but introduce more implementation complexity for a
relatively small accuracy gain.

**Why does `y_mixed` become a soft (non-binary) label?**
```python
y_mixed = lam * y + (1 - lam) * y[perm]
```
If `y=1` and `y[perm]=0` and `lam=0.7`, then `y_mixed = 0.7`. This is no longer
a hard 0/1 label — it's a **soft target** representing "70% confidence this is
class 1." `nn.BCELoss` accepts soft targets in `[0,1]` natively (it's
mathematically just cross-entropy against a Bernoulli parameter, which doesn't
require the target to be exactly 0 or 1), so no special loss modification
is needed.

### Why the Training Set Was Deliberately Shrunk

```python
Xtr, Xva, ytr, yva = train_test_split(X_full, y_full, test_size=0.7, ...)
print(f"Deliberately small train set: {len(Xtr_s)} samples (val set: {len(Xva_s)})")
```

**Why use `test_size=0.7` (keeping only 30% for training)?**
With the full ~640 training samples used in other sections, this simple 2D
problem doesn't overfit much even without augmentation (as seen in Topic 4's
experiments). To meaningfully demonstrate augmentation's regularizing effect,
we need a regime where overfitting is more likely — hence shrinking the
training set to only 240 samples while keeping a large 560-sample validation
set for a reliable performance estimate.

### Live Result — An Important Negative Finding

```
No Augmentation    | val_acc=95.4% | gap=-0.0139   ← already NOT overfitting!
Gaussian Noise     | val_acc=94.6% | gap=-0.0556
Feature Dropout    | val_acc=94.6% | gap=-0.1234
Mixup              | val_acc=94.6% | gap=-0.1679
```

**This result is genuinely instructive: augmentation did NOT improve validation
accuracy here, and the baseline gap was already negative (no overfitting to
fix).** Why?

The `make_moons` decision boundary is a smooth, low-dimensional curve. Even with
only 240 samples, a 128-128 MLP can learn this boundary well without memorizing
noise — the *problem itself* doesn't have enough complexity to trigger severe
overfitting at this model scale, even though the model is technically
over-parameterized (~17K params for 240 samples). All three augmentations
increased *train* loss substantially (harder optimization objective) without a
corresponding *validation* improvement, because there was no overfitting gap to
close in the first place.

**Lesson for practitioners:** Data augmentation is not a universal accuracy
booster — it helps specifically when the training set is the bottleneck (high
variance / overfitting regime). Applying it blindly to a problem that isn't
overfitting can make optimization harder for no generalization benefit. Always
verify there IS a train-val gap before reaching for augmentation as the fix.

---

## 4. Section D — K-Fold Cross-Validation

### Re-fitting the Scaler Per Fold

```python
for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_full, y_full)):
    Xtr_f, Xva_f = X_full[tr_idx], X_full[va_idx]
    sc = StandardScaler().fit(Xtr_f)        # ← NEW scaler each fold
    Xtr_f = sc.transform(Xtr_f); Xva_f = sc.transform(Xva_f)
```

**Why fit a brand-new `StandardScaler` inside the fold loop, rather than once
globally?**
Each fold has a different training subset (4/5 of the data, rotating). Fitting
the scaler once on the *entire* dataset before splitting would leak validation-fold
statistics into the normalization used for training — the same data leakage
principle from Topic 1, just applied K times here. Each fold must independently
fit its own scaler on only its own training portion.

### Why Vary the Random Seed Per Fold

```python
torch.manual_seed(SEED + fold_idx)
```

**Why not use the same `SEED` for every fold?**
If every fold used identical weight initialization, any variation in the
resulting fold accuracy would purely reflect data differences between folds —
which is actually what we want to measure! However, using `SEED + fold_idx`
here additionally ensures we're not accidentally creating a spurious correlation
between a specific weight initialization and a specific fold's data (a
pathological edge case), giving a cleaner per-fold independent estimate. This
is a minor robustness choice; using a fixed seed across folds would also be
defensible for tightly controlled comparisons.

### Live Result Interpretation

```
lr=1e-03, hidden=16  | mean=94.75% ± 1.88%
lr=1e-03, hidden=64  | mean=95.00% ± 1.43%   ← best mean AND lowest variance
lr=1e-02, hidden=64  | mean=94.00% ± 2.64%   ← highest variance (less reliable)
```

The `lr=1e-3, hidden=64` configuration wins on **both** criteria that matter for
robust model selection: highest mean accuracy AND lowest standard deviation
across folds. The `lr=1e-2` configuration, despite performing well in the
single-split search of Section A (`96.9%` there), shows notably higher variance
here (`±2.64%`) — a signal that its Section-A result may have been partly a
fortunate split rather than a robustly better hyperparameter. This illustrates
exactly why K-Fold CV is preferred over a single train/val split for final
hyperparameter selection: it surfaces configurations whose apparent performance
is unstable across different data partitions.

---

## Pitfalls Avoided

| Pitfall | Fix |
|---|---|
| Linear-uniform LR sampling biases toward large values | Log-uniform sampling via exponent |
| First warmup epoch trains with LR=0 | `(epoch+1)/warmup` instead of `epoch/warmup` |
| `ReduceLROnPlateau` called with wrong signature | Branch `.step(metric)` vs `.step()` |
| Scaler fit on all data before K-Fold split (leakage) | New `StandardScaler` fit per fold |
| Augmentation evaluated where no overfitting exists | Deliberately shrunk train set in Section C |
| Single train/val split misleads on hyperparameter robustness | 5-Fold CV with mean ± std reporting |

*Previous: [Topic 5 — Regularization & Optimizers](../05-regularization-optimizers-batchnorm/explanation.md)*
*Next: [Topic 7 — Linear Algebra & PyTorch Tensors](../07-extra-linear-algebra-pytorch-tensors/explanation.md)*
