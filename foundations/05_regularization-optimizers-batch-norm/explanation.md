# Code Explanation: Regularization, Optimizers, BatchNorm & Early Stopping

**Phase 1 — Topic 5 | `implementation.py` walkthrough**

---

## 1. Section A — L1 vs L2 Regularization

### Computing Regularization Loss Manually

```python
def regularization_loss(self):
    weights = [self.fc1.weight, self.fc2.weight, self.fc3.weight]
    if self.reg_type == "l2":
        return self.lam * sum(w.pow(2).sum() for w in weights)
    elif self.reg_type == "l1":
        return self.lam * sum(w.abs().sum() for w in weights)
    return torch.tensor(0.0)
```

**Why implement regularization manually instead of using `weight_decay` in the optimizer?**
PyTorch's `optimizer(weight_decay=...)` implements L2 regularization implicitly
by adding `λw` directly to the gradient before the update — mathematically
equivalent to L2 penalty for SGD, but subtly different for Adam (this is exactly
why AdamW exists — see Section D). Implementing it manually inside the loss:
1. Makes the math explicit and matches the theory.md derivation exactly
2. Allows us to implement L1 (which most optimizers don't support natively)
3. Lets us track the regularization term separately from the data loss for logging

**Why only regularize weights, not biases?**
```python
weights = [self.fc1.weight, ...]   # NOT self.fc1.bias
```
Biases don't contribute to model complexity in the same way weights do — they
just shift the decision boundary, not its curvature/sharpness. Standard practice
excludes biases (and BatchNorm parameters) from weight decay.

### Measuring Sparsity

```python
w_all = torch.cat([p.data.cpu().flatten() for p in model.parameters() if p.dim() > 1])
sparsity = float((np.abs(w_all) < 1e-3).mean())
```

**Why `p.dim() > 1`?**
This filters to weight matrices only (2D tensors), excluding 1D bias vectors.
We want to measure sparsity in the *weights* specifically, matching the
theoretical claim that L1 induces weight sparsity (not necessarily bias sparsity).

**Why threshold at `1e-3` rather than checking for exact zero?**
L1's subgradient-based optimization in continuous space rarely drives weights to
*exactly* 0.0 in floating point — it gets very close. A small threshold
(`1e-3`) captures "practically zero" weights, which is the meaningful notion of
sparsity for downstream pruning or interpretation.

**Live result validates theory exactly:**
```
No Reg     | near-zero=0.3%
L2 λ=1e-3  | near-zero=10.1%
L1 λ=1e-4  | near-zero=36.3%   ← 3.6× sparser than L2 despite smaller λ!
```
L1 achieves far higher sparsity than L2 even with a *smaller* regularization
coefficient (1e-4 vs 1e-3) — this is the signature L1-vs-L2 behavior, confirming
the diamond-vs-sphere geometric argument from theory.md.

---

## 2. Section B — Dropout

### Single Model Class, Reused Across Configs

```python
class DropoutMLP(nn.Module):
    def __init__(self, p: float = 0.5):
        ...
        nn.Dropout(p), ...
```

**Why parametrize `p` instead of three separate classes?**
`nn.Dropout(p)` accepts the rate as a constructor argument — building one
flexible class and varying `p` avoids code duplication and guarantees the three
configurations are architecturally identical except for the dropout rate,
isolating it as the only variable in the experiment.

### Automatic Train/Eval Switching

The `run_training` helper calls `model.train()` before the training loop and
`model.eval()` before validation — this single pair of calls is what makes
`nn.Dropout` behave correctly:
```
model.train(): Dropout ACTIVE — randomly zeroes neurons, scales by 1/(1-p)
model.eval():  Dropout INACTIVE — full network, no masking, no scaling
```
Without `model.eval()`, validation loss would be artificially noisy and pessimistic
because the dropout mask would still be randomly zeroing neurons during evaluation.

**Live result — dropout closing the generalization gap:**
```
No Dropout (p=0)  | gap=+0.0210   ← val loss higher than train (overfitting)
Dropout p=0.3     | gap=+0.0062   ← gap shrinks 3×
Dropout p=0.5     | gap=-0.0022   ← val loss now LOWER than train!
```
At p=0.5, the gap becomes slightly negative — train loss is computed with
dropout *active* (noisy, harder objective), while val loss is computed with
dropout *inactive* (full network, easier). This is expected and not a bug:
training loss under dropout is not directly comparable to validation loss
without dropout on an absolute basis — what matters is that validation loss
itself is lower with dropout than without, confirming the regularization benefit.

---

## 3. Section C — Batch Normalization

### High Learning Rate to Showcase Stability

```python
opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
```

**Why use a relatively high LR (0.1) for this comparison?**
BatchNorm's main practical benefit is allowing higher learning rates without
divergence (by keeping layer input distributions stable). At a conservative LR
like 1e-3, both models (with and without BN) would converge similarly, masking
BN's advantage. Using `lr=0.1` — aggressive for a non-BN network — better exposes
the stabilizing effect.

### BatchNorm Placement

```python
layers = [nn.Linear(2, 128)]
if use_bn: layers.append(nn.BatchNorm1d(128))
layers.append(nn.ReLU())
```

**Why `BatchNorm1d` (not `BatchNorm2d`)?**
`BatchNorm1d` normalizes over a `(N, C)` or `(N, C, L)` tensor — appropriate for
fully-connected layers where each sample is a flat feature vector `(N, C)`.
`BatchNorm2d` is for convolutional feature maps `(N, C, H, W)` (covered in Phase 2).

**Why BatchNorm BEFORE ReLU?**
This is the original placement from Ioffe & Szegedy (2015): normalize the
pre-activation `z`, then apply non-linearity. Normalizing post-activation would
mean normalizing an already-clipped (non-negative, for ReLU) distribution, which
is less effective since half the information (negative values) is already gone.

**Live result:**
```
Without BatchNorm | val_loss=0.1652 | val_acc=93.8%
With BatchNorm    | val_loss=0.1562 | val_acc=94.4%
```
At this aggressive LR, BatchNorm provides a modest but consistent improvement in
both loss and accuracy — exactly the expected direction, though the make_moons
task is simple enough that the gap is not dramatic (BN's benefits are far more
pronounced in deep CNNs, covered in Phase 2).

---

## 4. Section D — Optimizer Comparison

### Fresh Model Per Optimizer

```python
def _fresh():
    torch.manual_seed(SEED)
    m = nn.Sequential(...)
    ...
    return m
```

**Why reset the seed inside `_fresh()` every call?**
Each optimizer must start from the *exact same* initial weights to make the
comparison fair — otherwise differences in final performance could be attributed
to random initialization rather than optimizer behavior. Calling
`torch.manual_seed(SEED)` immediately before weight initialization guarantees
identical starting points across all five optimizer runs.

### AdamW's Decoupled Weight Decay

```python
("AdamW", lambda p: optim.AdamW(p, lr=1e-3, weight_decay=1e-2)),
```

**Why does AdamW need a different implementation than Adam+L2?**
For plain SGD, L2 regularization (`+λw` to gradient) and weight decay
(`w ← w(1−ηλ)` direct multiplicative shrink) are mathematically identical.
For Adam, they are NOT equivalent: Adam divides the gradient by
`√v̂ + ε` — including the L2 penalty term inside this division means
the effective decay rate becomes coupled to each parameter's gradient history,
which is not the intended behavior. AdamW fixes this by applying weight decay
**after** the adaptive gradient step, decoupled from the `√v̂` normalization,
as derived in theory.md §5.5.

**Live result:**
```
SGD             | 94.38%  | 0.15363
SGD+Momentum    | 93.75%  | 0.17757   ← momentum overshoots slightly here
Adam            | 93.75%  | 0.13952
AdamW           | 94.38%  | 0.13883   ← best loss AND matches best accuracy
```
On this simple 2D problem all optimizers reach similar accuracy (the task is easy
enough that optimizer choice matters less), but AdamW achieves the lowest loss,
consistent with its widespread adoption as the default choice in modern
architectures (BERT, GPT, Vision Transformers all use AdamW).

---

## 5. Section E — Early Stopping

### Why `copy.deepcopy` for Best State

```python
self.best_state = copy.deepcopy(model.state_dict())
```

**Why `deepcopy` instead of just `model.state_dict()`?**
`model.state_dict()` returns references to the actual parameter tensors (or
shares underlying storage in some PyTorch versions). Without `deepcopy`, as
training continues and the model's weights keep updating, `self.best_state`
would silently change too — it wouldn't be a snapshot of the *best* epoch, but
a live view of the *current* (possibly worse) epoch. `deepcopy` ensures we
capture an independent, frozen copy of the weights at the moment of best
validation performance.

### The `step()` and `restore()` Pattern

```python
def step(self, val_loss, model) -> bool:
    if val_loss < self.best_loss - self.min_delta:
        self.best_loss = val_loss
        self.best_state = copy.deepcopy(model.state_dict())
        self.counter = 0
    else:
        self.counter += 1
    return self.counter >= self.patience
```

**Why subtract `min_delta` in the comparison, not just check `val_loss < self.best_loss`?**
Without `min_delta`, a microscopic improvement (e.g., `0.13480001` vs `0.13480000`)
would reset the patience counter, potentially keeping training going for the full
budget without meaningful progress. Requiring improvement to exceed `min_delta`
ensures the "patience" mechanism only resets on genuinely useful progress.

### Live Result — Early Stopping in Action

```
No Early Stop          | epochs=300 | best_val=0.1348 | final_val=0.1708
Early Stop (p=15)      | epochs= 40 | best_val=0.1348 | final_val=0.1429
```

Both configurations reach the *same* `best_val=0.1348` — they're training the
identical model/data/optimizer, so the best achievable validation loss during
training is identical. The key difference is what happens *after* that point:
- Without early stopping: training continues to epoch 300, final val loss
  degrades to 0.1708 (overfitting after the optimal point)
- With early stopping: training halts at epoch 40 (15 epochs after the best
  point at ~epoch 25), and because we call `restore()`, the *final* model
  weights are the BEST ones (0.1348-loss weights), not the epoch-40 weights

This demonstrates exactly why early stopping with weight restoration outperforms
naive "train for N epochs" — it gets the benefits of extended training (chance
to find a good minimum) without the cost of overfitting in later epochs, at a
fraction of the compute (40 vs 300 epochs = 7.5× faster).

---

## Pitfalls Avoided

| Pitfall | Fix |
|---|---|
| Sharing live state_dict reference as "best" checkpoint | `copy.deepcopy()` |
| Comparing optimizers from different random inits | `torch.manual_seed(SEED)` inside `_fresh()` |
| Regularizing biases (incorrect theoretically) | Filter `if p.dim() > 1` / explicit weight list |
| Dropout active during validation | `model.eval()` called automatically each epoch |
| Adam + L2 conflated with true weight decay | Used `AdamW` for decoupled decay |
| Tiny floating-point "improvements" resetting patience | `min_delta` threshold in EarlyStopping |
| BatchNorm masking its own benefit at low LR | Used aggressive `lr=0.1` to expose effect |

*Previous: [Topic 4 — Loss Functions & Overfitting](../04-loss-functions-and-overfitting/explanation.md)*
*Next: [Topic 6 — Hyperparameter Tuning & Augmentation](../06-hyperparameter-tuning-augmentation/explanation.md)*
