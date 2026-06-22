# Code Explanation: Gradient Descent & Backpropagation

**Phase 1 — Topic 3 | `implementation.py` walkthrough**

---

## 1. Section A — ManualMLP: Column Convention

```python
X_col = X_tr.T   # (2, 400) — features as ROWS, samples as COLUMNS
Y_col = y_tr.reshape(1, -1)  # (1, 400)
```
**Why column convention?** It matches the mathematical notation in theory.md exactly:
`Z¹ = W¹X + b¹` where `W¹ ∈ ℝ^(n¹×n⁰)` and `X ∈ ℝ^(n⁰×N)` gives `Z¹ ∈ ℝ^(n¹×N)`.
PyTorch uses row convention (samples as rows) — neither is "correct", but column convention makes the matrix math read directly from textbooks.

```python
Z1 = self.W1 @ X + self.b1   # (n¹,n⁰)×(n⁰,N) + (n¹,1) → (n¹,N)
```
`self.b1` is shape `(n¹,1)` — NumPy broadcasts this across all N columns automatically.

---

## 2. The Cache — Why Every Tensor Is Saved

```python
self.cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
```
Backprop needs **all intermediate values** computed during the forward pass:
- `Z1` → compute `ReLU'(Z1)` (which neurons fired)
- `A1` → compute `dW2 = dZ2 @ A1.T`
- `A2` → compute `dZ2 = A2 - Y`
- `X`  → compute `dW1 = dZ1 @ X.T`

Without caching, each backward step would require rerunning the forward pass — multiplying time by O(L).

---

## 3. The Beautiful Simplification: δ² = Â − Y

```python
dZ2 = (1.0 / N) * (A2 - Y)
```
This combines two chain-rule steps into one elegant result. Full derivation:
```
∂L/∂A2 = -(Y/A2) + (1-Y)/(1-A2)       [BCE derivative]
∂A2/∂Z2 = A2*(1-A2)                    [sigmoid derivative]
∂L/∂Z2  = ∂L/∂A2 * ∂A2/∂Z2
         = [-(Y/A2)+(1-Y)/(1-A2)] * A2*(1-A2)
         = -Y*(1-A2) + (1-Y)*A2
         = A2 - Y
```
The `1/N` factor averages the gradient over the batch (matching the `1/N` in BCE).

---

## 4. The Transpose Rule in Backward

```python
dA1 = self.W2.T @ dZ2   # (n¹,1)ᵀ × (1,N) → (n¹,N)
```
**Why transpose `W2`?** In the forward pass: `Z2 = W2 @ A1`. For the backward pass, the gradient flows through the weight matrix in the **opposite direction**. The chain rule gives:
```
∂L/∂A1 = (∂Z2/∂A1)ᵀ · ∂L/∂Z2 = W2ᵀ · dZ2
```
This is the mathematical reason transposes appear in backprop: they reverse the linear map.

---

## 5. Hadamard Product for Element-wise Gradients

```python
dZ1 = dA1 * self._relu_grad(Z1)   # element-wise ⊙, NOT matrix multiply
```
`ReLU` is an element-wise function: `yᵢ = max(0, zᵢ)` independently. Its Jacobian is a **diagonal matrix** `diag(ReLU'(z₁), ..., ReLU'(zₙ))`. Multiplying by a diagonal matrix equals element-wise multiplication — hence `*` not `@`.

---

## 6. Section B — Gradient Check: Why Central Differences

```python
g_num[idx] = (lp - lm) / (2.0 * eps)   # central, not forward
```
**Forward difference:** `[f(θ+ε) - f(θ)] / ε` — error O(ε)
**Central difference:** `[f(θ+ε) - f(θ-ε)] / (2ε)` — error O(ε²)

With `ε=1e-5`, forward difference has error ~1e-5 while central has error ~1e-10. This is the difference between a "suspicious" and a "clean" gradient check.

```python
for idx in np.ndindex(param.shape):
```
`np.ndindex` generates all valid multi-dimensional indices for the array shape — equivalent to nested for-loops over all dimensions but written as one clean iterator.

```python
e_rel = np.linalg.norm(g_anal - g_num) / (np.linalg.norm(g_anal) + np.linalg.norm(g_num) + 1e-8)
```
The `+ 1e-8` prevents division by zero when both gradients are near zero (e.g., for dead neurons or heavily regularised weights).

---

## 7. Section C — GD Variants: Tracking Steps Not Epochs

```python
loss_per_step.append(loss.item())
step += 1
if step >= max_steps:
    break
```
We track **gradient steps** (not epochs) because:
- Full-batch GD: 1 step/epoch → 800 samples processed per step
- Mini-batch (32): ~25 steps/epoch → 32 samples per step
- Stochastic (1): 800 steps/epoch → 1 sample per step

On an "epochs" axis, full-batch GD would look deceptively good (only one point per epoch). On a "steps" axis, all methods are on equal computational footing — each step represents roughly the same GPU time.

```python
# Smooth SGD noise
smooth = np.convolve(raw, np.ones(k)/k, mode="valid")
```
`np.convolve` with a uniform kernel computes a sliding average — a simple but effective way to reveal the trend beneath the high-variance SGD loss.

---

## 8. Section D — Handling Diverged Loss

```python
if torch.isfinite(loss):
    loss.backward()
    optimizer.step()
    tl += loss.item() * len(Xb)
else:
    tl += 10.0 * len(Xb)   # cap diverged loss for plotting
```
With `lr=2.0`, gradients explode and the loss becomes `inf` or `NaN` within a few steps. `torch.isfinite()` checks for both. Capping at 10.0 keeps the plot axis bounded — otherwise a single `inf` would break `matplotlib`.

---

## 9. Section E — Loss Surface: Quadratic Bowl

```python
def f(w1, w2):  return (w1-1)**2 + 4*(w2-1)**2
def grad_f(w1, w2): return np.array([2*(w1-1), 8*(w2-1)])
```
The factor 4 in `4*(w2-1)²` makes the w₂ direction 4× steeper (curvature 8 vs 2). This is the **condition number κ=4** that causes zig-zagging when the learning rate is large — GD overshoots in the steep direction while barely moving in the flat one.

Convergence requires `η < 2/λ_max = 2/8 = 0.25`. With `η=0.24`, the w₂ direction alternates sign each step (oscillates) but converges because `|1 - 0.24×8| = 0.92 < 1`.

---

## 10. Section F — Autograd: Gradient Accumulation Bug

```python
for i in range(3):
    loss = x ** 2
    loss.backward()
    print(x.grad)   # 6, 12, 18 — grows each iteration!
```
PyTorch **adds** (not replaces) gradients on `.backward()`. This design is intentional: it enables gradient accumulation across micro-batches (when GPU memory is too small for the full batch). But forgetting `zero_grad()` in a standard loop causes gradients to grow without bound.

```python
with torch.no_grad():
    y = x * 2 + 1
# y.requires_grad = False — no computation graph built
```
`no_grad()` skips graph construction entirely. Not just "don't compute gradients" — it never allocates the graph nodes. This is why inference with `no_grad()` uses less memory and runs faster than `detach()`, which builds the graph and then detaches from it.

---

## Key Design Decisions Summary

| Decision | Why |
|---|---|
| Column convention `(n_features, N)` | Matches textbook math: `Z=WX+b` |
| Cache all intermediate tensors | Avoid recomputing forward pass in backward |
| `dZ2 = A2 - Y` (not two-step) | BCE+sigmoid simplification saves a step |
| `W2.T @ dZ2` (transpose) | Reverses the forward linear map |
| `*` not `@` for ReLU gradient | Element-wise Jacobian = diagonal matrix |
| Central differences `(L+−L-)/(2ε)` | O(ε²) vs O(ε) accuracy |
| Track gradient steps not epochs | Fair comparison of batch sizes |
| `torch.isfinite()` guard | Prevents NaN from breaking training loop |
