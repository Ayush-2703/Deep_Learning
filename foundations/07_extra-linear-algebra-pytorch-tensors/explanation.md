# Code Explanation: Linear Algebra & PyTorch Tensor Fundamentals

**Phase 1 — Topic 7 (Extra) | `implementation.py` walkthrough**

---

## 1. Section A — Core Linear Algebra Operations

### Verifying Identities with `assert` Instead of Just Printing

```python
C = A @ B
assert torch.allclose(C, torch.tensor([[19., 22.], [43., 50.]]))
```

**Why use `assert torch.allclose` rather than just printing and eyeballing?**
This file is meant to be a trustworthy reference. Hard-coding the expected
result and asserting equality means that if PyTorch's behavior ever changed,
or if a future edit introduced a typo, the script would fail loudly (raise
`AssertionError`) rather than silently printing a wrong value that looks
plausible. `torch.allclose` (rather than `==`) is used because floating-point
arithmetic can introduce tiny rounding differences even for mathematically
exact identities.

### Outer Product Shape

```python
outer = torch.outer(a, b)   # a: (3,), b: (3,) → outer: (3,3)
```

**Why does `outer(a,b)` produce a matrix when both inputs are vectors?**
The outer product `a⊗b = abᵀ` treats `a` as a column vector `(m,1)` and `b` as
a row vector `(1,n)`; their matrix product is `(m,1)@(1,n) = (m,n)`. This
differs fundamentally from `torch.dot(a,b)`, which treats both as row vectors
contracted to a scalar. The distinction matters directly in backprop: weight
gradients `∂L/∂W = δ·aᵀ` (Topic 3) are outer products, producing a full matrix
gradient — not the scalar you'd get from a dot product.

### Why `torch.linalg.inv` Rather Than Manual Cofactor Expansion

```python
A_inv = torch.linalg.inv(A)
```

PyTorch's `linalg.inv` uses LU decomposition internally (via LAPACK), which is
numerically stable and `O(n³)` — far better than the `O(n!)` cofactor-expansion
formula taught in introductory linear algebra courses. In practice, explicit
matrix inversion is rarely used in deep learning itself (most operations only
need `solve()` for linear systems, which is more stable than forming the
explicit inverse), but is included here for the conceptual identity `AA⁻¹=I`.

---

## 2. Section B — Matrix Calculus Verification

### Why Autograd is Used as Ground Truth, Not the Other Way Around

```python
f = x @ A @ x          # = xᵀAx
f.backward()
analytical = (A + A.T) @ x.detach()
print(f"Match: {torch.allclose(x.grad, analytical, atol=1e-4)}")
```

**Why verify the *hand-derived formula* against autograd, rather than just
trusting the formula?**
This inverts the typical relationship (usually we trust math and use it to
debug code, as in Topic 3's gradient checking). Here, the goal is pedagogical:
PyTorch's autograd is a black box that "just works," and confirming it agrees
with the textbook formula `∂(xᵀAx)/∂x=(A+Aᵀ)x` builds confidence in *both*
directions — it shows the formula is correct, AND it demonstrates that
autograd correctly implements the chain rule for matrix-valued expressions
without us having to manually derive backward passes for every operation
(unlike the `ManualMLP` in Topic 3, which we built by hand specifically to
contrast with this automatic approach).

### Why `.detach()` on `x` for the Analytical Computation

```python
analytical = (A + A.T) @ x.detach()
```

**Why detach `x` here specifically?**
`x` has `requires_grad=True`. If we computed `(A+A.T) @ x` without detaching,
this expression would itself be added to the *same* computation graph that
`f.backward()` already walked. While this particular code wouldn't error
(since we're not calling `.backward()` again on this new expression), detaching
is best practice to clearly signal "this is now just a numerical value I'm
comparing against, not a tensor I intend to differentiate again." It also
avoids accidentally growing the graph unnecessarily.

### Live Verification Results

```
[1] ∂(aᵀx)/∂x = a            → Match: True
[2] ∂(xᵀAx)/∂x = (A+Aᵀ)x      → Match: True
[3] ∂‖Ax−b‖²/∂x = 2Aᵀ(Ax−b)   → Match: True
```

All three identities matched to numerical precision (`atol=1e-4`), confirming
both the textbook formulas AND PyTorch's autograd implementation are mutually
consistent. Identity [3] is especially relevant: it is the gradient used in
linear regression's normal equations and is the same mathematical structure
underlying the output-layer gradient `δ=(Â−Y)` derived in Topic 3 — both come
from differentiating a squared/quadratic objective with respect to a linear
prediction.

---

## 3. Section C — Eigendecomposition & SVD

### Why `torch.linalg.eigh`, Not `torch.linalg.eig`

```python
eigvals, eigvecs = torch.linalg.eigh(A)    # eigh for symmetric matrices
```

**Why specifically the "h" (Hermitian/symmetric) variant?**
`torch.linalg.eig` is the general-purpose eigendecomposition that works for any
square matrix, but returns **complex-valued** eigenvalues/eigenvectors in
general (since non-symmetric real matrices can have complex eigenvalues).
`torch.linalg.eigh` is specialized for symmetric (or Hermitian) matrices,
which are *guaranteed* to have real eigenvalues and orthogonal eigenvectors —
exploiting this guarantee, `eigh` is both faster and returns clean real-valued
outputs without needing to discard imaginary parts. Since our example matrix
`A=[[4,1],[1,3]]` is symmetric, `eigh` is the mathematically correct and more
efficient choice.

### Verifying the Eigenvalue Equation Element-by-Element

```python
for i in range(len(eigvals)):
    lhs = A @ eigvecs[:, i]
    rhs = eigvals[i] * eigvecs[:, i]
    print(f"Verify Av=λv (i={i}): {torch.allclose(lhs, rhs, atol=1e-5)}")
```

**Why index `eigvecs[:, i]` (column i), not `eigvecs[i, :]` (row i)?**
By PyTorch/LAPACK convention, `torch.linalg.eigh` returns eigenvectors as the
**columns** of the returned matrix — i.e., `eigvecs[:, i]` is the i-th
eigenvector, corresponding to `eigvals[i]`. This matches the mathematical
convention `A = QΛQᵀ` where `Q`'s columns are the eigenvectors. Indexing
the wrong axis (`eigvecs[i, :]`) would silently grab the wrong vector and
produce an incorrect/failing verification.

### Low-Rank Approximation via SVD

```python
rank1_approx = S[0] * torch.outer(U[:, 0], Vt[0, :])
```

**Why does this single outer product reconstruct part of M?**
The full SVD reconstruction is `M = Σᵢ σᵢ·uᵢ⊗vᵢ` (a sum of rank-1 outer-product
terms, weighted by singular values, sorted from largest to smallest). Taking
just the first term (`i=0`, the largest singular value) gives the *best
possible* rank-1 approximation to `M` in the Frobenius-norm sense (this is
the Eckart-Young theorem). The live result shows this single term already
captures **86.9%** of the matrix's "energy" (`σ₁²/(σ₁²+σ₂²)`), which is the
core mathematical idea behind LoRA (Low-Rank Adaptation, covered in Phase 5):
a full weight update matrix can often be well-approximated by a much
lower-rank factorization, saving enormous numbers of trainable parameters.

---

## 4. Section D — PyTorch Tensor Fundamentals

### Demonstrating the `view()` Failure on Non-Contiguous Memory

```python
xt = x2.t()    # transpose makes memory non-contiguous
print(f"x2.t().is_contiguous() = {xt.is_contiguous()}")
try:
    xt.view(-1)
except RuntimeError as e:
    print("FAILED as expected: non-contiguous memory")
    print(f".reshape(-1) instead works: shape={xt.reshape(-1).shape}")
```

**Why does transpose break contiguity, and why does this matter?**
A tensor's underlying data is stored as a flat 1D buffer in memory, with a
"stride" pattern describing how to interpret it as a multi-dimensional array.
For a `(3,4)` tensor stored row-major, the strides are `(4,1)` — move 4
elements to advance one row, 1 element to advance one column. Transposing
swaps the logical shape to `(4,3)` but does NOT physically rearrange the
underlying buffer — it just swaps the stride metadata to `(1,4)`. This makes
the tensor "non-contiguous": its elements, read in the new logical row-major
order, are no longer adjacent in the underlying buffer. `view()` requires
contiguity because it reinterprets the existing buffer directly (zero-copy,
fast) — it cannot do this safely on non-contiguous data. `reshape()` detects
this case and transparently calls `.contiguous().view(...)` instead (an actual
memory copy), trading a small performance cost for robustness. This is exactly
why `reshape` is recommended as the "safe default" while `view` is an
explicit performance optimization for when you know memory is contiguous.

### `cat` vs `stack`: Concrete Shape Walkthrough

```python
a = torch.ones(3, 4); b = torch.zeros(3, 4)
cat_result   = torch.cat([a, b], dim=0)     # shape (6, 4)
stack_result = torch.stack([a, b], dim=0)   # shape (2, 3, 4)
```

**Why does `cat` along dim=0 give `(6,4)` but `stack` along dim=0 give `(2,3,4)`?**
`torch.cat` requires all input tensors to already have the same number of
dimensions, and it concatenates *along* an existing dimension — `(3,4)` and
`(3,4)` concatenated along dim 0 simply extends that dimension's size from 3
to `3+3=6`, keeping the result 2D. `torch.stack` instead inserts a brand-new
dimension at the specified position and arranges the inputs along it — two
`(3,4)` tensors become `(2,3,4)`, where the new size-2 dimension indexes
"which original tensor." A common real-world use case: stacking per-sample
loss values computed in a loop into a single batched tensor for later analysis,
versus concatenating multiple existing batches into one larger batch.

---

## 5. Section E — Broadcasting Rules

### The Outer-Sum Pattern

```python
col = torch.tensor([[1.], [2.], [3.]])      # shape (3,1)
row = torch.tensor([[10., 20., 30.]])        # shape (1,3)
outer_sum = col + row                          # shape (3,3)
```

**Why does adding a `(3,1)` and `(1,3)` tensor produce a `(3,3)` result, not
an error?**
Following the broadcasting rule from theory.md §7 (align from the right, expand
size-1 dimensions): comparing `(3,1)` and `(1,3)` dimension-by-dimension from
the right: the last dims are `1` vs `3` — compatible (1 expands to 3); the
second-to-last dims are `3` vs `1` — compatible (1 expands to 3). Both
tensors get broadcast to `(3,3)`, and the result `outer_sum[i,j] = col[i,0] +
row[0,j]` is an "outer sum" table — every pairwise combination of the two
input vectors. This pattern appears throughout deep learning, e.g., computing
pairwise distance matrices: `dist[i,j] = ‖x_i - x_j‖` style computations often
start from exactly this kind of broadcast.

### Multi-Dimensional Broadcasting Example

```python
X = torch.randn(8, 1, 6, 1)
Y = torch.randn(7, 1, 5)
Z = X + Y    # shape (8, 7, 6, 5)
```

**Walking through the alignment:**
```
X:        (8, 1, 6, 1)
Y: (implicit leading 1)   (1, 7, 1, 5)   ← Y is treated as if prepended with a 1
                            ↓   ↓   ↓   ↓
Compare:   8 vs 1 → 8     1 vs 7 → 7    6 vs 1 → 6     1 vs 5 → 5
Result:              (8,        7,         6,         5)
```
Y has only 3 explicit dimensions, but broadcasting implicitly treats it as
having a leading dimension of size 1 (left-padding with 1s) to match X's 4
dimensions. This is a common pattern in attention mechanisms (Phase 4), where
tensors of shape `(batch, heads, seq, seq)` need to broadcast against masks of
shape `(batch, 1, 1, seq)`.

### Why Example 4 Deliberately Triggers an Error

```python
try:
    bad = torch.ones(3, 4) + torch.ones(3, 5)
except RuntimeError as e:
    print("(3,4)+(3,5) FAILS as expected")
```

Demonstrating the *failure* case is as important as demonstrating success —
it shows precisely which broadcasting rule is violated (`4 ≠ 5` and neither
equals `1`), helping build the mental model of *why* certain shape mismatches
in real training code produce `RuntimeError: The size of tensor a (4) must
match the size of tensor b (5)` style messages, which are extremely common
debugging encounters in practice.

---

## 6. Section F — Autograd Mechanics

### Why `y.grad` is `None` by Default for Non-Leaf Tensors

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
z = y + 1
z.backward()
print(x.grad)   # 4.0 — populated
print(y.grad)   # None — NOT populated, even though y required grad implicitly
```

**Why does PyTorch not store gradients for `y` by default?**
This is a deliberate memory optimization. In a deep network with millions of
intermediate activations, storing `.grad` for every single non-leaf tensor
would consume enormous memory for values that are almost never needed after
backprop completes (we only need gradients with respect to the *parameters*
we're updating, which are leaves). PyTorch only populates `.grad` for leaf
tensors by default; if you specifically need an intermediate gradient (e.g.,
for visualization, as in this script, or for tasks like neural style transfer
that need gradients with respect to activations), call `.retain_grad()`
explicitly on that tensor before calling `backward()`.

### The In-Place Pitfall, Demonstrated Live

```python
x4 = torch.tensor([1.0, 2.0], requires_grad=True)
y4 = x4 * 2          # y4 depends on x4's CURRENT values
x4.add_(1.0)         # in-place modify x4 AFTER y4 was computed
y4.sum().backward()  # ERROR: autograd needs original x4
```

**Why does this specific sequence raise an error?**
Autograd's backward pass for `y4 = x4*2` needs to know `∂y4/∂x4 = 2` — this
particular gradient doesn't actually depend on x4's *value*, only its
existence, so you might expect this to work. However, PyTorch's autograd
engine conservatively tracks "version numbers" for tensors: every in-place
operation increments a tensor's version counter, and during `backward()`,
PyTorch checks whether any tensor it needs has been modified since the forward
pass recorded it. Since `x4` was modified in-place after `y4` was computed,
PyTorch detects the version mismatch and raises an error — even in this safe
case — as a conservative defense against the broader and genuinely dangerous
class of bugs where an in-place modification WOULD silently corrupt a needed
gradient (e.g., `y = x**2` truly does need `x`'s original value to compute
`dy/dx=2x`). The live output confirms PyTorch catches this defensively.

### `torch.no_grad()` Builds No Graph At All

```python
x6 = torch.tensor(5.0, requires_grad=True)
with torch.no_grad():
    y6 = x6 * 2
print(y6.requires_grad)   # False
print(y6.grad_fn)          # None
```

**Why does `y6.requires_grad` become `False` even though `x6.requires_grad`
is `True`?**
Inside a `no_grad()` context, PyTorch globally disables graph construction —
operations execute normally (computing correct values), but no `grad_fn` is
attached to any output, and `requires_grad` propagation is suppressed
entirely. This is precisely why `model.eval()` combined with
`with torch.no_grad():` (used throughout Topics 1–6 for validation/inference)
is both correct AND efficient: it skips the memory and compute overhead of
building a computation graph for a forward pass where we will never call
`.backward()`.

---

## Pitfalls Avoided / Demonstrated

| Pitfall | How This Script Handles It |
|---|---|
| Trusting hand-derived gradients without verification | Asserts identities against PyTorch autograd |
| Confusing `eig` vs `eigh` for symmetric matrices | Uses `eigh` explicitly, documents why |
| Indexing eigenvector rows instead of columns | Explicit `eigvecs[:, i]` with comment |
| Assuming `view()` always works after reshape-like ops | Demonstrates the contiguity failure live |
| Confusing `cat` (extend dim) with `stack` (new dim) | Side-by-side shape comparison |
| Silent broadcasting errors in production code | Explicit failure case (Example 4) shown |
| Expecting `.grad` on intermediate tensors | Shows `None` default + `retain_grad()` fix |
| In-place ops silently corrupting gradients | Live `RuntimeError` catch + safe alternative |

---

*Previous: [Topic 6 — Hyperparameter Tuning & Augmentation](../06-hyperparameter-tuning-augmentation/explanation.md)*

**Phase 1 — Deep Learning Foundations is now complete.** All 7 topics (Perceptron/MLP,
Activation Functions, Gradient Descent/Backprop, Loss Functions/Overfitting,
Regularization/Optimizers/BatchNorm, Hyperparameter Tuning/Augmentation, and
Linear Algebra/PyTorch Fundamentals) have full theory, working implementation,
and line-by-line explanation files.

*Next: Phase 2 — Convolutional Neural Networks & Computer Vision*
