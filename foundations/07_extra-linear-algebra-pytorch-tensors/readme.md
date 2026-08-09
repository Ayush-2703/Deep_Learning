# Theory: Linear Algebra & Calculus Refresher + PyTorch Tensor Fundamentals

**Deep Learning Mastery Repository**

---

## Table of Contents
1. [Scalars, Vectors, Matrices, Tensors](#1-scalars-vectors-matrices-tensors)
2. [Core Linear Algebra Operations](#2-core-linear-algebra-operations)
3. [Matrix Calculus](#3-matrix-calculus)
4. [Eigenvalues, Eigenvectors, and SVD](#4-eigenvalues-eigenvectors-and-svd)
5. [Probability Refresher](#5-probability-refresher)
6. [PyTorch Tensor Fundamentals](#6-pytorch-tensor-fundamentals)
7. [Broadcasting Rules](#7-broadcasting-rules)
8. [Autograd Mechanics](#8-autograd-mechanics)

---

## 1. Scalars, Vectors, Matrices, Tensors

```
Rank-0 (Scalar):    x ∈ ℝ                      e.g. a single loss value
Rank-1 (Vector):    x ∈ ℝⁿ                     e.g. one sample's features
Rank-2 (Matrix):    X ∈ ℝ^(m×n)                e.g. a batch of samples
Rank-3 (Tensor):    X ∈ ℝ^(c×h×w)              e.g. an image (channels, H, W)
Rank-4 (Tensor):    X ∈ ℝ^(b×c×h×w)            e.g. a batch of images
Rank-N (Tensor):    generalizes to any number of axes
```

In deep learning, "tensor" colloquially refers to any multi-dimensional array,
regardless of rank — this differs slightly from the strict mathematical/physics
definition of "tensor" but is the universal convention in ML frameworks.

---

## 2. Core Linear Algebra Operations

### 2.1 Matrix Multiplication

```
C = AB,    A ∈ ℝ^(m×k),  B ∈ ℝ^(k×n),  C ∈ ℝ^(m×n)

Cᵢⱼ = Σₗ₌₁ᵏ Aᵢₗ Bₗⱼ

Requirement: inner dimensions must match (A's columns = B's rows)
Complexity:  O(mnk) naive; O(n^2.37) via Strassen-like algorithms (rarely used in practice)
```

**Why matrix multiplication is THE core operation in deep learning:**
Every fully-connected layer computes `Wx + b` — a matrix-vector product. For a
batch, `WX + b` is a matrix-matrix product. GPUs are essentially optimized
matrix-multiplication machines (via cuBLAS/cuDNN), which is why deep learning's
explosive growth tracked GPU hardware improvements.

### 2.2 Transpose

```
(Aᵀ)ᵢⱼ = Aⱼᵢ

Properties:
  (Aᵀ)ᵀ = A
  (AB)ᵀ = BᵀAᵀ            ← reverses order!
  (A+B)ᵀ = Aᵀ + Bᵀ
```

### 2.3 Dot Product / Inner Product

```
a·b = aᵀb = Σᵢ aᵢbᵢ      (scalar result)

Geometric meaning:  a·b = ‖a‖‖b‖cos(θ)
  θ=0°:   vectors aligned, dot product maximal
  θ=90°:  vectors orthogonal, dot product = 0
  θ=180°: vectors opposite, dot product minimal (negative)
```

### 2.4 Outer Product

```
a ⊗ b = abᵀ ∈ ℝ^(m×n)    for a∈ℝᵘᵑ, b∈ℝⁿ

(a⊗b)ᵢⱼ = aᵢbⱼ            (matrix result, NOT scalar)
```

Used in backprop: `∂L/∂W = δ · aᵀ` is exactly an outer product (see Topic 3).

### 2.5 Norms

```
L1 norm:    ‖x‖₁ = Σᵢ|xᵢ|                      (Manhattan distance)
L2 norm:    ‖x‖₂ = √(Σᵢxᵢ²)                    (Euclidean distance)
L∞ norm:    ‖x‖∞ = maxᵢ|xᵢ|                    (Chebyshev distance)
Frobenius:  ‖A‖_F = √(Σᵢⱼ Aᵢⱼ²)               (matrix L2 analog)
```

### 2.6 Identity and Inverse

```
Identity:   I ∈ ℝ^(n×n),  Iᵢⱼ = 1 if i=j else 0
            AI = IA = A

Inverse:    A⁻¹ exists iff det(A) ≠ 0 (A is "non-singular")
            AA⁻¹ = A⁻¹A = I
```

---

## 3. Matrix Calculus

### 3.1 Gradient of a Scalar w.r.t. a Vector

```
f: ℝⁿ → ℝ,    ∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ  ∈ ℝⁿ
```

### 3.2 Key Identities Used Constantly in Deep Learning

```
1.  ∂(aᵀx)/∂x = a                       (linear form)
2.  ∂(xᵀAx)/∂x = (A + Aᵀ)x              (quadratic form)
    If A symmetric: ∂(xᵀAx)/∂x = 2Ax
3.  ∂(Ax)/∂x = Aᵀ                       (Jacobian of linear map)
4.  ∂(xᵀx)/∂x = 2x                       (special case of #2, A=I)
5.  ∂‖Ax−b‖²/∂x = 2Aᵀ(Ax−b)             (least squares gradient)
```

### 3.3 Jacobian Matrix

For f: ℝⁿ → ℝᵘᵑ:

```
J = ∂f/∂x ∈ ℝ^(m×n),    Jᵢⱼ = ∂fᵢ/∂xⱼ

This generalizes the gradient (m=1 case) to vector-valued functions.
Used heavily in backpropagation through layers (Topic 3).
```

### 3.4 Hessian Matrix (Second-Order)

```
H = ∇²f(x) ∈ ℝ^(n×n),    Hᵢⱼ = ∂²f/∂xᵢ∂xⱼ

Properties:
  H is symmetric (if f is twice continuously differentiable)
  H positive definite at x*  ⟹  x* is a local minimum
  H negative definite at x*  ⟹  x* is a local maximum
  H indefinite (mixed eigenvalue signs) ⟹  x* is a saddle point
```

Second-order optimizers (Newton's method, L-BFGS) use H; most deep learning
uses only first-order info (gradient) due to H's O(n²) memory cost.

### 3.5 Chain Rule (Matrix Form) — Backbone of Backprop

```
If z = g(y) and y = f(x), with x∈ℝⁿ, y∈ℝᵘᵑ, z∈ℝᵖ:

∂z/∂x = (∂z/∂y)(∂y/∂x)     ∈ ℝ^(p×n)
        [p×m]    [m×n]

This is exactly how backpropagation propagates gradients backward through layers.
```

---

## 4. Eigenvalues, Eigenvectors, and SVD

### 4.1 Eigendecomposition

```
Av = λv

where v ≠ 0 is an eigenvector, λ is the corresponding eigenvalue.

For symmetric A: A = QΛQᵀ
  Q = orthogonal matrix of eigenvectors (columns)
  Λ = diagonal matrix of eigenvalues
```

**Relevance to deep learning:**
- The Hessian's eigenvalues determine the loss landscape's curvature (Topic 3)
- PCA (dimensionality reduction) uses eigendecomposition of the covariance matrix
- Condition number `κ = λ_max/λ_min` predicts gradient descent convergence speed

### 4.2 Singular Value Decomposition (SVD)

```
A = UΣVᵀ,    A ∈ ℝ^(m×n)

U ∈ ℝ^(m×m):  orthogonal (left singular vectors)
Σ ∈ ℝ^(m×n):  diagonal, non-negative singular values σ₁≥σ₂≥...≥0
V ∈ ℝ^(n×n):  orthogonal (right singular vectors)
```

**Relevance:**
- Low-rank approximation: keep only the top-k singular values → compress weight matrices
- LoRA (Phase 5) decomposes weight UPDATES into low-rank factors using this idea
- PCA can be computed via SVD of the data matrix

---

## 5. Probability Refresher

### 5.1 Key Distributions in Deep Learning

```
Bernoulli(p):     P(x=1)=p, P(x=0)=1−p          ← binary classification output
Categorical(π):   P(x=k)=πₖ                      ← multi-class softmax output
Gaussian(μ,σ²):   p(x) = (1/√(2πσ²))e^{-(x-μ)²/2σ²}  ← weight init, VAEs (Phase 5)
```

### 5.2 Maximum Likelihood Estimation (MLE) → Loss Functions

```
MLE objective: θ* = argmax_θ  Πᵢ p(yᵢ|xᵢ;θ)
             = argmax_θ  Σᵢ log p(yᵢ|xᵢ;θ)         (log for numerical stability)
             = argmin_θ  −Σᵢ log p(yᵢ|xᵢ;θ)          (negate to minimize)

For Bernoulli likelihood:  −log p(y|x) = BCE loss     (exact derivation!)
For Gaussian likelihood:   −log p(y|x) = MSE loss (up to constants)
```

**This is THE reason BCE/MSE exist as loss functions** — they are not arbitrary;
they are the negative log-likelihood of the assumed output distribution.

### 5.3 KL Divergence (revisited from Topic 04)

```
KL(P‖Q) = Eₓ~P[log(P(x)/Q(x))] ≥ 0, equality iff P=Q
```

---

## 6. PyTorch Tensor Fundamentals

### 6.1 Tensor Creation

```python
torch.tensor([1,2,3])              # from list
torch.zeros(3,4)                    # all zeros, shape (3,4)
torch.ones(3,4)                     # all ones
torch.randn(3,4)                    # standard normal N(0,1)
torch.rand(3,4)                     # uniform [0,1)
torch.arange(0,10,2)                # [0,2,4,6,8]
torch.eye(3)                        # 3×3 identity matrix
torch.full((2,2), 7)                 # filled with value 7
torch.zeros_like(x)                  # same shape/dtype/device as x
```

### 6.2 Key Tensor Attributes

```python
x.shape       # torch.Size — dimensions
x.dtype       # torch.float32, torch.int64, etc.
x.device      # cpu or cuda:0
x.requires_grad  # whether autograd tracks this tensor
x.ndim        # number of dimensions
x.numel()     # total number of elements
```

### 6.3 Indexing and Slicing

```python
x[0]          # first row (for 2D)
x[:, 0]       # first column
x[1:3]        # rows 1 to 2
x[..., 0]     # last dimension, index 0 (ellipsis = "all preceding dims")
x[x > 0]      # boolean mask indexing — returns 1D tensor of matching elements
```

### 6.4 Reshaping Operations

```python
x.view(2, 6)         # reshape — REQUIRES contiguous memory, shares storage
x.reshape(2, 6)       # like view but copies if needed (safer, slightly slower)
x.flatten()           # collapse to 1D
x.squeeze()           # remove all dimensions of size 1
x.squeeze(0)          # remove dimension 0 specifically (if size 1)
x.unsqueeze(0)        # add a dimension of size 1 at position 0
x.permute(1, 0, 2)    # reorder dimensions arbitrarily
x.transpose(0, 1)     # swap two specific dimensions
```

**`view` vs `reshape`:** `view` requires the underlying memory to be contiguous
(no gaps/strides that prevent simple reinterpretation). After operations like
`transpose`, memory is non-contiguous, and `view` will raise an error;
`reshape` automatically falls back to copying. Use `.contiguous()` before
`view` if needed.

### 6.5 Common Tensor Operations

```python
torch.matmul(A, B)    # matrix multiply — also: A @ B
torch.sum(x, dim=0)    # sum along dimension 0
torch.mean(x, dim=1)   # mean along dimension 1
torch.max(x, dim=0)    # returns (values, indices)
torch.cat([a,b], dim=0) # concatenate along existing dimension
torch.stack([a,b], dim=0) # stack along NEW dimension
x.T                     # transpose (2D only, shorthand)
x.t()                   # transpose (2D only, explicit)
```

**`cat` vs `stack`:**
```
cat:    [a,b] each shape (3,4) → cat dim=0 → shape (6,4)   (extends existing dim)
stack:  [a,b] each shape (3,4) → stack dim=0 → shape (2,3,4)  (creates new dim)
```

### 6.6 Device Management

```python
x = x.to("cuda")           # move to GPU
x = x.to(device)            # move to whatever `device` variable specifies
x = x.cpu()                 # move back to CPU
x = x.cuda()                # shorthand for .to("cuda")

# Idiomatic device-agnostic code:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
x = x.to(device)
```

### 6.7 Dtype Management

```python
x.float()      # cast to float32
x.double()     # cast to float64
x.long()       # cast to int64
x.to(torch.float16)   # half precision (for mixed-precision training)

# Common mistake: mixing dtypes in operations raises RuntimeError
a = torch.tensor([1,2,3])           # int64 by default
b = torch.tensor([1.0, 2.0, 3.0])   # float32 by default
a + b   # ERROR in older PyTorch versions / requires explicit cast in some ops
```

---

## 7. Broadcasting Rules

PyTorch (like NumPy) automatically expands tensors of different shapes to
make element-wise operations possible, following these rules:

```
Rule: Align shapes from the RIGHT. Two dimensions are compatible if:
  (a) they are equal, OR
  (b) one of them is 1 (it gets broadcast/expanded to match the other)

Example 1:
  A shape: (3, 4)
  B shape:    (4,)     ← treated as (1, 4) for alignment
  Result:  (3, 4)       ← B is broadcast across the first dimension

Example 2:
  A shape: (8, 1, 6, 1)
  B shape:    (7, 1, 5)
  Aligned:  (8, 1, 6, 1)
            (1, 7, 1, 5)
  Result:  (8, 7, 6, 5)  ← each size-1 dim expands to match the other

Example 3 (INCOMPATIBLE):
  A shape: (3, 4)
  B shape: (3, 5)        ← neither equal nor 1 in last dim → ERROR
```

**Why broadcasting matters in deep learning:**
```python
Z = W @ X + b
# W@X: (out_features, batch)
# b:   (out_features,) or (out_features, 1)
# Broadcasting expands b across the batch dimension automatically
```

This is exactly the mechanism that let us write `Z1 = W1 @ X + b1` in Topic 3's
manual backprop without manually tiling `b1` across all N samples.

---

## 8. Autograd Mechanics

### 8.1 The `requires_grad` Flag

```python
x = torch.tensor(2.0, requires_grad=True)
```

Marks `x` as a leaf node that should accumulate gradients. Operations involving
`x` build a dynamic computation graph (`grad_fn` chain).

### 8.2 Leaf vs Non-Leaf Tensors

```python
x = torch.tensor(2.0, requires_grad=True)  # LEAF (user-created, requires_grad=True)
y = x ** 2                                   # NON-LEAF (result of an operation)

x.is_leaf  # True
y.is_leaf  # False
y.grad     # None — PyTorch doesn't populate .grad for non-leaf tensors by default
           # (use y.retain_grad() if you need it)
```

### 8.3 Detaching from the Graph

```python
y = x.detach()      # new tensor, SAME data, requires_grad=False, no graph history
```

Useful for: logging values without keeping the graph alive, using a tensor as a
constant in further computation, converting to NumPy (`.detach().numpy()`).

### 8.4 In-Place Operations and Autograd

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
x.add_(1.0)   # in-place: modifies x's underlying data directly

# CAUTION: in-place ops on tensors needed for backward can cause:
# "RuntimeError: a leaf Variable that requires grad is being used in an
#  in-place operation" or silently incorrect gradients in more complex graphs.
```

**Why this repository avoids in-place ops on tracked tensors:** Autograd needs
the ORIGINAL values of intermediate tensors to compute certain gradients (e.g.,
`d(x²)/dx = 2x` needs the original `x`). In-place modification can destroy
the values autograd needs, leading to silently wrong gradients in non-trivial
graphs. PyTorch detects many (not all) such cases and raises an error.

### 8.5 Computation Graph Lifecycle

```
1. Forward pass: operations build the graph (grad_fn references chain backward)
2. .backward(): walks the graph in reverse topological order, accumulating
   gradients into each leaf's .grad
3. By default, the graph is FREED after backward() (to save memory)
   → calling .backward() twice without retain_graph=True raises an error
```

---

## Key Equations Summary

| Concept | Formula |
|---|---|
| Matrix multiply | Cᵢⱼ = Σₗ AᵢₗBₗⱼ |
| Quadratic form gradient | ∂(xᵀAx)/∂x = (A+Aᵀ)x |
| Chain rule (matrix) | ∂z/∂x = (∂z/∂y)(∂y/∂x) |
| Eigendecomposition | Av=λv, A=QΛQᵀ (symmetric) |
| SVD | A=UΣVᵀ |
| MLE → BCE | −log p(y\|x) for Bernoulli = BCE |
| Broadcasting | align from right; dims match if equal or one is 1 |
