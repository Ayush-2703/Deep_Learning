"""
Topic: Linear Algebra, Calculus & PyTorch Tensor Fundamentals
===================================================================================
Repository : deep-learning/foundations/07-extra-linear-algebra-pytorch-tensors/
File       : implementation.py

Sections:
  A │ Core linear algebra operations — matmul, transpose, norms, dot/outer product
  B │ Matrix calculus verification — gradient identities checked via autograd
  C │ Eigendecomposition & SVD — visualized on a 2D transformation
  D │ PyTorch tensor fundamentals — creation, indexing, reshaping, dtype/device
  E │ Broadcasting rules — worked examples with shape tracing
  F │ Autograd mechanics — leaf/non-leaf, detach, in-place pitfalls
  G │ Visualization dashboard
"""

import os, warnings
import numpy as np
import matplotlib, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

SEED = 42
RESULTS = "results"; os.makedirs(RESULTS, exist_ok=True)
np.random.seed(SEED); torch.manual_seed(SEED)
print(f"[CONFIG] PyTorch: {torch.__version__}")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION A — CORE LINEAR ALGEBRA OPERATIONS
# ═════════════════════════════════════════════════════════════════════════════

def linear_algebra_basics():
    print("\n" + "="*65)
    print("SECTION A — Core Linear Algebra Operations")
    print("="*65)

    A = torch.tensor([[1., 2.], [3., 4.]])
    B = torch.tensor([[5., 6.], [7., 8.]])
    a = torch.tensor([1., 2., 3.])
    b = torch.tensor([4., 5., 6.])

    print(f"\n  A =\n{A}\n  B =\n{B}")

    # ── Matrix multiplication ──────────────────────────────────────────────
    C = A @ B                          # equivalently torch.matmul(A, B)
    print(f"\n  A @ B =\n{C}")
    # Verify manually: C[0,0] = 1*5 + 2*7 = 19
    assert torch.allclose(C, torch.tensor([[19., 22.], [43., 50.]]))
    print("  ✓ Manual verification: C[0,0]=1·5+2·7=19 matches")

    # ── Transpose ───────────────────────────────────────────────────────────
    print(f"\n  Aᵀ =\n{A.T}")
    assert torch.allclose((A @ B).T, B.T @ A.T)
    print("  ✓ (AB)ᵀ = BᵀAᵀ verified")

    # ── Dot product ─────────────────────────────────────────────────────────
    dot = torch.dot(a, b)
    print(f"\n  a·b = {dot.item():.1f}   (1·4+2·5+3·6 = {1*4+2*5+3*6})")

    # ── Outer product ───────────────────────────────────────────────────────
    outer = torch.outer(a, b)
    print(f"\n  a⊗b =\n{outer}")
    print(f"  Shape: {outer.shape}  (a has {a.shape[0]} elems, "
          f"b has {b.shape[0]} elems → {a.shape[0]}×{b.shape[0]} matrix)")

    # ── Norms ────────────────────────────────────────────────────────────────
    x = torch.tensor([3., -4., 0.])
    print(f"\n  x = {x.tolist()}")
    print(f"  ‖x‖₁ = {torch.norm(x, p=1).item():.2f}   (= 3+4+0 = 7)")
    print(f"  ‖x‖₂ = {torch.norm(x, p=2).item():.2f}   (= √(9+16+0) = 5)")
    print(f"  ‖x‖∞ = {torch.norm(x, p=float('inf')).item():.2f}   (= max|xᵢ| = 4)")

    # ── Identity & Inverse ─────────────────────────────────────────────────
    I = torch.eye(2)
    A_inv = torch.linalg.inv(A)
    print(f"\n  A⁻¹ =\n{A_inv}")
    print(f"  A @ A⁻¹ ≈ I: {torch.allclose(A @ A_inv, I, atol=1e-6)}")

    return {"A": A, "B": B, "a": a, "b": b}

# ═════════════════════════════════════════════════════════════════════════════
# SECTION B — MATRIX CALCULUS VERIFICATION
# Verify analytical identities using PyTorch autograd as ground truth
# ═════════════════════════════════════════════════════════════════════════════

def matrix_calculus_verification():
    print("\n" + "="*65)
    print("SECTION B — Matrix Calculus Identity Verification")
    print("="*65)

    torch.manual_seed(SEED)
    n = 4

    # ── Identity 1: ∂(aᵀx)/∂x = a ──────────────────────────────────────────
    a = torch.randn(n)
    x = torch.randn(n, requires_grad=True)
    f = torch.dot(a, x)
    f.backward()
    print(f"\n  [1] ∂(aᵀx)/∂x = a")
    print(f"      Autograd grad: {x.grad.numpy().round(3)}")
    print(f"      Analytical a:  {a.numpy().round(3)}")
    print(f"      Match: {torch.allclose(x.grad, a)}")

    # ── Identity 2: ∂(xᵀAx)/∂x = (A+Aᵀ)x ──────────────────────────────────
    A = torch.randn(n, n)
    x = torch.randn(n, requires_grad=True)
    f = x @ A @ x
    f.backward()
    analytical = (A + A.T) @ x.detach()
    print(f"\n  [2] ∂(xᵀAx)/∂x = (A+Aᵀ)x")
    print(f"      Autograd grad: {x.grad.numpy().round(3)}")
    print(f"      Analytical:    {analytical.numpy().round(3)}")
    print(f"      Match: {torch.allclose(x.grad, analytical, atol=1e-4)}")

    # ── Identity 3: ∂‖Ax−b‖²/∂x = 2Aᵀ(Ax−b) ──────────────────────────────
    A = torch.randn(n, n)
    b = torch.randn(n)
    x = torch.randn(n, requires_grad=True)
    residual = A @ x - b
    f = torch.dot(residual, residual)     # ‖Ax-b‖²
    f.backward()
    analytical = 2 * A.T @ (A @ x.detach() - b)
    print(f"\n  [3] ∂‖Ax−b‖²/∂x = 2Aᵀ(Ax−b)   (least squares gradient)")
    print(f"      Autograd grad: {x.grad.numpy().round(3)}")
    print(f"      Analytical:    {analytical.numpy().round(3)}")
    print(f"      Match: {torch.allclose(x.grad, analytical, atol=1e-4)}")

    return {"n": n}

# ═════════════════════════════════════════════════════════════════════════════
# SECTION C — EIGENDECOMPOSITION & SVD
# ═════════════════════════════════════════════════════════════════════════════

def eigendecomposition_svd():
    print("\n" + "="*65)
    print("SECTION C — Eigendecomposition & SVD")
    print("="*65)

    # Symmetric matrix for clean real eigendecomposition
    A = torch.tensor([[4., 1.], [1., 3.]])
    print(f"\n  A (symmetric) =\n{A}")

    eigvals, eigvecs = torch.linalg.eigh(A)    # eigh for symmetric matrices
    print(f"\n  Eigenvalues: {eigvals.numpy().round(4)}")
    print(f"  Eigenvectors (columns):\n{eigvecs.numpy().round(4)}")

    # Verify: A @ v = λ @ v for each eigenpair
    for i in range(len(eigvals)):
        lhs = A @ eigvecs[:, i]
        rhs = eigvals[i] * eigvecs[:, i]
        print(f"  Verify Av=λv (i={i}): {torch.allclose(lhs, rhs, atol=1e-5)}")

    # Reconstruct: A = QΛQᵀ
    Q, L = eigvecs, torch.diag(eigvals)
    A_recon = Q @ L @ Q.T
    print(f"\n  Reconstruction A=QΛQᵀ matches original: "
          f"{torch.allclose(A, A_recon, atol=1e-5)}")

    # ── SVD on a non-square matrix ─────────────────────────────────────────
    M = torch.randn(4, 2)
    U, S, Vt = torch.linalg.svd(M, full_matrices=False)
    print(f"\n  M shape: {M.shape}  →  U:{U.shape}  S:{S.shape}  Vt:{Vt.shape}")
    M_recon = U @ torch.diag(S) @ Vt
    print(f"  Reconstruction M=UΣVᵀ matches original: "
          f"{torch.allclose(M, M_recon, atol=1e-5)}")

    # Low-rank approximation demo
    rank1_approx = S[0] * torch.outer(U[:, 0], Vt[0, :])
    error_full = torch.norm(M - M_recon).item()
    error_rank1 = torch.norm(M - rank1_approx).item()
    print(f"\n  Full reconstruction error: {error_full:.2e}")
    print(f"  Rank-1 approx error:       {error_rank1:.4f}  "
          f"(σ₁={S[0]:.3f}, σ₂={S[1]:.3f} — captures "
          f"{S[0]**2/(S[0]**2+S[1]**2)*100:.1f}% of variance)")

    return {"A": A, "eigvals": eigvals, "eigvecs": eigvecs, "M": M,
            "U": U, "S": S, "Vt": Vt}

# ═════════════════════════════════════════════════════════════════════════════
# SECTION D — PYTORCH TENSOR FUNDAMENTALS
# ═════════════════════════════════════════════════════════════════════════════

def tensor_fundamentals():
    print("\n" + "="*65)
    print("SECTION D — PyTorch Tensor Fundamentals")
    print("="*65)

    # ── Creation ────────────────────────────────────────────────────────────
    print("\n  [Creation]")
    z = torch.zeros(2, 3); o = torch.ones(2, 3); r = torch.randn(2, 3)
    print(f"    zeros(2,3).shape = {z.shape}, dtype = {z.dtype}")
    print(f"    arange(0,10,2)   = {torch.arange(0, 10, 2).tolist()}")
    print(f"    eye(3) =\n{torch.eye(3)}")

    # ── Attributes ──────────────────────────────────────────────────────────
    x = torch.randn(2, 3, 4)
    print(f"\n  [Attributes] x = randn(2,3,4)")
    print(f"    shape={x.shape}  ndim={x.ndim}  numel={x.numel()}  dtype={x.dtype}")

    # ── Indexing ────────────────────────────────────────────────────────────
    print(f"\n  [Indexing] x = randn(3,4)")
    x2 = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    print(f"    x =\n{x2}")
    print(f"    x[0]    = {x2[0].tolist()}            ← first row")
    print(f"    x[:,1]  = {x2[:, 1].tolist()}              ← second column")
    print(f"    x[x>5]  = {x2[x2 > 5].tolist()}  ← boolean mask")

    # ── Reshaping ───────────────────────────────────────────────────────────
    print(f"\n  [Reshaping] starting shape: {x2.shape}")
    print(f"    .view(2,6).shape    = {x2.view(2, 6).shape}")
    print(f"    .flatten().shape    = {x2.flatten().shape}")
    print(f"    .unsqueeze(0).shape = {x2.unsqueeze(0).shape}  ← adds dim at front")
    y = x2.unsqueeze(0)
    print(f"    .squeeze().shape    = {y.squeeze().shape}     ← removes size-1 dims")

    # view vs reshape after transpose (non-contiguous memory)
    xt = x2.t()    # transpose makes memory non-contiguous
    print(f"\n    After transpose, x2.t().is_contiguous() = {xt.is_contiguous()}")
    try:
        xt.view(-1)   # this will fail on non-contiguous tensor
        print("    .view(-1) succeeded (unexpected)")
    except RuntimeError as e:
        print(f"    .view(-1) FAILED as expected: non-contiguous memory")
        print(f"    .reshape(-1) instead works: shape={xt.reshape(-1).shape}")

    # ── cat vs stack ────────────────────────────────────────────────────────
    a = torch.ones(3, 4); b = torch.zeros(3, 4)
    cat_result   = torch.cat([a, b], dim=0)
    stack_result = torch.stack([a, b], dim=0)
    print(f"\n  [cat vs stack] a.shape=b.shape={a.shape}")
    print(f"    cat([a,b],   dim=0).shape = {cat_result.shape}   ← extends existing dim")
    print(f"    stack([a,b], dim=0).shape = {stack_result.shape} ← creates new dim")

    # ── Device-agnostic pattern ────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_dev = torch.randn(3).to(device)
    print(f"\n  [Device] Detected device: {device}")
    print(f"    Tensor moved to {x_dev.device}")

    return {"x2": x2}

# ═════════════════════════════════════════════════════════════════════════════
# SECTION E — BROADCASTING RULES
# ═════════════════════════════════════════════════════════════════════════════

def broadcasting_examples():
    print("\n" + "="*65)
    print("SECTION E — Broadcasting Rules")
    print("="*65)

    # Example 1: matrix + vector
    A = torch.ones(3, 4)
    b = torch.tensor([1., 2., 3., 4.])
    result = A + b
    print(f"\n  [Example 1] A.shape={A.shape} + b.shape={b.shape}")
    print(f"    Result shape: {result.shape}  (b broadcast across rows)")
    print(f"    Result[0] = {result[0].tolist()}")

    # Example 2: column vector + row vector (outer-sum pattern)
    col = torch.tensor([[1.], [2.], [3.]])      # shape (3,1)
    row = torch.tensor([[10., 20., 30.]])        # shape (1,3)
    outer_sum = col + row
    print(f"\n  [Example 2] col.shape={col.shape} + row.shape={row.shape}")
    print(f"    Result shape: {outer_sum.shape}")
    print(f"    Result =\n{outer_sum}")
    print(f"    (Each element: col[i]+row[j] — like an outer addition table)")

    # Example 3: multi-dim broadcasting
    X = torch.randn(8, 1, 6, 1)
    Y = torch.randn(7, 1, 5)
    Z = X + Y
    print(f"\n  [Example 3] X.shape={tuple(X.shape)} + Y.shape={tuple(Y.shape)}")
    print(f"    Result shape: {tuple(Z.shape)}  (aligned from the right, "
          f"size-1 dims expand)")

    # Example 4: incompatible shapes
    print(f"\n  [Example 4] Incompatible shapes:")
    try:
        bad = torch.ones(3, 4) + torch.ones(3, 5)
        print("    Unexpectedly succeeded!")
    except RuntimeError as e:
        print(f"    (3,4)+(3,5) FAILS as expected — neither dim matches nor is 1")

    # Practical DL example: linear layer bias broadcast
    W_out = torch.randn(32, 10)   # (batch, out_features) — pretend this is W@X
    bias  = torch.randn(10)        # (out_features,)
    Z_out = W_out + bias
    print(f"\n  [DL Example] Linear layer: (W@X).shape={W_out.shape} + bias.shape={bias.shape}")
    print(f"    Result shape: {Z_out.shape}  ← bias auto-broadcast across batch dim")

    return {"outer_sum": outer_sum}

# ═════════════════════════════════════════════════════════════════════════════
# SECTION F — AUTOGRAD MECHANICS
# ═════════════════════════════════════════════════════════════════════════════

def autograd_mechanics():
    print("\n" + "="*65)
    print("SECTION F — Autograd Mechanics: Leaf, Detach, In-Place Pitfalls")
    print("="*65)

    # ── Leaf vs non-leaf ────────────────────────────────────────────────────
    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 2
    z = y + 1
    print(f"\n  [Leaf vs Non-Leaf]")
    print(f"    x.is_leaf = {x.is_leaf}   (user-created)")
    print(f"    y.is_leaf = {y.is_leaf}   (result of operation)")
    print(f"    z.is_leaf = {z.is_leaf}   (result of operation)")

    z.backward()
    print(f"    x.grad = {x.grad.item()}   (populated — x is a leaf)")
    print(f"    y.grad = {y.grad}          (None by default — y is non-leaf)")

    # retain_grad to inspect intermediate gradients
    x2 = torch.tensor(2.0, requires_grad=True)
    y2 = x2 ** 2
    y2.retain_grad()           # explicitly request gradient storage
    z2 = y2 + 1
    z2.backward()
    print(f"\n  [retain_grad] After y2.retain_grad():")
    print(f"    y2.grad = {y2.grad.item()}   (now populated)")

    # ── Detach ──────────────────────────────────────────────────────────────
    x3 = torch.tensor(3.0, requires_grad=True)
    y3 = x3 ** 2
    y3_detached = y3.detach()
    print(f"\n  [Detach]")
    print(f"    y3.requires_grad          = {y3.requires_grad}")
    print(f"    y3_detached.requires_grad = {y3_detached.requires_grad}")
    print(f"    Same underlying data: {torch.equal(y3, y3_detached)}")

    # ── In-place operation pitfall ─────────────────────────────────────────
    print(f"\n  [In-Place Pitfall]")
    try:
        x4 = torch.tensor([1.0, 2.0], requires_grad=True)
        y4 = x4 * 2          # y4 depends on x4
        x4.add_(1.0)         # in-place modify x4 AFTER y4 was computed
        y4.sum().backward()  # autograd needs original x4 — but it changed!
        print(f"    Unexpectedly succeeded: x4.grad={x4.grad}")
    except RuntimeError as e:
        print(f"    RuntimeError caught (as expected):")
        print(f"    'a leaf Variable that requires grad is being used in an in-place op'")

    # Safe alternative: out-of-place
    x5 = torch.tensor([1.0, 2.0], requires_grad=True)
    y5 = x5 * 2
    x5_new = x5 + 1.0       # out-of-place: creates NEW tensor, x5 unchanged
    y5.sum().backward()
    print(f"\n    Safe (out-of-place) alternative: x5.grad = {x5.grad.tolist()}")

    # ── no_grad context ─────────────────────────────────────────────────────
    x6 = torch.tensor(5.0, requires_grad=True)
    with torch.no_grad():
        y6 = x6 * 2
    print(f"\n  [no_grad context]")
    print(f"    y6.requires_grad = {y6.requires_grad}  (graph not built)")
    print(f"    y6.grad_fn       = {y6.grad_fn}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION G — VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def build_figures(eig_data):
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Phase 1 — Topic 7: Linear Algebra & PyTorch Tensor Fundamentals",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    a = [fig.add_subplot(gs[r,c]) for r in range(2) for c in range(3)]

    # ── Panel 1: Eigenvector visualization ────────────────────────────────
    A = eig_data["A"].numpy()
    eigvals = eig_data["eigvals"].numpy()
    eigvecs = eig_data["eigvecs"].numpy()

    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.stack([np.cos(theta), np.sin(theta)])
    ellipse = A @ circle      # A transforms the unit circle into an ellipse

    a[0].plot(circle[0], circle[1], "--", color="gray", lw=1.5, label="Unit circle")
    a[0].plot(ellipse[0], ellipse[1], color="#3498db", lw=2.2, label="A·(unit circle)")
    for i in range(2):
        v = eigvecs[:, i] * eigvals[i]
        a[0].annotate("", xy=(v[0], v[1]), xytext=(0,0),
                     arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2.5))
        a[0].text(v[0]*1.15, v[1]*1.15, f"λ{i+1}={eigvals[i]:.2f}",
                  fontsize=9, color="#e74c3c", fontweight="bold")
    a[0].set_title("Eigenvectors as Principal Axes of A", fontweight="bold", fontsize=10)
    a[0].set_xlabel("x₁"); a[0].set_ylabel("x₂")
    a[0].legend(fontsize=8); a[0].set_aspect("equal"); a[0].grid(True, alpha=0.3)

    # ── Panel 2: Singular values bar chart ────────────────────────────────
    S = eig_data["S"].numpy()
    a[1].bar(range(len(S)), S, color="#27ae60", edgecolor="white")
    a[1].set_title("Singular Values of M (4×2)", fontweight="bold", fontsize=10)
    a[1].set_xlabel("Index"); a[1].set_ylabel("σᵢ")
    a[1].set_xticks(range(len(S)))
    a[1].grid(True, axis="y", alpha=0.3)

    # ── Panel 3: Broadcasting outer-sum heatmap ──────────────────────────
    col = np.array([[1.],[2.],[3.]]); row = np.array([[10.,20.,30.]])
    grid_sum = col + row
    im = a[2].imshow(grid_sum, cmap="viridis", aspect="auto")
    for i in range(grid_sum.shape[0]):
        for j in range(grid_sum.shape[1]):
            a[2].text(j, i, f"{grid_sum[i,j]:.0f}", ha="center", va="center",
                     color="white", fontweight="bold")
    a[2].set_title("Broadcasting: col(3,1)+row(1,3)", fontweight="bold", fontsize=10)
    a[2].set_xticks(range(3)); a[2].set_yticks(range(3))
    plt.colorbar(im, ax=a[2], fraction=0.046)

    # ── Panel 4: Matrix multiplication as linear transform ─────────────────
    A2 = np.array([[2., 0.5], [0.3, 1.5]])
    grid_pts = np.array([[x,y] for x in np.linspace(-1,1,5) for y in np.linspace(-1,1,5)]).T
    transformed = A2 @ grid_pts
    a[3].scatter(grid_pts[0], grid_pts[1], c="#95a5a6", s=20, label="Original grid")
    a[3].scatter(transformed[0], transformed[1], c="#e74c3c", s=20, label="A·grid")
    a[3].set_title("Matrix as Linear Transformation", fontweight="bold", fontsize=10)
    a[3].legend(fontsize=8); a[3].set_aspect("equal"); a[3].grid(True, alpha=0.3)

    # ── Panel 5: Gradient identity verification (bar of errors) ────────────
    labels = ["∂(aᵀx)/∂x", "∂(xᵀAx)/∂x", "∂‖Ax-b‖²/∂x"]
    # Re-run quick verification for error magnitudes
    torch.manual_seed(SEED)
    n=4
    errs = []
    a_ = torch.randn(n); x_ = torch.randn(n, requires_grad=True)
    (a_ @ x_).backward(); errs.append(torch.norm(x_.grad - a_).item())

    A_ = torch.randn(n,n); x_ = torch.randn(n, requires_grad=True)
    (x_ @ A_ @ x_).backward()
    errs.append(torch.norm(x_.grad - (A_+A_.T)@x_.detach()).item())

    A_ = torch.randn(n,n); b_ = torch.randn(n); x_ = torch.randn(n, requires_grad=True)
    r = A_@x_-b_; (r@r).backward()
    errs.append(torch.norm(x_.grad - 2*A_.T@(A_@x_.detach()-b_)).item())

    a[4].bar(labels, errs, color="#9b59b6")
    a[4].set_title("Autograd vs Analytical: ‖Error‖", fontweight="bold", fontsize=10)
    a[4].set_ylabel("L2 Error (should be ~0)")
    a[4].tick_params(axis='x', labelsize=7)
    a[4].grid(True, axis="y", alpha=0.3)

    # ── Panel 6: Computation graph illustration (text-based diagram) ───────
    a[5].axis("off")
    graph_text = (
        "Computation Graph: z = (x² + 1)\n\n"
        "  x (leaf, requires_grad=True)\n"
        "  │\n"
        "  ▼  pow(2)\n"
        "  y = x²  (non-leaf)\n"
        "  │\n"
        "  ▼  add(1)\n"
        "  z = y+1  (non-leaf)\n\n"
        "backward(): z→y→x\n"
        "  dz/dy = 1\n"
        "  dy/dx = 2x\n"
        "  dz/dx = dz/dy · dy/dx = 2x"
    )
    a[5].text(0.05, 0.95, graph_text, fontsize=9.5, family="monospace",
             va="top", transform=a[5].transAxes)
    a[5].set_title("Autograd Computation Graph", fontweight="bold", fontsize=10)

    plt.tight_layout()
    path = os.path.join(RESULTS, "07_linalg_tensors.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  [VIZ] Figure saved → {path}")
    plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "▓"*65)
    print("  Phase 1 — Topic 7: Linear Algebra & PyTorch Tensor Fundamentals")
    print("▓"*65)

    linear_algebra_basics()
    matrix_calculus_verification()
    eig_data = eigendecomposition_svd()
    tensor_fundamentals()
    broadcasting_examples()
    autograd_mechanics()
    build_figures(eig_data)

    print("\n  ✓ Topic 7 complete. Phase 1 — Foundations is now FULLY complete.\n")

if __name__ == "__main__":
    main()
