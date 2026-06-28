"""
Topic: Convolutions, Kernels, Pooling & Receptive Fields
=======================================================================
Repository : deep-learning/cnns/01-convolution-basics/
File       : implementation.py

Sections:
  A │ 1D convolution from scratch (NumPy) — verified against PyTorch nn.Conv1d
  B │ 2D convolution from scratch (NumPy) — verified against PyTorch nn.Conv2d
  C │ Classic hand-designed kernels — edge detection, blur, sharpen
  D │ Padding & stride effects — output size formula verification
  E │ Pooling from scratch — MaxPool & AvgPool, verified against PyTorch
  F │ Receptive field growth — empirical measurement across stacked layers
  G │ im2col — convolution as matrix multiplication, verified equivalence
  H │ Parameter counting — Conv vs Dense comparison
  I │ Visualization dashboard
"""

import os, warnings
import numpy as np
import matplotlib, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

SEED = 42
RESULTS = "results"; os.makedirs(RESULTS, exist_ok=True)
np.random.seed(SEED); torch.manual_seed(SEED)
print(f"[CONFIG] PyTorch: {torch.__version__}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION A — 1D CONVOLUTION FROM SCRATCH
# ═════════════════════════════════════════════════════════════════════════════

def conv1d_scratch(x: np.ndarray, w: np.ndarray, bias: float = 0.0,
                    stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    1D cross-correlation (the deep-learning "convolution") from scratch.

    x : (L,)       input signal
    w : (K,)       kernel
    Returns: (L_out,)   where L_out = floor((L+2P-K)/S)+1
    """
    if padding > 0:
        x = np.pad(x, (padding, padding), mode="constant")
    L, K = len(x), len(w)
    L_out = (L - K) // stride + 1
    out = np.zeros(L_out)
    for i in range(L_out):
        start = i * stride
        out[i] = np.sum(x[start:start + K] * w) + bias
    return out


def section_a_conv1d():
    print("\n" + "="*65)
    print("SECTION A — 1D Convolution: Scratch vs PyTorch")
    print("="*65)

    x = np.random.randn(20).astype(np.float32)
    w = np.random.randn(5).astype(np.float32)
    bias = 0.3

    for stride, padding in [(1, 0), (2, 0), (1, 2), (2, 1)]:
        scratch_out = conv1d_scratch(x, w, bias, stride, padding)

        # PyTorch reference: nn.Conv1d expects (batch, channels, length)
        conv = nn.Conv1d(1, 1, kernel_size=len(w), stride=stride, padding=padding)
        with torch.no_grad():
            conv.weight[:] = torch.tensor(w).view(1, 1, -1)
            conv.bias[:]   = torch.tensor([bias])
        torch_out = conv(torch.tensor(x).view(1, 1, -1)).detach().numpy().flatten()

        match = np.allclose(scratch_out, torch_out, atol=1e-4)
        print(f"  stride={stride}, padding={padding} | "
              f"out_len={len(scratch_out):2d} | match={match}")
        assert match, "1D conv mismatch!"

    print("\n  ✓ All 1D convolution configurations verified against PyTorch")
    return x, w


# ═════════════════════════════════════════════════════════════════════════════
# SECTION B — 2D CONVOLUTION FROM SCRATCH
# ═════════════════════════════════════════════════════════════════════════════

def conv2d_scratch(x: np.ndarray, w: np.ndarray, bias: float = 0.0,
                    stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    2D cross-correlation, single channel in / single channel out.

    x : (H, W)        input image
    w : (Kh, Kw)       kernel
    Returns: (H_out, W_out)
    """
    if padding > 0:
        x = np.pad(x, ((padding, padding), (padding, padding)), mode="constant")
    H, W = x.shape
    Kh, Kw = w.shape
    H_out = (H - Kh) // stride + 1
    W_out = (W - Kw) // stride + 1
    out = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            hi, wj = i * stride, j * stride
            patch = x[hi:hi + Kh, wj:wj + Kw]
            out[i, j] = np.sum(patch * w) + bias
    return out


def conv2d_multichannel_scratch(x: np.ndarray, w: np.ndarray, bias: np.ndarray,
                                 stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    Full multi-channel 2D convolution.

    x : (C_in, H, W)
    w : (C_out, C_in, Kh, Kw)
    bias: (C_out,)
    Returns: (C_out, H_out, W_out)
    """
    C_in, H, W = x.shape
    C_out, _, Kh, Kw = w.shape
    if padding > 0:
        x = np.pad(x, ((0, 0), (padding, padding), (padding, padding)), mode="constant")
        H, W = x.shape[1], x.shape[2]
    H_out = (H - Kh) // stride + 1
    W_out = (W - Kw) // stride + 1
    out = np.zeros((C_out, H_out, W_out))
    for k in range(C_out):
        for i in range(H_out):
            for j in range(W_out):
                hi, wj = i * stride, j * stride
                patch = x[:, hi:hi + Kh, wj:wj + Kw]      # (C_in, Kh, Kw)
                out[k, i, j] = np.sum(patch * w[k]) + bias[k]
    return out


def section_b_conv2d():
    print("\n" + "="*65)
    print("SECTION B — 2D Convolution: Scratch vs PyTorch")
    print("="*65)

    # Single-channel test
    x = np.random.randn(10, 10).astype(np.float32)
    w = np.random.randn(3, 3).astype(np.float32)
    bias = 0.2

    for stride, padding in [(1, 0), (2, 0), (1, 1), (2, 2)]:
        scratch_out = conv2d_scratch(x, w, bias, stride, padding)
        conv = nn.Conv2d(1, 1, kernel_size=3, stride=stride, padding=padding)
        with torch.no_grad():
            conv.weight[:] = torch.tensor(w).view(1, 1, 3, 3)
            conv.bias[:]   = torch.tensor([bias])
        torch_out = conv(torch.tensor(x).view(1, 1, 10, 10)).detach().numpy()[0, 0]
        match = np.allclose(scratch_out, torch_out, atol=1e-4)
        print(f"  [single-channel] stride={stride}, padding={padding} | "
              f"out_shape={scratch_out.shape} | match={match}")
        assert match

    # Multi-channel test (C_in=3, C_out=4)
    x_mc = np.random.randn(3, 8, 8).astype(np.float32)
    w_mc = np.random.randn(4, 3, 3, 3).astype(np.float32)
    b_mc = np.random.randn(4).astype(np.float32)

    scratch_mc = conv2d_multichannel_scratch(x_mc, w_mc, b_mc, stride=1, padding=1)
    conv_mc = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)
    with torch.no_grad():
        conv_mc.weight[:] = torch.tensor(w_mc)
        conv_mc.bias[:]   = torch.tensor(b_mc)
    torch_mc = conv_mc(torch.tensor(x_mc).unsqueeze(0)).detach().numpy()[0]
    match_mc = np.allclose(scratch_mc, torch_mc, atol=1e-4)
    print(f"\n  [multi-channel C_in=3,C_out=4] out_shape={scratch_mc.shape} | "
          f"match={match_mc}")
    assert match_mc

    print("\n  ✓ All 2D convolution configurations verified against PyTorch")
    return x, w


# ═════════════════════════════════════════════════════════════════════════════
# SECTION C — CLASSIC HAND-DESIGNED KERNELS
# ═════════════════════════════════════════════════════════════════════════════

def make_synthetic_image(size=64) -> np.ndarray:
    """Generate a synthetic grayscale image with shapes for kernel demos."""
    img = np.zeros((size, size), dtype=np.float32)
    # White square
    img[10:30, 10:30] = 1.0
    # Gradient triangle region
    for i in range(35, 55):
        for j in range(35, 55):
            if j - 35 <= (i - 35):
                img[i, j] = 0.6
    # Diagonal line
    for k in range(size):
        if 0 <= k < size and 0 <= size - 1 - k < size:
            img[k, max(0, size - 1 - k - 1):size - 1 - k + 1] = 0.9
    return img


def apply_kernel_demo():
    print("\n" + "="*65)
    print("SECTION C — Classic Hand-Designed Kernels")
    print("="*65)

    img = make_synthetic_image()

    kernels = {
        "Vertical Edge (Prewitt)": np.array([[1,0,-1],[1,0,-1],[1,0,-1]], dtype=np.float32),
        "Horizontal Edge (Prewitt)": np.array([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=np.float32),
        "Sharpen": np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32),
        "Box Blur": np.ones((3,3), dtype=np.float32) / 9.0,
        "Sobel-X": np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32),
        "Sobel-Y": np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32),
    }

    results = {"original": img}
    print(f"\n  Applying {len(kernels)} classic kernels to {img.shape} synthetic image:")
    for name, k in kernels.items():
        out = conv2d_scratch(img, k, bias=0.0, stride=1, padding=1)
        results[name] = out
        print(f"    {name:25s} | output range: [{out.min():.2f}, {out.max():.2f}]")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# SECTION D — PADDING & STRIDE EFFECTS ON OUTPUT SIZE
# ═════════════════════════════════════════════════════════════════════════════

def output_size_formula(N, K, P, S):
    """N_out = floor((N + 2P - K)/S) + 1"""
    return (N + 2*P - K) // S + 1


def section_d_padding_stride():
    print("\n" + "="*65)
    print("SECTION D — Padding & Stride: Output Size Formula Verification")
    print("="*65)

    configs = [
        (224, 7, 3, 2, "ResNet first-layer config"),
        (32,  3, 1, 1, "Same padding, 32x32 input"),
        (32,  3, 0, 1, "Valid padding (shrinks by 2)"),
        (32,  2, 0, 2, "Standard 2x2 maxpool stride 2"),
        (28,  5, 2, 1, "LeNet-style 5x5 same padding"),
    ]

    print(f"\n  {'N':>4} {'K':>3} {'P':>3} {'S':>3} | {'Formula':>10} | {'PyTorch':>9} | Description")
    print("  " + "─"*70)

    results = []
    for N, K, P, S, desc in configs:
        formula_out = output_size_formula(N, K, P, S)

        conv = nn.Conv2d(1, 1, kernel_size=K, stride=S, padding=P)
        with torch.no_grad():
            dummy = torch.randn(1, 1, N, N)
            torch_out = conv(dummy).shape[-1]

        match = formula_out == torch_out
        results.append((N, K, P, S, formula_out, desc))
        print(f"  {N:>4} {K:>3} {P:>3} {S:>3} | {formula_out:>10} | "
              f"{torch_out:>9} | {desc}  {'✓' if match else '✗ MISMATCH'}")
        assert match

    print("\n  ✓ Output-size formula matches PyTorch for all configurations")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# SECTION E — POOLING FROM SCRATCH
# ═════════════════════════════════════════════════════════════════════════════

def maxpool2d_scratch(x: np.ndarray, k: int = 2, stride: int = 2) -> np.ndarray:
    """Max pooling, single channel. x: (H,W) -> (H_out,W_out)"""
    H, W = x.shape
    H_out, W_out = (H - k)//stride + 1, (W - k)//stride + 1
    out = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            hi, wj = i*stride, j*stride
            out[i, j] = np.max(x[hi:hi+k, wj:wj+k])
    return out


def avgpool2d_scratch(x: np.ndarray, k: int = 2, stride: int = 2) -> np.ndarray:
    """Average pooling, single channel."""
    H, W = x.shape
    H_out, W_out = (H - k)//stride + 1, (W - k)//stride + 1
    out = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            hi, wj = i*stride, j*stride
            out[i, j] = np.mean(x[hi:hi+k, wj:wj+k])
    return out


def section_e_pooling():
    print("\n" + "="*65)
    print("SECTION E — Pooling: Scratch vs PyTorch")
    print("="*65)

    x = np.random.randn(8, 8).astype(np.float32)

    max_scratch = maxpool2d_scratch(x, k=2, stride=2)
    max_torch = F.max_pool2d(torch.tensor(x).view(1,1,8,8), kernel_size=2, stride=2)
    max_torch = max_torch.numpy()[0,0]
    match_max = np.allclose(max_scratch, max_torch, atol=1e-5)
    print(f"\n  MaxPool(k=2,s=2):  shape={max_scratch.shape} | match={match_max}")
    assert match_max

    avg_scratch = avgpool2d_scratch(x, k=2, stride=2)
    avg_torch = F.avg_pool2d(torch.tensor(x).view(1,1,8,8), kernel_size=2, stride=2)
    avg_torch = avg_torch.numpy()[0,0]
    match_avg = np.allclose(avg_scratch, avg_torch, atol=1e-5)
    print(f"  AvgPool(k=2,s=2):  shape={avg_scratch.shape} | match={match_avg}")
    assert match_avg

    # Demonstrate translation robustness of max pooling
    shifted = np.roll(x, shift=1, axis=1)   # shift by 1 pixel horizontally
    max_orig = maxpool2d_scratch(x, 2, 2)
    max_shift = maxpool2d_scratch(shifted, 2, 2)
    diff_pct = np.mean(max_orig != max_shift) * 100
    print(f"\n  Translation test: shifting input by 1px changes "
          f"{diff_pct:.1f}% of max-pooled outputs")
    print("  (Small shifts are partially absorbed by the pooling window)")

    print("\n  ✓ Pooling operations verified against PyTorch")
    return x, max_scratch, avg_scratch


# ═════════════════════════════════════════════════════════════════════════════
# SECTION F — RECEPTIVE FIELD GROWTH
# ═════════════════════════════════════════════════════════════════════════════

def compute_receptive_field(layer_configs: list) -> list:
    """
    Compute receptive field after each layer using the recurrence:
      RF_L = RF_{L-1} + (K_L - 1) * cumulative_stride_so_far

    layer_configs: list of (kernel_size, stride) tuples
    Returns: list of receptive field sizes, one per layer
    """
    rf = 1
    cumulative_stride = 1
    rf_history = [rf]
    for K, S in layer_configs:
        rf = rf + (K - 1) * cumulative_stride
        cumulative_stride *= S
        rf_history.append(rf)
    return rf_history


def empirical_receptive_field(layer_configs: list, input_size: int = 64) -> int:
    """
    Empirically measure receptive field by backpropagating a gradient
    from a single output unit and counting non-zero input gradient pixels.
    """
    layers = []
    for K, S in layer_configs:
        pad = K // 2     # same-ish padding to keep things simple
        layers.append(nn.Conv2d(1, 1, kernel_size=K, stride=S, padding=pad, bias=False))
    model = nn.Sequential(*layers)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.ones_(m.weight)    # all-ones kernel: any non-zero contributes

    x = torch.zeros(1, 1, input_size, input_size, requires_grad=True)
    out = model(x)
    center = out.shape[-1] // 2
    out[0, 0, center, center].backward()

    nonzero_mask = (x.grad[0, 0].abs() > 1e-8)
    rows = torch.where(nonzero_mask.any(dim=1))[0]
    if len(rows) == 0:
        return 0
    return int(rows.max() - rows.min() + 1)


def section_f_receptive_field():
    print("\n" + "="*65)
    print("SECTION F — Receptive Field Growth")
    print("="*65)

    # Case 1: all stride=1, 3x3 kernels (typical VGG-style stack)
    configs_s1 = [(3, 1)] * 6
    rf_s1 = compute_receptive_field(configs_s1)
    emp_s1 = empirical_receptive_field(configs_s1)
    print(f"\n  [Stride=1 throughout, 3x3 kernels × 6 layers]")
    print(f"    Formula RF per layer: {rf_s1}")
    print(f"    Empirical RF (last layer): {emp_s1}  (formula predicts {rf_s1[-1]})")

    # Case 2: alternating stride=2 (typical downsampling network)
    configs_s2 = [(3, 2), (3, 2), (3, 2), (3, 2)]
    rf_s2 = compute_receptive_field(configs_s2)
    emp_s2 = empirical_receptive_field(configs_s2, input_size=128)
    print(f"\n  [Stride=2 throughout, 3x3 kernels × 4 layers]")
    print(f"    Formula RF per layer: {rf_s2}")
    print(f"    Empirical RF (last layer): {emp_s2}  (formula predicts {rf_s2[-1]})")

    # Case 3: VGG argument — three 3x3 vs one 7x7
    configs_3x3 = [(3,1),(3,1),(3,1)]
    rf_3x3 = compute_receptive_field(configs_3x3)[-1]
    rf_7x7 = compute_receptive_field([(7,1)])[-1]
    params_3x3 = 3 * (3*3)     # per input/output channel pair
    params_7x7 = 1 * (7*7)
    print(f"\n  [VGG argument: three 3x3 vs one 7x7]")
    print(f"    Three 3x3 convs: RF={rf_3x3}, relative params={params_3x3}")
    print(f"    One 7x7 conv:    RF={rf_7x7}, relative params={params_7x7}")
    print(f"    Same RF achieved with {params_3x3/params_7x7*100:.0f}% of the parameters")

    return {"s1": rf_s1, "s2": rf_s2, "vgg_3x3": rf_3x3, "vgg_7x7": rf_7x7}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION G — IM2COL: CONVOLUTION AS MATRIX MULTIPLICATION
# ═════════════════════════════════════════════════════════════════════════════

def im2col(x: np.ndarray, Kh: int, Kw: int, stride: int = 1) -> tuple:
    """
    Transform input image into column matrix for matmul-based convolution.

    x: (C_in, H, W)
    Returns: (X_col, H_out, W_out)
      X_col: (Kh*Kw*C_in, H_out*W_out)
    """
    C_in, H, W = x.shape
    H_out = (H - Kh) // stride + 1
    W_out = (W - Kw) // stride + 1

    col = np.zeros((C_in, Kh, Kw, H_out, W_out), dtype=x.dtype)
    for i in range(Kh):
        for j in range(Kw):
            col[:, i, j, :, :] = x[:, i:i+stride*H_out:stride, j:j+stride*W_out:stride]

    X_col = col.reshape(C_in*Kh*Kw, H_out*W_out)
    return X_col, H_out, W_out


def conv2d_via_im2col(x: np.ndarray, w: np.ndarray, bias: np.ndarray,
                        stride: int = 1) -> np.ndarray:
    """
    Full multi-channel convolution implemented via im2col + matmul.
    x: (C_in,H,W),  w: (C_out,C_in,Kh,Kw),  bias: (C_out,)
    """
    C_out, C_in, Kh, Kw = w.shape
    X_col, H_out, W_out = im2col(x, Kh, Kw, stride)         # (C_in*Kh*Kw, H_out*W_out)
    W_row = w.reshape(C_out, C_in*Kh*Kw)                       # (C_out, C_in*Kh*Kw)
    Y = W_row @ X_col + bias.reshape(-1, 1)                    # (C_out, H_out*W_out)
    return Y.reshape(C_out, H_out, W_out)


def section_g_im2col():
    print("\n" + "="*65)
    print("SECTION G — im2col: Convolution as Matrix Multiplication")
    print("="*65)

    x = np.random.randn(3, 12, 12).astype(np.float32)
    w = np.random.randn(8, 3, 3, 3).astype(np.float32)
    b = np.random.randn(8).astype(np.float32)

    direct_out = conv2d_multichannel_scratch(x, w, b, stride=1, padding=0)
    im2col_out = conv2d_via_im2col(x, w, b, stride=1)

    match = np.allclose(direct_out, im2col_out, atol=1e-4)
    print(f"\n  Direct loop-based conv shape: {direct_out.shape}")
    print(f"  im2col matmul-based shape:    {im2col_out.shape}")
    print(f"  Outputs match: {match}")
    assert match

    # Demonstrate the matrix shapes explicitly
    X_col, H_out, W_out = im2col(x, 3, 3, 1)
    print(f"\n  im2col transforms input {x.shape} into column matrix {X_col.shape}")
    print(f"  Kernel reshaped from {w.shape} into row matrix ({w.shape[0]}, {3*3*3})")
    print(f"  Single matmul produces output: ({w.shape[0]}, {H_out*W_out}) → "
          f"reshaped to {direct_out.shape}")

    print("\n  ✓ im2col matmul-based convolution verified equivalent to direct convolution")
    return X_col.shape


# ═════════════════════════════════════════════════════════════════════════════
# SECTION H — PARAMETER COUNTING: CONV vs DENSE
# ═════════════════════════════════════════════════════════════════════════════

def section_h_param_counting():
    print("\n" + "="*65)
    print("SECTION H — Parameter Counting: Conv vs Dense")
    print("="*65)

    configs = [
        (32, 3, 256, "32x32x3 image, 256-unit hidden layer"),
        (64, 3, 512, "64x64x3 image, 512-unit hidden layer"),
        (224, 3, 1000, "224x224x3 image (ImageNet-scale), 1000-unit layer"),
    ]

    print(f"\n  {'Input':>20} | {'Dense Params':>15} | {'Conv3x3 Params':>16} | Ratio")
    print("  " + "─"*68)

    for size, c_in, out_units, desc in configs:
        dense_params = size*size*c_in*out_units + out_units
        conv_params  = 3*3*c_in*out_units + out_units
        ratio = dense_params / conv_params
        print(f"  {size}x{size}x{c_in:<10} | {dense_params:>15,} | {conv_params:>16,} | {ratio:>6.0f}×")

    print(f"\n  Conv layers achieve equivalent output channel count with"
          f" orders of magnitude fewer parameters,")
    print(f"  while ALSO preserving spatial structure that dense layers discard.")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION I — VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def build_figures(kernel_results, rf_results, pooling_data):
    # ── Figure 1: Kernel effects on synthetic image ───────────────────────
    fig1 = plt.figure(figsize=(16, 8))
    fig1.suptitle("Phase 2 — Topic 1: Classic Convolution Kernels", fontsize=13, fontweight="bold")
    names = list(kernel_results.keys())
    n = len(names)
    cols = 4
    rows = (n + cols - 1) // cols
    for idx, name in enumerate(names):
        ax = fig1.add_subplot(rows, cols, idx+1)
        ax.imshow(kernel_results[name], cmap="gray")
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.axis("off")
    plt.tight_layout()
    path1 = os.path.join(RESULTS, "01_kernel_effects.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"\n  [VIZ] Figure 1 saved → {path1}")
    plt.close(fig1)

    # ── Figure 2: Receptive field + pooling + im2col concept ──────────────
    fig2 = plt.figure(figsize=(16, 9))
    fig2.suptitle("Phase 2 — Topic 1: Receptive Field, Pooling & Parameter Efficiency",
                  fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.4, wspace=0.32)
    ax1, ax2 = fig2.add_subplot(gs[0,0]), fig2.add_subplot(gs[0,1])
    ax3, ax4 = fig2.add_subplot(gs[1,0]), fig2.add_subplot(gs[1,1])

    # Panel 1: RF growth comparison (stride1 vs stride2)
    ax1.plot(range(len(rf_results["s1"])), rf_results["s1"], "o-",
             color="#3498db", lw=2, label="Stride=1 (linear growth)")
    ax1.plot(range(len(rf_results["s2"])), rf_results["s2"], "s-",
             color="#e74c3c", lw=2, label="Stride=2 (exponential growth)")
    ax1.set_title("Receptive Field Growth", fontweight="bold", fontsize=10)
    ax1.set_xlabel("Layer"); ax1.set_ylabel("Receptive Field (pixels)")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    # Panel 2: VGG argument bar chart
    labels = ["Three 3×3\nconvs", "One 7×7\nconv"]
    rfs    = [rf_results["vgg_3x3"], rf_results["vgg_7x7"]]
    params = [27, 49]
    x_pos = np.arange(2)
    ax2b = ax2.twinx()
    ax2.bar(x_pos-0.2, rfs, 0.35, color="#27ae60", label="Receptive Field")
    ax2b.bar(x_pos+0.2, params, 0.35, color="#e67e22", label="Relative Params")
    ax2.set_xticks(x_pos); ax2.set_xticklabels(labels)
    ax2.set_ylabel("Receptive Field", color="#27ae60")
    ax2b.set_ylabel("Relative Parameters", color="#e67e22")
    ax2.set_title("VGG Insight: Same RF, Fewer Params", fontweight="bold", fontsize=10)

    # Panel 3: Original vs MaxPool vs AvgPool
    x, max_s, avg_s = pooling_data
    ax3.imshow(x, cmap="viridis")
    ax3.set_title(f"Original (8×8)", fontweight="bold", fontsize=10)
    ax3.axis("off")

    # Panel 4: side by side maxpool/avgpool
    combined = np.concatenate([max_s, np.full((max_s.shape[0],1), np.nan), avg_s], axis=1)
    im = ax4.imshow(combined, cmap="viridis")
    ax4.set_title("MaxPool (left) vs AvgPool (right), 4×4 each", fontweight="bold", fontsize=10)
    ax4.axis("off")

    plt.tight_layout()
    path2 = os.path.join(RESULTS, "01_receptive_field_pooling.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"  [VIZ] Figure 2 saved → {path2}")
    plt.close(fig2)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "▓"*65)
    print("  Phase 2 — Topic 1: Convolutions, Kernels, Pooling & Receptive Fields")
    print("▓"*65)

    section_a_conv1d()
    section_b_conv2d()
    kernel_results = apply_kernel_demo()
    section_d_padding_stride()
    pooling_data = section_e_pooling()
    rf_results = section_f_receptive_field()
    section_g_im2col()
    section_h_param_counting()

    build_figures(kernel_results, rf_results, pooling_data)

    print("\n" + "▓"*65)
    print("  ✓ Topic 1 complete. All NumPy implementations verified against PyTorch.")
    print("▓"*65 + "\n")


if __name__ == "__main__":
    main()

