# Theory: Convolutions, Kernels, Pooling & Receptive Fields

**Deep Learning Mastery Repository**

---

## Table of Contents
1. [Why Convolutions? Motivation Over Dense Layers](#1-why-convolutions-motivation-over-dense-layers)
2. [The Convolution Operation](#2-the-convolution-operation)
3. [1D, 2D, and 3D Convolutions](#3-1d-2d-and-3d-convolutions)
4. [Multi-Channel Convolution](#4-multi-channel-convolution)
5. [Padding](#5-padding)
6. [Stride](#6-stride)
7. [Output Size Formula](#7-output-size-formula)
8. [Parameter Sharing & Parameter Counting](#8-parameter-sharing--parameter-counting)
9. [Receptive Field](#9-receptive-field)
10. [Pooling](#10-pooling)
11. [Convolution as Matrix Multiplication (im2col)](#11-convolution-as-matrix-multiplication-im2col)

---

## 1. Why Convolutions? Motivation Over Dense Layers

### The Problem with Fully-Connected Layers on Images

A 224×224 RGB image flattened for a dense layer has `224×224×3 = 150,528`
input features. A single dense layer mapping this to just 1000 hidden units
requires:

```
Parameters = 150,528 × 1,000 + 1,000 ≈ 150.5 million parameters
            for ONE layer, before any depth is added
```

This is computationally wasteful and statistically inefficient for three reasons:

```
1. NO TRANSLATION INVARIANCE
   A dense layer learns a SEPARATE weight for every pixel position.
   A cat's ear detector learned at position (10,10) tells the network
   NOTHING about detecting an ear at position (200,200) — it must
   relearn the same pattern independently at every location.

2. IGNORES SPATIAL STRUCTURE
   Flattening destroys the 2D neighborhood relationships between pixels.
   A dense layer treats pixel (0,0) and pixel (0,1) — physically adjacent —
   exactly the same as pixel (0,0) and pixel (223,223) — physically distant.

3. PARAMETER EXPLOSION
   Parameters scale with O(H×W×C_in×C_out) for a single dense layer,
   making deep stacks of dense layers on raw images computationally infeasible.
```

### The Convolutional Solution

A convolutional layer instead learns a **small set of filters** (e.g., 3×3 or
5×5) that are **slid across the entire image**, reusing the SAME weights at
every spatial location:

```
Convolutional layer parameters = kernel_height × kernel_width × C_in × C_out + C_out
                                = independent of input image size H, W!

Example: 3×3 kernel, 3 input channels, 64 output channels:
  Parameters = 3×3×3×64 + 64 = 1,792
  (vs. 150+ million for the dense equivalent above)
```

This single design choice — **parameter sharing** — is the foundational
insight that makes CNNs both statistically efficient (fewer parameters to
learn, less prone to overfitting) and translation-equivariant (a learned
feature detector works the same regardless of where in the image it appears).

---

## 2. The Convolution Operation

### Mathematical Definition (True Convolution)

The continuous convolution of two functions is defined as:

```
(f * g)(t) = ∫ f(τ)g(t−τ)dτ
```

For discrete 1D signals:

```
(f * g)[n] = Σₘ f[m]·g[n−m]
```

Note the **flipped** kernel `g[n−m]` — true mathematical convolution flips
the kernel before sliding it.

### Cross-Correlation (What Deep Learning Actually Uses)

```
(f ⋆ g)[n] = Σₘ f[n+m]·g[m]
```

This does NOT flip the kernel — it directly slides the kernel across the input.

**Deep learning frameworks (PyTorch, TensorFlow) implement cross-correlation,
but call it "convolution."** This is not a bug — since the kernel weights are
learned from data anyway, whether the kernel is flipped or not during the
forward pass doesn't matter: the network can simply learn the flipped version
of whatever filter it needs. We adopt the deep-learning convention (cross-
correlation, no flip) throughout this repository, as virtually all DL
literature does.

### 2D Convolution — Concrete Walkthrough

```
Input (5×5):                    Kernel (3×3):
┌───┬───┬───┬───┬───┐           ┌───┬───┬───┐
│ 1 │ 2 │ 3 │ 0 │ 1 │           │ 1 │ 0 │ -1│
├───┼───┼───┼───┼───┤           ├───┼───┼───┤
│ 4 │ 5 │ 6 │ 1 │ 2 │           │ 1 │ 0 │ -1│
├───┼───┼───┼───┼───┤           ├───┼───┼───┤
│ 7 │ 8 │ 9 │ 2 │ 3 │           │ 1 │ 0 │ -1│
├───┼───┼───┼───┼───┤           └───┴───┴───┘
│ 1 │ 2 │ 1 │ 0 │ 4 │           (vertical edge detector)
├───┼───┼───┼───┼───┤
│ 0 │ 1 │ 2 │ 3 │ 1 │
└───┴───┴───┴───┴───┘

Output[0,0] = (1·1 + 2·0 + 3·-1) + (4·1 + 5·0 + 6·-1) + (7·1 + 8·0 + 9·-1)
            = (1+0-3) + (4+0-6) + (7+0-9)
            = -2 + -2 + -2 = -6

The kernel slides across all valid positions, producing a 3×3 output
(for a 5×5 input, 3×3 kernel, stride=1, no padding — see §7 for the formula).
```

### Why This Specific Kernel Detects Vertical Edges

The kernel `[[1,0,-1],[1,0,-1],[1,0,-1]]` computes `(left column sum) −
(right column sum)`. When the input has a sharp vertical edge (bright on the
left, dark on the right, or vice versa), this difference is large in
magnitude. When the input is uniform (no edge), left and right columns are
similar, and the difference is near zero. This is the classic **Prewitt
operator** for vertical edge detection — one of many hand-designed kernels
that CNNs learn to discover automatically from data.

---

## 3. 1D, 2D, and 3D Convolutions

### 3.1 1D Convolution

```
Input shape:  (C_in, L)              L = sequence length
Kernel shape: (C_out, C_in, K)        K = kernel size
Output shape: (C_out, L_out)

Used for: time series, audio waveforms, genomic sequences, text (character-level)
```

### 3.2 2D Convolution

```
Input shape:  (C_in, H, W)
Kernel shape: (C_out, C_in, Kh, Kw)
Output shape: (C_out, H_out, W_out)

Used for: images, spectrograms (treating frequency×time as a 2D "image")
```

### 3.3 3D Convolution

```
Input shape:  (C_in, D, H, W)         D = depth (e.g., time, or volumetric depth)
Kernel shape: (C_out, C_in, Kd, Kh, Kw)
Output shape: (C_out, D_out, H_out, W_out)

Used for: video (D=time/frames), medical volumetric scans (CT/MRI, D=slices)
```

**Key distinction — 3D conv on video vs. 2D conv applied per-frame:**
A 3D convolution's kernel spans MULTIPLE frames simultaneously, allowing it to
directly learn temporal patterns (e.g., motion) within a single layer. Applying
a 2D convolution independently to each frame (and combining afterward with a
separate temporal module, e.g., an RNN) cannot capture this joint
spatio-temporal pattern within the convolution itself.

---

## 4. Multi-Channel Convolution

A real convolutional layer operates on multi-channel inputs (e.g., RGB = 3
channels) and produces multi-channel outputs (multiple learned filters).

```
Input:   X ∈ ℝ^(C_in × H × W)
Kernel:  one output channel's kernel: W_k ∈ ℝ^(C_in × Kh × Kw)
         ALL output channels:         W   ∈ ℝ^(C_out × C_in × Kh × Kw)
Bias:    b ∈ ℝ^(C_out)

For output channel k, output spatial position (i,j):

Y[k,i,j] = b[k] + Σ_{c=0}^{C_in-1} Σ_{u=0}^{Kh-1} Σ_{v=0}^{Kw-1}
                    W[k,c,u,v] · X[c, i·s+u, j·s+v]

           ↑ sum over ALL input channels AND the full kernel window
```

**Critical insight:** each output channel's kernel spans the *entire* input
depth (`C_in`), not just one channel. A 3×3 kernel applied to a 3-channel RGB
image actually has `3×3×3 = 27` learnable weights (plus 1 bias) — it
mixes information across all three color channels at every spatial step.

---

## 5. Padding

### Why Pad?

Without padding, each convolution shrinks the spatial dimensions (since the
kernel can't center on border pixels without going out-of-bounds). Padding
adds artificial border pixels (typically zeros) to control output size.

### Padding Modes

```
VALID (no padding, p=0):
  Output shrinks by (K-1) per dimension.
  5×5 input, 3×3 kernel → 3×3 output

SAME (padding to preserve size, for stride=1):
  p = (K-1)/2   (assuming odd K)
  5×5 input, 3×3 kernel, p=1 → 5×5 output (same size preserved)

FULL (maximum padding, kernel touches every input pixel at least once):
  p = K-1
  5×5 input, 3×3 kernel, p=2 → 7×7 output (output LARGER than input)
```

### Why "Same" Padding Matters for Deep Networks

Without same-padding, a 20-layer CNN with 3×3 kernels would shrink a 224×224
image by `2×(K-1)/2 = 2` pixels per layer → after 20 layers, spatial size
shrinks to `224 - 20×2 = 184` — manageable, but for deeper networks (50+
layers) or larger kernels, the image could shrink to nothing before reaching
the network's intended depth. Same-padding allows arbitrary depth without
this constraint, decoupling depth from kernel size.

---

## 6. Stride

Stride controls how far the kernel moves between applications.

```
Stride=1: kernel moves 1 pixel at a time (dense, overlapping windows)
Stride=2: kernel moves 2 pixels at a time (skips every other position)

Effect: stride=s roughly DIVIDES output spatial size by s.

Use case: strided convolutions are commonly used INSTEAD of pooling to
downsample feature maps while learning the downsampling pattern (rather than
using a fixed max/average rule) — common in modern architectures (ResNet's
first conv uses stride=2).
```

---

## 7. Output Size Formula

For an input of size `N`, kernel size `K`, padding `P`, stride `S`:

```
N_out = ⌊(N + 2P − K) / S⌋ + 1
```

This formula applies independently to height and width (and depth for 3D).

**Worked examples:**

```
N=224, K=7, P=3, S=2:    N_out = ⌊(224+6-7)/2⌋+1 = ⌊223/2⌋+1 = 111+1 = 112
                          (this is exactly ResNet's first conv layer config!)

N=32,  K=3, P=1, S=1:    N_out = ⌊(32+2-3)/1⌋+1 = 31+1 = 32   (same padding ✓)

N=32,  K=3, P=0, S=1:    N_out = ⌊(32+0-3)/1⌋+1 = 29+1 = 30   (valid padding, shrinks by 2)

N=32,  K=2, P=0, S=2:    N_out = ⌊(32+0-2)/2⌋+1 = 15+1 = 16   (standard 2×2 maxpool, stride 2)
```

---

## 8. Parameter Sharing & Parameter Counting

### Convolutional Layer Parameter Count

```
Params = (Kh × Kw × C_in × C_out) + C_out
                                      ↑ one bias per output channel
```

### Comparison Table: Conv vs Dense for Same Input

```
Input: 32×32×3 image (3,072 values)

Dense layer → 256 units:
  Params = 3,072 × 256 + 256 = 786,688

Conv layer, 3×3 kernel, 3→256 channels:
  Params = 3×3×3×256 + 256 = 7,168

Ratio: Dense uses 110× MORE parameters than the equivalent conv layer,
       while the conv layer additionally PRESERVES spatial structure and
       gains translation equivariance — a strictly better trade for image data.
```

### Translation Equivariance (Formal Property)

```
Let T_d denote a translation (shift) by d pixels.
Convolution f satisfies:   f(T_d(x)) = T_d(f(x))

In words: shifting the input shifts the output by the same amount —
the SAME features are detected regardless of position.

(Pooling, covered in §10, additionally gives approximate translation
INVARIANCE — meaning small shifts barely change the output AT ALL,
useful for classification where exact position doesn't matter.)
```

---

## 9. Receptive Field

### Definition

The receptive field of a unit in layer `L` is the region of the *original
input image* that can influence that unit's value, accounting for all
convolution/pooling operations between the input and layer `L`.

### Growth Formula (Single Path, No Dilation)

```
RF_L = RF_{L-1} + (K_L - 1) × ∏_{i<L} S_i

where:
  K_L = kernel size at layer L
  S_i = stride at layer i (product of all PREVIOUS strides)
  RF_0 = 1 (a single input pixel)
```

**Worked example — 3 layers, all 3×3 kernels, stride 1:**

```
Layer 0 (input):      RF=1
Layer 1 (3×3, s=1):   RF = 1 + (3-1)×1 = 3
Layer 2 (3×3, s=1):   RF = 3 + (3-1)×1 = 5
Layer 3 (3×3, s=1):   RF = 5 + (3-1)×1 = 7

After 3 layers of 3×3 convs, each output unit "sees" a 7×7 region of the
ORIGINAL input — even though each individual kernel is only 3×3.
```

**Why this matters — VGGNet's key insight (Phase 2 Topic 2):**
Three stacked 3×3 convolutions achieve the SAME 7×7 receptive field as a
single 7×7 convolution, but with FEWER parameters:

```
Three 3×3 convs (C→C channels each): 3 × (3×3×C×C) = 27C²
One 7×7 conv (C→C channels):              7×7×C×C = 49C²

The stacked approach uses 27/49 ≈ 55% of the parameters for the SAME
receptive field, while ALSO adding two extra non-linearities (more
expressive power) along the way.
```

### Receptive Field with Strided/Pooling Layers

Strided convolutions and pooling layers cause the receptive field to grow
MUCH faster (multiplicatively) because the stride term `∏ S_i` compounds:

```
4 layers of 3×3 conv, stride=2 each:
  RF_1 = 1+(3-1)×1   = 3       (cumulative stride so far: 1)
  RF_2 = 3+(3-1)×2   = 7       (cumulative stride so far: 2)
  RF_3 = 7+(3-1)×4   = 15      (cumulative stride so far: 4)
  RF_4 = 15+(3-1)×8  = 31      (cumulative stride so far: 8)

Compare: 4 layers of 3×3 conv, stride=1 throughout → RF = 9 only.

Strided/pooling layers are why deep CNNs can achieve receptive fields
covering the ENTIRE input image after relatively few layers.
```

---

## 10. Pooling

### Max Pooling

```
MaxPool(X)[i,j] = max over window {X[i·s:i·s+k, j·s:j·s+k]}

Typical: k=2, s=2 → halves spatial dimensions, keeps the strongest activation
```

**Why max (not average)?** Max pooling preserves the presence of the
strongest detected feature in a local region, discarding weaker (likely noise)
activations. This provides a degree of robustness to small spatial
translations: if a feature shifts by 1 pixel but stays within the same 2×2
pooling window, the max-pooled output is UNCHANGED.

### Average Pooling

```
AvgPool(X)[i,j] = mean over window {X[i·s:i·s+k, j·s:j·s+k]}
```

Smooths the feature map rather than emphasizing peaks. Less commonly used in
hidden layers (loses sharp feature information), but standard for the FINAL
pooling before classification (Global Average Pooling, see below).

### Global Average Pooling (GAP)

```
GAP(X) = (1/(H×W)) Σᵢⱼ X[:, i, j]      ∈ ℝ^(C)   (collapses H×W → 1×1)
```

Used in modern architectures (ResNet, DenseNet, GoogLeNet) to replace the
final flatten+dense layers, drastically reducing parameters and making the
network agnostic to input image size (since GAP works for any H×W).

### Pooling vs Strided Convolution

Both reduce spatial dimensions. The difference:
```
Pooling:            fixed, non-learnable downsampling rule (max or average)
Strided convolution: learnable downsampling — network decides HOW to combine
                      and downsample simultaneously

Modern trend: many recent architectures (ResNet variants) REPLACE pooling
with strided convolutions for the learnable benefit, while keeping max-pool
only at a few strategic points (or eliminating it almost entirely).
```

---

## 11. Convolution as Matrix Multiplication (im2col)

GPUs are extremely optimized for matrix multiplication (via cuBLAS), but
convolution is naturally a sliding-window operation, not a matrix multiply.
The **im2col** ("image to column") transformation bridges this gap.

### The im2col Trick

```
Step 1: For each output position, extract the corresponding input patch
        (same size as the kernel) and FLATTEN it into a column vector.

Step 2: Stack all these column vectors into a matrix:
        X_col ∈ ℝ^(Kh·Kw·C_in × H_out·W_out)

Step 3: Flatten the kernel weights into a matrix:
        W_row ∈ ℝ^(C_out × Kh·Kw·C_in)

Step 4: A single matrix multiplication computes ALL output positions at once:
        Y = W_row @ X_col      ∈ ℝ^(C_out × H_out·W_out)

Step 5: Reshape Y back to (C_out, H_out, W_out)
```

**Trade-off:** im2col duplicates input data (each input pixel can appear in
multiple overlapping patches), trading increased memory usage for the ability
to use highly optimized GEMM (General Matrix Multiply) routines — this
trade-off is almost always worthwhile on modern GPU hardware, which is why
cuDNN and most deep learning framework backends use im2col-based (or
closely related Winograd/FFT-based) algorithms internally.

---

## Key Equations Summary

| Concept | Formula |
|---|---|
| Cross-correlation (DL "convolution") | (f⋆g)[n] = Σₘ f[n+m]g[m] |
| Conv layer parameters | Kh·Kw·C_in·C_out + C_out |
| Output size | N_out = ⌊(N+2P−K)/S⌋+1 |
| Receptive field growth | RF_L = RF_{L-1}+(K_L−1)·∏_{i<L}S_i |
| Max pooling | max over local window |
| Global avg pooling | (1/HW)Σᵢⱼ X[:,i,j] |
| im2col matmul | Y = W_row @ X_col |

