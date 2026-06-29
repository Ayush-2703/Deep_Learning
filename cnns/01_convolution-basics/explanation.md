# Code Explanation: Convolutions, Kernels, Pooling & Receptive Fields

**`implementation.py` walkthrough**

---

## 1. Section A & B — Convolution From Scratch

### Why Triple-Nested Loops, Not Vectorized NumPy

```python
for i in range(H_out):
    for j in range(W_out):
        hi, wj = i * stride, j * stride
        patch = x[hi:hi + Kh, wj:wj + Kw]
        out[i, j] = np.sum(patch * w) + bias
```

**Why use explicit loops here when NumPy excels at vectorization?**
This is intentionally the most literal, readable translation of the
mathematical definition `Y[i,j] = Σ X[hi+u, wj+v]·W[u,v] + b` — the goal in
this section is pedagogical transparency, not speed. Section G (`im2col`)
shows the vectorized, production-style approach that achieves identical
results via a single matrix multiplication. Keeping the two implementations
separate lets us verify the optimized version against the obviously-correct
slow version.

### Verifying Against PyTorch by Injecting Known Weights

```python
conv = nn.Conv1d(1, 1, kernel_size=len(w), stride=stride, padding=padding)
with torch.no_grad():
    conv.weight[:] = torch.tensor(w).view(1, 1, -1)
    conv.bias[:]   = torch.tensor([bias])
torch_out = conv(torch.tensor(x).view(1, 1, -1)).detach().numpy().flatten()
```

**Why manually overwrite `conv.weight` instead of just comparing random-init
PyTorch output to scratch output?**
`nn.Conv1d(1,1,kernel_size=5)` initializes its weight randomly. To verify our
scratch implementation computes the SAME function as PyTorch, both must
operate on the EXACT SAME kernel values. We inject our NumPy kernel `w`
directly into PyTorch's `.weight` tensor (inside a `torch.no_grad()` block,
since we're modifying a parameter in-place outside the optimizer's normal
update path), guaranteeing both implementations process identical inputs
with identical weights — so any output mismatch must come from a logic
bug, not from comparing apples to oranges.

### Shape Conventions: `(1, 1, -1)` for PyTorch

```python
torch.tensor(x).view(1, 1, -1)
```

**Why `(1, 1, -1)` and not just `x` directly?**
PyTorch's `nn.Conv1d` expects input shape `(batch, channels, length)`. Our raw
NumPy array `x` is shape `(L,)` — a 1D vector. `view(1, 1, -1)` reshapes it
to `(1, 1, L)`: batch size 1, 1 channel, original length preserved via `-1`
(infer this dimension automatically). Without this reshape, PyTorch would
raise a dimension-mismatch error, since it always expects the
batch+channel structure even for a single 1D signal.

### Multi-Channel Convolution: The Triple Sum

```python
for k in range(C_out):
    for i in range(H_out):
        for j in range(W_out):
            hi, wj = i * stride, j * stride
            patch = x[:, hi:hi + Kh, wj:wj + Kw]      # (C_in, Kh, Kw)
            out[k, i, j] = np.sum(patch * w[k]) + bias[k]
```

**Why does `patch * w[k]` correctly sum over channels too, with no explicit
channel loop?**
`patch` has shape `(C_in, Kh, Kw)` and `w[k]` (the k-th output filter) also has
shape `(C_in, Kh, Kw)`. NumPy's `*` performs element-wise multiplication
across ALL three dimensions simultaneously, and `np.sum()` with no `axis`
argument sums over every element — collapsing channels, height, AND width
in one call. This implicitly performs the channel-sum `Σ_c` from the
theory.md formula without writing a fourth nested loop, since `np.sum`
already iterates over every array dimension by default.

---

## 2. Section C — Classic Hand-Designed Kernels

### Building the Synthetic Test Image Procedurally

```python
def make_synthetic_image(size=64) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.float32)
    img[10:30, 10:30] = 1.0                       # white square
    for i in range(35, 55):
        for j in range(35, 55):
            if j - 35 <= (i - 35):
                img[i, j] = 0.6                      # triangular gradient region
```

**Why generate a synthetic image rather than loading a real one?**
This environment has no network access to image datasets or stock photo
sources (see network configuration — only PyPI/GitHub/crates.io domains are
allowed). A procedurally-generated image with clear geometric edges (a sharp
square boundary, a diagonal triangle edge, a diagonal line) is actually IDEAL
for demonstrating edge-detection kernels, since we know exactly where the
edges are and can visually confirm the kernels correctly highlight them —
real photographs would have noisier, less interpretable edge structure for
a first introduction to these kernels.

### Why `bias=0.0` for All Kernel Demos

```python
out = conv2d_scratch(img, k, bias=0.0, stride=1, padding=1)
```

Classic image-processing kernels (Sobel, Prewitt, sharpen, blur) are defined
purely as convolution operations without a bias term — the bias is a deep
learning addition for *learned* filters. Setting `bias=0.0` here keeps the
demonstration faithful to the classical kernels' original definitions.

### Why `padding=1` for All These 3×3 Kernels

Using `padding=1` with a 3×3 kernel and `stride=1` gives same-padding (output
size equals input size, per the formula in theory.md §5), which is the right
choice for visualization — we want the kernel-filtered output to be directly
comparable pixel-for-pixel against the original image without size
mismatches at the borders.

---

## 3. Section D — Output Size Formula Verification

### Constructing a Dummy Conv2d Just to Read Its Output Shape

```python
conv = nn.Conv2d(1, 1, kernel_size=K, stride=S, padding=P)
with torch.no_grad():
    dummy = torch.randn(1, 1, N, N)
    torch_out = conv(dummy).shape[-1]
```

**Why build a full `nn.Conv2d` layer just to check a shape, rather than using
a lighter-weight shape-calculation utility?**
This deliberately tests the EXACT same code path that a real training script
would use. If PyTorch's actual `Conv2d` forward pass ever produced a
different output size than our textbook formula predicts (e.g., due to a
subtle rounding difference or version-specific behavior change), this test
would catch it immediately — unlike testing against a separate "shape
calculator" function, which could itself silently drift out of sync with
PyTorch's real behavior. We don't care about the dummy data's VALUES (hence
`torch.randn`, never inspected) — only the resulting tensor's spatial
dimension.

### Why the "ResNet First-Layer Config" Is Highlighted

```python
(224, 7, 3, 2, "ResNet first-layer config"),
```

ResNet's very first layer (before any residual blocks) uses a 7×7 kernel,
stride 2, padding 3 on a 224×224 input — producing a 112×112 output. This is
a deliberately memorable real-world configuration (verified live to match
PyTorch exactly: `112 = 112 ✓`) that previews the architecture deep-dive
in Topic 2, grounding the abstract formula in a config actually used in a
landmark CNN.

---

## 4. Section E — Pooling From Scratch

### Demonstrating Translation Robustness Quantitatively

```python
shifted = np.roll(x, shift=1, axis=1)
max_orig = maxpool2d_scratch(x, 2, 2)
max_shift = maxpool2d_scratch(shifted, 2, 2)
diff_pct = np.mean(max_orig != max_shift) * 100
```

**Why use `np.roll` (circular shift) rather than a simple slice-based shift?**
`np.roll` shifts every column by 1 position and wraps the last column around
to the front — this guarantees the shifted array has the SAME shape as the
original (no need to handle a now-missing edge column), which keeps the
subsequent `maxpool2d_scratch` call and comparison straightforward. A
non-circular shift would either change the array's shape or require padding
decisions that complicate the comparison without adding to the pedagogical point.

**Why does the live result show 62.5% of outputs CHANGING, seemingly
contradicting the "translation robustness" claim in theory.md?**
This is a precise and honest result, not a contradiction: pooling provides
robustness to shifts that stay WITHIN a single pooling window (e.g., a
feature shifting by 1 pixel inside an aligned 2×2 block leaves the max
unchanged), but a 1-pixel shift will often move a value ACROSS a window
boundary into an adjacent pooling region, which DOES change that window's
max. With random Gaussian noise input (no actual coherent "features" to
track), nearly every window boundary is crossed somewhere, producing a high
change percentage. The robustness claim is about ROBUSTNESS TO SMALL SHIFTS
RELATIVE TO FEATURE SIZE, not perfect invariance to any shift — this
distinction is exactly why the theory.md text says "**partially** absorbed,"
not "fully invariant."

---

## 5. Section F — Receptive Field Growth

### The All-Ones Kernel Trick for Empirical Measurement

```python
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.ones_(m.weight)    # all-ones kernel: any non-zero contributes
```

**Why initialize every kernel weight to exactly 1.0, rather than random
values, for this specific experiment?**
We want to measure WHICH input pixels can possibly influence a given output
unit — a purely STRUCTURAL question about the network's connectivity pattern,
not about the magnitude of any particular learned weight. With random
weights, a gradient contribution from a connected pixel could (by unlucky
coincidence) be exactly zero due to cancellation, making it look like that
pixel has NO influence when it actually does (just with zero net effect at
this particular random initialization). All-ones weights guarantee every
structurally-connected pixel contributes a strictly positive, non-cancelling
amount to the gradient, so the "non-zero gradient" test cleanly captures
connectivity, not coincidental cancellation.

### Why Backprop From a Single Output Pixel

```python
x = torch.zeros(1, 1, input_size, input_size, requires_grad=True)
out = model(x)
center = out.shape[-1] // 2
out[0, 0, center, center].backward()
```

**Why pick the CENTER output pixel specifically?**
Output pixels near the image border have a receptive field that gets
"clipped" by the image boundary (some of their theoretical receptive field
falls outside the actual input, since we padded with zeros — those padded
positions don't count as real influenceable input pixels). The center pixel
is far enough from any border that its full theoretical receptive field fits
entirely within the input image, giving a clean, unclipped measurement that
directly matches the formula's prediction.

### Why `requires_grad=True` on the INPUT, Not the Weights

This is the reverse of typical training, where we backprop to compute weight
gradients. Here we backprop to the INPUT specifically because we want to know
"`which input pixels affect this output`," which is exactly what
`x.grad` answers: a non-zero entry in `x.grad[i,j]` means pixel `(i,j)`
influenced the chosen output pixel; a zero entry means it didn't. This is a
clever repurposing of the same backward-pass machinery for a structural
analysis question rather than an optimization update.

### Live Result: Exact Formula-Empirical Match

```
[Stride=1, 3×3 ×6 layers]:   Formula=[1,3,5,7,9,11,13]  Empirical=13  ✓
[Stride=2, 3×3 ×4 layers]:   Formula=[1,3,7,15,31]       Empirical=31  ✓
```

Both cases show PERFECT agreement between the closed-form recurrence formula
and the empirically measured receptive field via gradient backpropagation —
strong confirmation that the formula in theory.md §9 correctly models how
PyTorch's actual convolution operations compose.

---

## 6. Section G — im2col

### Building the Column Matrix With a Strided Slice Trick

```python
col = np.zeros((C_in, Kh, Kw, H_out, W_out), dtype=x.dtype)
for i in range(Kh):
    for j in range(Kw):
        col[:, i, j, :, :] = x[:, i:i+stride*H_out:stride, j:j+stride*W_out:stride]
```

**Why loop over `(i,j)` — the kernel's spatial offsets — rather than over
output positions `(H_out, W_out)`?**
This is a key efficiency insight: instead of extracting one small patch per
output position (which is what `conv2d_multichannel_scratch` does, with
`H_out × W_out` iterations), we instead loop over the much smaller
`Kh × Kw` kernel positions (typically 3×3=9 or 5×5=25 iterations) and use a
SINGLE strided NumPy slice to grab ALL output positions' contribution for that
kernel offset simultaneously. The slice `x[:, i:i+stride*H_out:stride, ...]`
uses NumPy's native stride-slicing to extract every "stride-th" pixel
starting at offset `i` — exactly the pixels needed across all output
positions for kernel row `i`. This trades `H_out*W_out` (potentially
thousands) of small extractions for just `Kh*Kw` (often <50) large
vectorized slice operations — the core reason im2col enables fast,
GPU-friendly convolution.

### Why Reshape Order Matters: `(C_out, C_in*Kh*Kw)`

```python
W_row = w.reshape(C_out, C_in*Kh*Kw)
```

**Why must `w`'s reshape order exactly match `col`'s construction order?**
`w` has shape `(C_out, C_in, Kh, Kw)`. NumPy's `.reshape()` flattens dimensions
in row-major (C) order by default — meaning the LAST dimension varies
fastest. So flattening `(C_in, Kh, Kw)` into one axis produces an ordering
where, for a fixed `C_in` index, `Kw` varies fastest, then `Kh`, then `C_in`
slowest. Our `col` array was deliberately constructed with axis order
`(C_in, Kh, Kw, H_out, W_out)` — matching this exact same `(C_in, Kh, Kw)`
nesting — so that when both are flattened, corresponding entries line up
correctly for the matrix multiplication `W_row @ X_col` to compute the right
dot products. A mismatched axis order between the two reshapes would silently
produce numerically plausible-looking but WRONG results (the matmul would
still execute without error, just compute garbage) — this is a classic
im2col implementation pitfall, avoided here by keeping both constructions
consistent.

### Live Verification

```
Direct loop-based conv shape: (8, 10, 10)
im2col matmul-based shape:    (8, 10, 10)
Outputs match: True
```

The two completely different implementation strategies — naive triple-nested
loops vs. a single big matrix multiplication — produce numerically identical
results, confirming the im2col reshape logic is correct. This single matmul
is exactly the operation that GPU-accelerated cuDNN performs internally
(at a much larger scale) for every convolution call in PyTorch.

---

## 7. Section H — Parameter Counting

### Why the Same `+ out_units` / `+ C_out` Bias Term in Both Formulas

```python
dense_params = size*size*c_in*out_units + out_units
conv_params  = 3*3*c_in*out_units + out_units
```

Both formulas include exactly one bias parameter per output unit/channel —
this term is IDENTICAL between dense and conv layers (both have the same
number of output units in this comparison), so it doesn't affect the RATIO
much for large layers, but is included for formula completeness and
correctness (a layer with zero learnable bias would technically be a
different, less expressive operation).

### Live Result Interpretation

```
224x224x3 | Dense: 150,529,000 | Conv3x3: 28,000 | Ratio: 5376×
```

At ImageNet scale, a single dense layer mapping a flattened image to 1000
units would require over 150 MILLION parameters — more than some entire
modern CNN architectures use for their ENTIRE network (e.g., ResNet-18 has
~11M total parameters). This single comparison is the clearest possible
numerical justification for why convolutional layers, not dense layers,
are the foundation of image-processing neural networks.

---

## Pitfalls Avoided

| Pitfall | Fix Applied |
|---|---|
| Comparing scratch vs PyTorch with different random weights | Inject identical weights into both before comparing |
| Wrong PyTorch input shape causing cryptic errors | Explicit `.view(1,1,-1)` / `.view(1,1,H,W)` reshapes |
| Random-weight gradient cancellation hiding true connectivity | All-ones kernel init for receptive field experiment |
| Measuring receptive field at a border pixel (clipped by input edge) | Backprop from the CENTER output pixel |
| Mismatched flatten order between kernel and im2col column matrix | Consistent `(C_in, Kh, Kw, ...)` axis ordering throughout |
| No real image datasets available (no internet access) | Procedurally generated synthetic test image with known edges |

---
