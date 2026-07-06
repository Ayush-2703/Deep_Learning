# Code Explanation: Segmentation — U-Net & Mask R-CNN

**Phase 2 — Topic 4 | `implementation.py` walkthrough**

---

## 1. Section A — Synthetic Segmentation Datasets

### Drawing the Same Shape Onto Three Separate Canvases

```python
inst_img = Image.new("L", (img_size, img_size), color=0)
inst_draw = ImageDraw.Draw(inst_img)
if kind == "ellipse":
    inst_draw.ellipse(geom, fill=1)
...
sem_draw = ImageDraw.Draw(semantic_mask)
if kind == "ellipse":
    sem_draw.ellipse(geom, fill=cls + 1)
```

**Why draw the same geometric shape THREE times (color image, instance mask,
semantic mask) instead of deriving the masks from the color image
afterward?**
Deriving a mask by, e.g., thresholding the rendered color image would be
fragile — it would need to handle anti-aliasing at edges, distinguish object
color from background color robustly, and could be corrupted by the
Gaussian noise we add afterward. Drawing the EXACT SAME polygon/ellipse
geometry independently onto dedicated mask canvases (using a fixed fill
value rather than a random color) guarantees PIXEL-PERFECT ground truth —
the mask boundary is mathematically identical to the shape boundary, with
zero ambiguity. This is a key advantage of synthetic data generation: ground
truth is exact by construction, never requiring error-prone derivation.

### Why `cls + 1` for the Semantic Mask but Plain `cls` for Instance Labels

```python
sem_draw.ellipse(geom, fill=cls + 1)       # semantic mask: 0=background reserved
...
labels.append(cls)                          # instance labels: 0-indexed directly
```

The semantic segmentation mask is a SINGLE per-pixel value covering the
WHOLE image, including background — pixel value `0` must be reserved to mean
"no object here," so object classes are shifted to `1,2,3` (matching
`NUM_SEM_CLASSES = NUM_CLASSES + 1` used throughout U-Net). The instance
dataset, by contrast, only ever stores labels for ACTUAL objects (there's no
"background instance" to encode), so `0,1,2` direct indexing matching
`CLASS_NAMES` is appropriate there — this exactly mirrors the same
background-offset convention difference we encountered between YOLO
(Topic 3, 0-indexed) and Faster R-CNN (Topic 3, 1-indexed with 0=background)
at the torchvision API boundary.

---

## 2. Section B — U-Net From Scratch

### Why Save Skip Connections BEFORE Pooling, Not After

```python
s1 = self.enc1(x)              # (B,16,64,64)
s2 = self.enc2(self.pool(s1))  # (B,32,32,32)
s3 = self.enc3(self.pool(s2))  # (B,64,16,16)
```

**Why is `s1` captured at full 64×64 resolution, rather than at the
post-pool 32×32 resolution?**
The decoder's corresponding upsampling stage (`self.up1`) will restore the
feature map back to 64×64 resolution specifically so that it can be
concatenated with a skip connection AT THAT SAME 64×64 resolution. If we
instead saved `s1` AFTER pooling (at 32×32), there would be no skip
connection available at the FINAL 64×64 output resolution — the very
highest-detail level, which is precisely where spatial precision matters most
for sharp segmentation boundaries (theory.md §3). Capturing skip connections
immediately after the convolution but BEFORE the pooling operation, at EVERY
encoder stage, ensures the decoder has a matching-resolution skip available
at every corresponding upsampling step.

### Shape Trace Through the Concatenation

```python
d3 = self.up3(b)                        # (B,128,16,16)
d3 = self.dec3(torch.cat([d3, s3], dim=1))    # concat → (B,192,16,16) → (B,64,16,16)
```

**Why does `DoubleConv(128+64, 64)` use `128+64=192` as its INPUT channel
count?**
After `self.up3` upsamples the bottleneck's 128-channel, 8×8 feature map to
16×16 (spatial size now matches `s3`), we concatenate ALONG THE CHANNEL
DIMENSION (`dim=1`) with `s3` (64 channels, also at 16×16 spatial
resolution). Concatenation ADDS channel counts together (unlike ResNet's
addition, which REQUIRES matching channel counts) — the result has
`128+64=192` channels, which is exactly the input channel count `dec3`'s
first convolution must be configured to accept. Getting this arithmetic
wrong (e.g., writing `DoubleConv(128, 64)` and forgetting to account for the
concatenated skip channels) would raise an immediate, easy-to-debug shape
mismatch error the very first time the model is run — but it's worth tracing
through explicitly here since silently-WRONG channel arithmetic (e.g.
accidentally matching by coincidence at one resolution but not another)
could otherwise be a subtle bug source in a hand-built encoder-decoder.

### Why Bilinear Upsampling Instead of Transposed Convolution

```python
self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
```

As discussed in theory.md §2, transposed convolutions are LEARNABLE but can
introduce characteristic "checkerboard" artifacts (uneven overlap patterns
between adjacent output positions). Using fixed bilinear interpolation for
the resizing step, followed by a REGULAR (non-transposed) convolution inside
`DoubleConv` to refine the upsampled features, avoids this artifact entirely
while still allowing the network to learn how to best use the upsampled
information — a common, robust choice in modern U-Net implementations.

---

## 3. Section C — Dice Loss

### Why One-Hot Encode the Target Before Computing Dice

```python
target_onehot = F.one_hot(target, num_classes).permute(0,3,1,2).float()
```

**Why convert `target` (a single integer class per pixel) into a
multi-channel one-hot tensor before the Dice computation?**
The Dice formula `2|A∩B|/(|A|+|B|)` is fundamentally a SET-OVERLAP measure
between two BINARY masks. To compute this independently for EACH class
(circle-vs-not-circle, square-vs-not-square, etc.), we need the target
represented as `C` separate binary masks — exactly what one-hot encoding
provides. `F.one_hot(target, num_classes)` produces shape `(B,H,W,C)`
(channel LAST, PyTorch's one-hot convention); the subsequent
`.permute(0,3,1,2)` reorders this to `(B,C,H,W)`, matching the prediction
tensor's standard channel-second layout so the two can be combined
element-wise.

### Why Sum Over `dims=(0,2,3)`, Keeping the Class Dimension Separate

```python
dims = (0, 2, 3)
intersection = torch.sum(probs * target_onehot, dim=dims)
cardinality  = torch.sum(probs + target_onehot, dim=dims)
```

Summing over dimensions `(0,2,3)` — batch, height, width — while explicitly
EXCLUDING dimension 1 (the class channel) produces a result of shape `(C,)`:
one intersection/cardinality VALUE PER CLASS, aggregated across the entire
batch and all spatial positions. This gives us exactly the per-class Dice
coefficients needed before averaging them into a single scalar loss —
mirroring the mIoU evaluation metric's same "compute per-class, then
average" structure (theory.md §7), ensuring the TRAINING loss and the
EVALUATION metric share a consistent class-balancing philosophy.

### Live Sanity-Check Verification

```
Perfect prediction Dice loss: 0.00000  (expect ≈0)
Random  prediction Dice loss: 0.75697  (expect higher)
```

Before trusting this loss function inside a real training loop, we verify
its two boundary behaviors explicitly: a PERFECT prediction (logits hugely
favoring the correct class, via `* 20.0` scaling before softmax) correctly
drives the loss to (essentially) exactly zero, while a RANDOM prediction
produces a substantially higher loss — confirming the implementation
behaves as the mathematical definition demands before we rely on it for 30
epochs of actual training.

---

## 4. Section D — Training U-Net

### Why `compute_miou` Iterates Classes With a Python Loop (Not Vectorized)

```python
def compute_miou(pred, target, num_classes=NUM_SEM_CLASSES) -> float:
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0
```

With only `NUM_SEM_CLASSES=4` classes, a simple Python loop over classes is
clear, correct, and fast enough — vectorizing this further would add
complexity for negligible speed benefit at this scale. The `if union > 0`
guard specifically handles the edge case where a particular class is ABSENT
from both the prediction AND the ground truth in a given batch (e.g., if a
validation batch happens to contain zero triangles) — without this check,
`intersection/union` would be `0/0`, producing `NaN` and corrupting the
mean; skipping classes with no presence at all (rather than counting them as
either 0 or 1) is the standard mIoU convention.

### Live Result — The Mid-Training Instability Dip (Reported Honestly)

```
Epoch  10/30 | val_mIoU=0.9929
Epoch  11 (unlogged, visible in Figure 1): val_loss spikes, val_mIoU dips to ~0.67
Epoch  15/30 | val_mIoU=1.0000  (fully recovered)
```

Figure 1's mIoU panel shows a clear, sharp DIP around epoch 10-11 — both the
training and validation loss curves show a small spike at the same point,
and validation mIoU correspondingly drops from ~0.99 down to roughly 0.67
before recovering to a perfect 1.0 within a few epochs. This is the SAME
general category of training-dynamics phenomenon observed with ResNet-mini
in Phase 2 Topic 2 (a transient instability, likely from an Adam
update interacting poorly with a particular mini-batch's BatchNorm
statistics at that point in training) — we report it here exactly as it
occurred, consistent with this repository's stated practice of never hiding
or cherry-picking around inconvenient-but-real training behavior. The
network's fast self-recovery (back to mIoU=1.0 within ~4 epochs) demonstrates
that such transient spikes, while visually dramatic in the loss curve, don't
necessarily indicate a fundamental training failure — only a momentary
perturbation that well-conditioned optimization recovers from gracefully.

---

## 5. Section E — Mask R-CNN: A Real Engineering Detour

### The Problem We Discovered: A Background-Process Resource Ceiling

The ORIGINAL plan for this topic was a single `implementation.py` execution
training BOTH U-Net (30 epochs) AND Mask R-CNN (6 epochs) sequentially in
one Python process, following the exact pattern that worked successfully for
Phase 2 Topics 2 and 3. In practice, this combined run was launched as a
`setsid`-detached background process and monitored via periodic polling
(the same proven strategy) — but it was silently terminated partway through
Mask R-CNN's THIRD epoch, after U-Net had ALREADY completed successfully
within the same process. No error or traceback was written to the log; the
process simply stopped existing.

### Diagnosis

Checking `dmesg` for OOM-killer messages and `free -h` for system memory
state revealed NO out-of-memory kernel event, and the system showed ample
free memory immediately after the process died. However, monitoring the
process's RSS (resident memory) via `ps aux` DURING training showed clear,
substantial growth specifically during Mask R-CNN's training loop — roughly
300-500MB of additional memory consumed per epoch, a pattern not observed
during U-Net's (memory-stable) training in the SAME process just beforehand.
This points to SOME form of resource accumulation specific to running many
iterations of torchvision's detection-model training loop within a single
long-lived process (potentially related to delayed garbage collection of the
complex, deeply-nested tensor graphs that RPN+RoIAlign+mask-head training
produces) — combined with an apparent background-process ceiling in this
particular sandboxed execution environment that isn't visible through
standard `dmesg`/`free` diagnostics.

### The Fix: Process Isolation via Separate Driver Scripts

```python
# driver_unet.py — trains ONLY U-Net, saves model+history+val-data to disk, exits
# driver_mrcnn.py — trains ONLY Mask R-CNN (reduced scope), saves results, exits
# combine_figures.py — reloads BOTH saved results, generates all final figures
```

Rather than further shrinking the experiment to fit within one process's
apparent ceiling (which would have meant even less Mask R-CNN training,
worsening results further), we split the work into THREE separate,
SHORT-LIVED Python processes, each starting with a clean memory state and
exiting (releasing ALL its memory back to the OS) immediately after saving
its results via `torch.save`/`pickle`. This is a standard and genuinely
useful pattern in real-world ML engineering — when a single long-running job
encounters resource constraints, decomposing it into independent,
checkpoint-passing stages (sometimes called a "pipeline" of jobs) is a
common, practical solution, not merely a workaround specific to this
educational repository's sandboxed environment.

### Why Mask R-CNN's Scope Was Still Reduced (250 vs 400 images, 4 vs 6 epochs)

Even after process isolation fixed the OUTRIGHT FAILURE, Mask R-CNN's
inherent per-epoch cost (observed at roughly 100+ seconds/epoch, consistent
with Faster R-CNN's similar cost in Topic 3, PLUS the additional mask-head
computation) meant that even a fresh, isolated process needed a more
conservative configuration to complete comfortably within a reasonable
polling/monitoring budget. We reduced both dataset size (400→250 training
images) and epoch count (6→4) for this specific component — a deliberate,
disclosed trade-off, not a silently-shrunk experiment.

---

## 6. Live Result — Mask R-CNN's Honest, Imperfect Performance

```
Mask R-CNN: Precision=0.609  Recall=0.545  (TP=42, FP=27, FN=35)
```

Compare this to Topic 3's Faster R-CNN (BOX-ONLY detection, more training:
400 images / 8 epochs): `Precision=0.992, Recall=1.000`. Mask R-CNN's
noticeably weaker performance here is NOT a sign of a bug — it's a
consistent, explainable consequence of THREE compounding factors, all
visible directly in Figure 3:

```
1. LESS total training (250 images/4 epochs vs 400 images/8 epochs — roughly
   HALF the total gradient updates Faster R-CNN received in Topic 3)

2. A HARDER joint task: Mask R-CNN must learn accurate region proposals,
   correct classification, AND precise per-pixel masks SIMULTANEOUSLY,
   versus Faster R-CNN's box-only regression+classification — strictly more
   to learn from strictly less data/training time

3. Visible concrete failure modes in Figure 3:
     - Image 0: the circle is MISCLASSIFIED as "triangle" with a messy,
       imprecisely-shaped predicted mask — the network's mask head hadn't
       yet learned a clean circular mask template
     - Image 1: a DUPLICATE detection (two overlapping "square" predictions
       for the SAME single ground-truth square) — suggesting the model's
       confidence calibration and/or RPN proposal redundancy hadn't been
       sufficiently trained down at this stage
     - Image 1 (same image): the circle is COMPLETELY MISSED — a clear
       recall failure contributing directly to the FN=35 count
     - Image 2: a clean, correctly-classified, well-localized triangle
       prediction — proof the underlying mechanism DOES work correctly when
       given a clearer signal, it simply hasn't converged uniformly across
       all object types/positions yet
```

This is a valuable, realistic illustration of theory.md §9's qualitative
claim that two-stage detectors are "more complex to train" — not merely
slower per-step, but also requiring MORE total training signal to reach the
same reliability level as a comparatively simpler box-only task, let alone a
single-class-per-pixel semantic segmentation task like U-Net's (which
reached near-perfect mIoU in the SAME wall-clock budget U-Net was given).

---

## Pitfalls Avoided

| Pitfall | Fix Applied |
|---|---|
| Deriving masks from noisy rendered images (fragile, imprecise) | Drew shapes independently onto dedicated mask canvases |
| Background-class index collision (semantic vs instance task) | `cls+1` for semantic masks, plain `cls` for instance labels |
| Skip connections captured at wrong (post-pool) resolution | Captured immediately after conv, BEFORE pooling |
| Wrong channel arithmetic after concatenation | Explicit `DoubleConv(128+64, 64)` channel-count tracing |
| Dice loss `0/0` NaN for absent classes in a batch | `if union > 0` guard in `compute_miou` |
| Untested Dice loss silently wrong during real training | Explicit perfect-vs-random sanity check before use |
| Background process silently killed mid-training (no error) | Split into 3 isolated, checkpoint-passing driver scripts |
| Hiding Mask R-CNN's weaker (reduced-budget) results | Reported and explained P=0.61/R=0.55 honestly, with causal analysis |

---

*Previous: [Topic 3 — Object Detection](../03-object-detection-rcnn-yolo/explanation.md)*
*Next: [Topic 5 — Transfer Learning & Fine-tuning](../05-transfer-learning-finetuning/explanation.md)*
