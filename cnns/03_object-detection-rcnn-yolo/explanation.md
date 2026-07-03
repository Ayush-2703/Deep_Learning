# Code Explanation: Object Detection — Faster R-CNN & YOLO

**`implementation.py` walkthrough**

---

## 1. Section A — Synthetic Detection Dataset

### Rejection Sampling to Avoid Ambiguous Overlaps

```python
too_close = any(abs(cx - pcx) < (r + pr + 4) and abs(cy - pcy) < (r + pr + 4)
                for pcx, pcy, pr in placed)
if too_close:
    continue
```

**Why reject overlapping placements instead of allowing them?**
Real object detection datasets DO contain overlapping objects, and a
production-grade detector must handle this. For this educational
implementation, however, we deliberately keep objects well-separated
(`+4` pixel margin beyond their combined radii) so that ground-truth boxes
are UNAMBIGUOUS: every grid cell in YOLO's encoding scheme, and every anchor
in Faster R-CNN's matching scheme, has a clear single best-matching object
with no contested boundary cases. This isolates the core detection
mechanics (box regression, classification, confidence) from the
additional complexity of occlusion handling, which is a reasonable scoping
choice for a from-scratch educational implementation.

### Why a Custom `collate_fn` Is Required

```python
def detection_collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    boxes  = [b[1] for b in batch]
    labels = [b[2] for b in batch]
    return images, boxes, labels
```

**Why can't we just use PyTorch's default `DataLoader` collation?**
The default collate function tries to `torch.stack` every field across the
batch — this works fine for `images` (all the same fixed shape `(3,64,64)`),
but FAILS for `boxes`/`labels`, since different images in the SAME batch
have DIFFERENT NUMBERS of objects (1 or 2 in our dataset). You cannot stack
tensors of shape `(1,4)` and `(2,4)` into a single regular tensor. The
custom collate function sidesteps this by keeping `boxes`/`labels` as plain
Python LISTS of variable-length tensors (one tensor per image), while still
stacking the fixed-shape `images` normally — this is the standard PyTorch
pattern for object detection dataloaders, used identically by torchvision's
own detection reference training scripts.

---

## 2. Section B — IoU & NMS From Scratch

### Why Verify Against `torchvision.ops`, Not Just Trust the Formula

```python
iou_torch_mat = torchvision_box_iou(...)
match_iou = np.allclose(iou_scratch_mat, iou_torch_mat, atol=1e-5)
assert match_iou
```

Both IoU and NMS are used PERVASIVELY downstream — inside YOLO's evaluation
pipeline, inside the detection-quality metric in Section F, and conceptually
underlying everything Faster R-CNN's RPN does internally (even though we
don't reimplement the RPN's anchor-matching logic by hand, since we use
torchvision's built-in implementation for that part). A subtle bug in our
hand-written IoU (e.g., an off-by-one in the intersection boundary
condition) would silently corrupt evaluation metrics and the NMS-based
post-processing without any visible crash — asserting exact agreement with
the well-tested `torchvision.ops` reference implementation catches this
category of bug immediately, before it can propagate into misleading
downstream results.

### The Dtype Bug We Caught and Fixed

```python
scores_nms = np.array([0.95, 0.90, 0.85, 0.30], dtype=np.float32)
```

During development, this line originally omitted `dtype=np.float32`. NumPy's
default float dtype is `float64`, while the `boxes_nms` array was explicitly
`float32`. When both were converted to PyTorch tensors and passed to
`torchvision_nms`, the function raised `RuntimeError: dets should have the
same type as scores` — torchvision's NMS kernel requires both tensors to
share an identical dtype. This is a common, easy-to-miss bug whenever NumPy
arrays with implicit (default) dtypes are mixed with explicitly-typed
arrays in the same computation; we keep the explicit `dtype=np.float32` fix
in place as a concrete, real example of why mixed-precision pitfalls matter
in practice, not just in theory.

### NMS Algorithm: The Iterative Reduction

```python
order = np.argsort(scores)[::-1]
keep = []
while len(order) > 0:
    current = order[0]
    keep.append(int(current))
    if len(order) == 1:
        break
    rest = order[1:]
    ious = np.array([iou_scratch(boxes[current], boxes[r]) for r in rest])
    order = rest[ious < iou_threshold]
```

**Why `rest[ious < iou_threshold]` rather than explicitly removing
high-IoU boxes?**
This is a boolean-mask filtering idiom: `ious < iou_threshold` produces a
boolean array marking which remaining boxes have LOW overlap with the
just-kept box (i.e., are NOT duplicates). Indexing `rest` with this boolean
mask keeps only those low-overlap boxes for the NEXT iteration — functionally
identical to "removing all boxes with IoU≥threshold," but expressed as a
positive selection rather than an explicit removal, which is the more
idiomatic and less error-prone NumPy pattern (avoiding in-place list
mutation while iterating).

### Live Verification Result

```
NMS scratch keeps indices: [0, 2]  (boxes A,C expected)
NMS torchvision keeps:     [0, 2]
Match: True
```

Exactly matches the hand-worked example from theory.md §4 (box B is
suppressed as a duplicate of A, box D is suppressed as a duplicate of C) —
both confirming our scratch implementation's correctness AND giving a
concrete, traceable example connecting the abstract algorithm description to
actual executed code.

---

## 3. Section D — YOLO-mini (Fully From Scratch)

### Why Sigmoid on ALL Outputs (A Deliberate Deviation From YOLOv1)

```python
pred_sig = torch.sigmoid(pred)
```

The original YOLOv1 paper applies sigmoid only to some outputs and leaves
width/height as raw (unbounded) linear regression targets. In this
implementation, we apply sigmoid to EVERY output channel — objectness, x, y,
w, h, AND class scores — because every one of our targets is, by
construction, a value in `[0,1]` (offsets within a cell, sizes normalized by
image dimensions, one-hot class indicators). Bounding the network's raw
output through sigmoid BEFORE comparing to these `[0,1]`-ranged targets
gives more stable, better-behaved gradients early in training (avoiding
wild raw-logit values that would need to coincidentally land near the right
magnitude) — a standard and well-documented practical adaptation also seen
in later YOLO versions (v2 onward) which similarly bound coordinate
predictions through sigmoid/logistic transforms.

### Target Encoding: Why `min(int(cx // stride), grid_size - 1)`

```python
col = min(int(cx // stride), grid_size - 1)
row = min(int(cy // stride), grid_size - 1)
```

**Why the `min(..., grid_size-1)` clamp?**
If an object's center coordinate `cx` is EXACTLY at the image's right/bottom
edge (`cx = IMG_SIZE`, i.e., `64`), then `cx // stride = 64 // 8 = 8` — but
valid grid indices only run from `0` to `7` (`grid_size - 1`). Without
clamping, this edge case would produce an out-of-bounds index and crash with
an `IndexError` during training. Since our shape generator places object
centers strictly inside the image with margin (see Section A), this edge
case never actually triggers in practice with our synthetic data — but the
clamp is defensive coding that guards against the theoretical boundary
condition regardless, which is good practice for any production-intended
target-encoding function.

### Why Confidence Target Is Always 1.0 for Object Cells (Simplification)

```python
target[0, row, col] = 1.0
```

Per theory.md §7, YOLO's confidence is formally defined as
`P(object) × IoU(predicted_box, ground_truth_box)` — meaning the IDEAL
target confidence depends on how well the CURRENT prediction's box aligns
with the ground truth, not a fixed constant. With our design choice of
`B=1` (exactly one predicted box per cell, rather than the original
paper's `B=2`), the single predicted box for an object-containing cell IS by
definition the "responsible" predictor — there's no competition between
multiple candidate boxes within the cell to resolve via IoU-based selection.
We use the simplified fixed target `1.0` (rather than computing a
live IoU-based target during each forward pass) because it is both simpler
to implement correctly and, for `B=1`, an entirely standard simplification
used in many introductory YOLO implementations — the network still learns
correct localization through the separate coordinate-regression loss terms;
only the confidence SIGNAL is simplified from "weighted by current box
accuracy" to "binary object presence indicator."

### The `clamp` Before `sqrt` in Width/Height Loss

```python
pred_wh_sqrt   = torch.sqrt(torch.clamp(pred_sig[:,3:5], min=1e-6))
target_wh_sqrt = torch.sqrt(torch.clamp(target[:,3:5], min=1e-6))
```

**Why clamp to `1e-6` rather than `0` before taking the square root?**
Even though `pred_sig` (after sigmoid) is mathematically guaranteed to be in
`(0,1)`, floating-point underflow can occasionally produce values
indistinguishable from exactly `0.0` for extremely negative pre-sigmoid
logits early in training (before the network has learned sensible weights).
`torch.sqrt(0.0)` returns `0.0` safely, but `torch.sqrt` of a tiny NEGATIVE
number (which can occur due to floating-point rounding even when the
"true" value should be a tiny positive number) returns `NaN`, silently
poisoning the entire loss and all subsequent gradients. Clamping to a tiny
positive floor (`1e-6`) eliminates this edge case entirely while having
negligible effect on the loss value for any normal (non-degenerate)
prediction.

### Live Result — Clean, Stable Convergence

```
Epoch   1/40 | train_loss=4.384 | val_loss=4.180
Epoch  40/40 | train_loss=0.012 | val_loss=0.078
```

Unlike some of Phase 2 Topic 2's architectures (which showed visible
training instability spikes), YOLO-mini's loss curve (Figure 1) is smooth
and monotonic throughout all 40 epochs — both train and validation loss
decrease together with no divergence, indicating the sigmoid-bounded output
design and clamped square-root loss successfully avoided the numerical
instabilities that can plague naive from-scratch detector implementations.

---

## 4. Section E — Faster R-CNN via torchvision

### Why a Custom Backbone Needs an Explicit `.out_channels` Attribute

```python
class TinyBackbone(nn.Module):
    def __init__(self, out_channels=64):
        super().__init__()
        self.body = nn.Sequential(...)
        self.out_channels = out_channels    # REQUIRED attribute
```

**Why does torchvision's `FasterRCNN` constructor specifically require
this attribute on the backbone module?**
Internally, `FasterRCNN` needs to know the CHANNEL DIMENSIONALITY of
whatever feature map the backbone produces, in order to correctly size the
RPN's internal convolutional layers (which take the backbone's feature map
as input) and the RoI head's input projection. Rather than requiring the
user to pass this number as a SEPARATE constructor argument (which could
drift out of sync if the backbone is later modified), torchvision's API
design queries it directly from the backbone object itself via
`backbone.out_channels` — our `TinyBackbone` must therefore expose this as
a plain Python attribute (not a method) for `FasterRCNN`'s constructor to
read successfully.

### Why `min_size=IMG_SIZE, max_size=IMG_SIZE`

```python
model = FasterRCNN(
    backbone, num_classes=num_classes,
    rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler,
    min_size=IMG_SIZE, max_size=IMG_SIZE,
)
```

**Why explicitly constrain the input resizing range to a single fixed
value?**
By default, torchvision's `FasterRCNN` automatically RESIZES input images to
fall within a `[min_size, max_size]` range (defaults are tuned for
ImageNet-scale ~800px images) before feeding them to the backbone — this
internal resizing is usually helpful for handling variable-sized real-world
images, but for our deliberately small, FIXED-size `64×64` synthetic images,
allowing the default resizing behavior would upscale our images
significantly (toward the default ~800px range), wasting computation and
distorting the carefully-tuned anchor scales we specified (which were
chosen relative to OUR 64×64 image scale, not an 800px scale). Setting both
`min_size` and `max_size` to exactly `IMG_SIZE` disables this resizing
behavior, keeping images at their native small resolution throughout the
pipeline.

### Why Labels Are 1-Indexed for Faster R-CNN, But 0-Indexed for YOLO

```python
targets = [{"boxes": b.to(DEVICE), "labels": (l+1).to(DEVICE)}
          for b, l in zip(boxes_list, labels_list)]
```

This is a torchvision-specific convention: class label `0` is RESERVED to
mean "background" (no object) within the RPN/RoI-head's internal
classification logic — every REAL object class must be indexed starting
from `1`. Our dataset internally stores labels as `0,1,2` (matching
`CLASS_NAMES = ["circle","square","triangle"]`), so we add `1` when
constructing Faster R-CNN's target dictionaries, and correspondingly
SUBTRACT `1` back when decoding predictions (`labels - 1` in
`faster_rcnn_predict`) to return to our dataset's natural 0-indexed
convention for fair comparison against YOLO's predictions, which has no
such background-class offset (YOLO's grid cells encode "no object" via the
separate objectness/confidence channel, not via a reserved class index).

### Why the Training Loop Sums `loss_dict.values()` Directly

```python
loss_dict = model(images, targets)
loss = sum(loss_dict.values())
```

When called in TRAINING mode with both `images` AND `targets` provided,
torchvision's `FasterRCNN.forward()` does NOT return predicted boxes at
all — it returns a dictionary of the four loss components described in
theory.md §6 (`loss_classifier`, `loss_box_reg`, `loss_objectness`,
`loss_rpn_box_reg`). Summing these four scalar tensors gives exactly the
combined multi-task loss `L = L_rpn_cls + L_rpn_box + L_head_cls +
L_head_box` from the theory, ready for a single `.backward()` call —
torchvision handles all of the anchor-matching, positive/negative sampling,
and per-component loss computation internally, which is precisely why using
the library's built-in implementation (rather than re-deriving this from
absolute scratch) is the standard, practical engineering choice for this
architecture.

### Live Result — Faster R-CNN's Per-Epoch Cost

```
Faster R-CNN training complete in 776.4s   (8 epochs → ~97s/epoch)
YOLO-mini training complete in 70.5s       (40 epochs → ~1.8s/epoch)
```

Faster R-CNN's per-epoch cost is roughly **54× higher** than YOLO-mini's,
despite Faster R-CNN's backbone being SMALLER in raw parameter count
ratio-wise relative to its total model size — this enormous per-epoch cost
difference is the direct, empirically-observed manifestation of theory.md
§9's "two-stage is slower" claim: every training step for Faster R-CNN
involves generating thousands of anchors, running the RPN, sampling
proposals, RoI-aligning each one individually, and THEN running the
classification head — a fundamentally more sequential, multi-stage
computation graph compared to YOLO's single forward pass through one CNN.

---

## 5. Section F — Evaluation

### Why Greedy Matching, Not a Full Hungarian/Optimal Assignment

```python
for pbox, plabel in zip(pred_boxes, pred_labels):
    best_iou, best_idx = 0.0, -1
    for idx, (gbox, glabel) in enumerate(zip(gt_boxes, gt_labels)):
        if idx in matched_gt or glabel != plabel:
            continue
        iou = iou_scratch(pbox, gbox)
        if iou > best_iou:
            best_iou, best_idx = iou, idx
```

**Why does each prediction simply grab its single best UNMATCHED ground
truth, rather than solving a globally optimal assignment problem?**
This greedy approach processes predictions in DESCENDING confidence order
(both detectors' prediction lists are explicitly sorted by score before
being passed to `evaluate_detector`), so the highest-confidence prediction
always gets first choice of the best-matching ground truth box, and lower-
confidence predictions can only match what's left over. This greedy
strategy is the standard simplification used in most practical detection
evaluation code (including the conceptual basis of the official COCO/VOC
evaluation protocols) — a full bipartite-matching solution (e.g., via the
Hungarian algorithm) would rarely change the result in practice for
well-separated objects like ours, while adding substantial implementation
complexity for negligible benefit at this dataset's scale (1-2 objects per
image, well-separated).

### Live Result — Near-Perfect, Nearly Tied Performance

```
YOLO-mini:     Precision=1.000  Recall=0.992  (TP=119, FP=0, FN=1)
Faster R-CNN:  Precision=0.992  Recall=1.000  (TP=120, FP=1, FN=0)
```

Both detectors achieve essentially PERFECT performance on this synthetic
task — YOLO-mini misses exactly 1 ground-truth object across the entire
80-image validation set (1 false negative, zero false positives), while
Faster R-CNN catches every object but with exactly 1 spurious extra
detection (1 false positive, zero false negatives). This near-mirror-image
result (one detector's sole error is a "miss," the other's sole error is a
"false alarm") is a striking coincidence at this small evaluation scale,
but more importantly demonstrates that BOTH fundamentally different
architectural paradigms — one-stage direct grid regression vs. two-stage
propose-then-classify — are equally capable of solving a well-defined,
appropriately-scoped detection problem when correctly implemented. The
PRACTICAL difference between them, as quantified in Section E above, is
not detection quality on this task but computational cost: YOLO-mini reaches
comparable accuracy using roughly 1/11th the total training wall-clock time.

---

## 6. Why Background-Process Execution Was Needed Again

Following the pattern established in Phase 2 Topic 2, this implementation's
total runtime (YOLO: 70.5s + Faster R-CNN: 776.4s + dataset generation +
evaluation + visualization ≈ 900+ seconds) exceeds a single tool
invocation's practical execution window. The training was again launched via
`setsid nohup python implementation.py &` and monitored through periodic
polling calls — this is now an established, reliable pattern for any
experiment in this repository whose realistic, scientifically meaningful
configuration (sufficient epochs, sufficient data) requires more wall-clock
time than a single synchronous command execution comfortably allows.

---

## Pitfalls Avoided

| Pitfall | Fix Applied |
|---|---|
| Stacking variable-length box/label tensors crashes default DataLoader | Custom `collate_fn` keeping boxes/labels as lists |
| Ambiguous overlapping ground-truth objects complicate cell/anchor assignment | Rejection-sampled non-overlapping shape placement |
| Implicit float64 NumPy array breaks dtype-strict torchvision NMS kernel | Explicit `dtype=np.float32` on scores array |
| Out-of-bounds grid index for object centered exactly at image edge | `min(int(cx//stride), grid_size-1)` defensive clamp |
| `sqrt` of a tiny-negative float-rounding artifact produces NaN | `torch.clamp(x, min=1e-6)` before every `sqrt` call |
| Unbounded raw regression targets for w,h destabilize early training | Sigmoid applied to ALL YOLO output channels |
| Background-class index collision between YOLO (0-indexed) and torchvision (1-indexed reserved for background) | Explicit `+1`/`-1` conversion at the Faster R-CNN boundary |
| Default image auto-resizing distorts anchor scales tuned for 64×64 input | Explicit `min_size=max_size=IMG_SIZE` |
| Single-invocation execution window shorter than full training time | `setsid` background process + periodic polling |

---

*Previous: [Topic 2 — CNN Architectures](../02-architectures-lenet-to-densenet/explanation.md)*
*Next: [Topic 4 — Segmentation: U-Net & Mask R-CNN](../04-segmentation-unet-maskrcnn/explanation.md)*
