# Theory: Object Detection — Region-Based CNNs (Faster R-CNN) & YOLO

**Deep Learning Mastery Repository**

---

## Table of Contents
1. [Object Detection: Problem Definition](#1-object-detection-problem-definition)
2. [Intersection over Union (IoU)](#2-intersection-over-union-iou)
3. [Anchor Boxes](#3-anchor-boxes)
4. [Non-Maximum Suppression (NMS)](#4-non-maximum-suppression-nms)
5. [The R-CNN Family Evolution](#5-the-r-cnn-family-evolution)
6. [Faster R-CNN Architecture in Depth](#6-faster-r-cnn-architecture-in-depth)
7. [YOLO: You Only Look Once](#7-yolo-you-only-look-once)
8. [YOLO Loss Function](#8-yolo-loss-function)
9. [Two-Stage vs One-Stage: Trade-offs](#9-two-stage-vs-one-stage-trade-offs)
10. [Evaluation: mean Average Precision (mAP)](#10-evaluation-mean-average-precision-map)

---

## 1. Object Detection: Problem Definition

Unlike classification (one label per image), object detection requires
predicting, for an arbitrary and unknown number of objects:

```
For each object in the image:
  1. A bounding box: (x_min, y_min, x_max, y_max) or (cx, cy, w, h)
  2. A class label: which category the object belongs to
  3. (Implicitly) a confidence score: how certain the model is an object exists here

Output: a VARIABLE-LENGTH set of (box, class, confidence) tuples per image
```

This variable-length output is the central technical challenge — classification
networks output a FIXED-size vector (one score per class); detection
networks must somehow produce an unbounded number of detections from a
fixed-size network. The two major architectural paradigms (covered in this
topic) solve this differently:

```
Two-stage (R-CNN family): First PROPOSE candidate regions, THEN classify each
One-stage (YOLO family):  Directly predict a FIXED dense grid of boxes,
                            most of which are trained to predict "no object"
```

---

## 2. Intersection over Union (IoU)

IoU is the fundamental metric for comparing two bounding boxes — used for
training (matching predictions to ground truth), inference (suppressing
duplicates), and evaluation (mAP).

```
Given two boxes A and B:

IoU(A, B) = Area(A ∩ B) / Area(A ∪ B)

         = Area(A ∩ B) / [Area(A) + Area(B) − Area(A ∩ B)]

Range: [0, 1]
  IoU = 0: boxes don't overlap at all
  IoU = 1: boxes are identical
```

### Computing Intersection Area

```
Given A = (xA1,yA1,xA2,yA2),  B = (xB1,yB1,xB2,yB2):

x_left   = max(xA1, xB1)
y_top    = max(yA1, yB1)
x_right  = min(xA2, xB2)
y_bottom = min(yA2, yB2)

if x_right < x_left or y_bottom < y_top:
    intersection = 0          (boxes don't overlap)
else:
    intersection = (x_right − x_left) × (y_bottom − y_top)
```

### IoU Thresholds in Practice

```
IoU ≥ 0.5:   commonly used threshold for "this prediction counts as correct"
             in the standard PASCAL VOC mAP calculation
IoU ≥ 0.7:   stricter threshold, used for "this anchor is a positive sample"
             during Faster R-CNN's RPN training (theory §6)
IoU < 0.3:   commonly the threshold for "this anchor is a definite negative"
0.3-0.7:     "ambiguous" zone, often excluded from loss computation entirely
```

---

## 3. Anchor Boxes

### The Problem Anchors Solve

A naive detector might try to directly regress `(x, y, w, h)` for each
object from scratch — but this is a HARD regression problem with no good
starting point, especially for objects of vastly different sizes/aspect
ratios appearing anywhere in the image.

**Anchors provide a reference**: a set of pre-defined candidate boxes
(varying in scale and aspect ratio) tiled densely across the image. Instead
of regressing absolute coordinates, the network only needs to predict small
ADJUSTMENTS (offsets) relative to the nearest anchor — a much easier learning
problem.

### Anchor Generation

```
For each anchor "center" location on a grid (e.g., every 16 pixels):
  Generate K anchors with different (scale, aspect_ratio) combinations:

  scales = [64, 128, 256]              (in pixels, e.g. for 3 scales)
  aspect_ratios = [0.5, 1.0, 2.0]       (width:height ratios)

  → 3 scales × 3 ratios = 9 anchors PER grid location

Total anchors for a 16×16 grid: 16×16×9 = 2,304 anchors covering the image
```

### Anchor-to-Ground-Truth Matching (Training)

```
For each anchor a:
  best_gt = argmax_g IoU(a, g)    over all ground truth boxes g

  if IoU(a, best_gt) ≥ pos_threshold (e.g. 0.7):
      a is a POSITIVE anchor → train to predict best_gt's offset + its class
  elif IoU(a, best_gt) < neg_threshold (e.g. 0.3):
      a is a NEGATIVE anchor → train to predict "background"
  else:
      a is IGNORED (ambiguous, excluded from loss)

Additionally: the anchor with the HIGHEST IoU for each ground-truth box is
ALWAYS marked positive (even if below pos_threshold), ensuring every object
has at least one responsible anchor.
```

### Offset Parameterization (What the Network Actually Predicts)

Rather than predicting absolute pixel coordinates, the network predicts a
SCALE-INVARIANT transformation relative to the matched anchor:

```
Given anchor (xa, ya, wa, ha) and target ground truth (x*, y*, w*, h*):

tx = (x* − xa) / wa          ty = (y* − ya) / ha
tw = log(w* / wa)             th = log(h* / ha)

The network predicts (tx, ty, tw, th); the actual box is then reconstructed:

x_pred = xa + tx_pred·wa       y_pred = ya + ty_pred·ha
w_pred = wa · exp(tw_pred)      h_pred = ha · exp(th_pred)
```

**Why log for width/height?** Width and height are always positive and can
vary over orders of magnitude. A raw linear regression on `w` would need to
predict very different magnitudes for small vs. large objects. Predicting
`log(w*/wa)` instead means the network predicts a small, roughly
zero-centered number regardless of absolute object size — only the RATIO to
the anchor matters, making the regression problem far better-conditioned.

---

## 4. Non-Maximum Suppression (NMS)

### The Problem NMS Solves

A detector typically outputs MANY overlapping candidate boxes for the SAME
underlying object (since many anchors near an object will have high
confidence). NMS filters this down to one box per object.

### The Algorithm

```
Input: list of (box, confidence_score) pairs, an IoU threshold τ (e.g. 0.5)

1. Sort all boxes by confidence score, descending
2. Initialize empty list `keep`
3. While boxes remain:
     a. Take the box with the HIGHEST remaining confidence → add to `keep`
     b. Remove this box from consideration
     c. Remove ALL remaining boxes with IoU ≥ τ against the box just kept
        (these are considered duplicate detections of the same object)
4. Return `keep`
```

### Worked Example

```
Boxes (sorted by confidence): A(0.95), B(0.90), C(0.85), D(0.30)
IoU(A,B)=0.8, IoU(A,C)=0.1, IoU(A,D)=0.05, IoU(C,D)=0.6

Step 1: Keep A (highest conf). Remove B (IoU=0.8≥0.5 with A).
        C, D remain (low IoU with A).
Step 2: Keep C (highest among remaining). Remove D (IoU=0.6≥0.5 with C).
Step 3: No boxes remain.

Final result: {A, C} — B was a duplicate of A, D was a duplicate of C.
```

### Class-Aware NMS

In multi-class detection, NMS is typically applied SEPARATELY per class —
boxes of DIFFERENT predicted classes are never suppressed against each
other, even if they overlap heavily (e.g., a "person" box and a "skateboard"
box can legitimately overlap a lot without being duplicates of each other).

---

## 5. The R-CNN Family Evolution

### R-CNN (2014)

```
1. Generate ~2,000 candidate region proposals using a separate, NON-LEARNED
   algorithm (Selective Search) — no neural network involved in proposing regions
2. Warp/crop each region to a fixed size (e.g. 224×224)
3. Run EACH region independently through a CNN (e.g. AlexNet) for feature extraction
4. Classify each region's features with an SVM
5. Refine box coordinates with a separate linear regressor

Major bottleneck: running the CNN ~2,000 times PER IMAGE is extremely slow
(~47 seconds per image on a GPU at the time).
```

### Fast R-CNN (2015)

```
Key improvement: run the CNN ONCE on the whole image to get a shared feature
map, then use "RoI Pooling" to extract a fixed-size feature vector for each
proposed region directly FROM this shared feature map (instead of re-running
the CNN per region).

1. Run full image through CNN backbone ONCE → shared feature map
2. For each of the ~2,000 region proposals (still from Selective Search):
     RoI Pool the corresponding region of the feature map → fixed-size vector
3. Classify + regress box offsets via fully-connected layers (per region)

Speedup: ~25× faster than R-CNN, since the expensive CNN backbone runs only once.
Remaining bottleneck: Selective Search (region proposal step) is still a
slow, NON-LEARNED, CPU-based algorithm — not trainable end-to-end.
```

### Faster R-CNN (2015)

```
Key improvement: REPLACE Selective Search with a small, LEARNED neural
network — the Region Proposal Network (RPN) — that shares the same backbone
feature map as the classification head, making the ENTIRE pipeline
end-to-end trainable with backpropagation.

1. Run full image through CNN backbone ONCE → shared feature map
2. RPN slides over the feature map, proposing regions using ANCHOR BOXES
   (§3) at each location, predicting objectness + box refinement
3. Take the RPN's top-scoring proposals (e.g. top 2000, then NMS to ~300)
4. RoI Pool/Align each proposal's features from the SAME shared feature map
5. Final classification + box regression head (per proposal)

Speedup: ~10× faster than Fast R-CNN, AND fully end-to-end trainable
(the RPN itself is learned, not a fixed heuristic algorithm).
```

---

## 6. Faster R-CNN Architecture in Depth

### Full Pipeline Diagram

```
Input Image
    │
    ▼
┌─────────────────┐
│  Backbone CNN    │   (e.g. ResNet, VGG — shared feature extractor)
│  (+ optional FPN)│
└────────┬─────────┘
         │  shared feature map
         ▼
┌─────────────────────────┐
│ Region Proposal Network  │
│  (RPN)                   │
│  - slides anchors over    │
│    feature map            │
│  - predicts: objectness   │
│    score + box deltas     │
│    PER ANCHOR              │
└────────┬─────────────────┘
         │  ~2000 raw proposals → NMS → ~300 proposals
         ▼
┌─────────────────────────┐
│  RoI Align                │   (extracts fixed-size feature per proposal
└────────┬─────────────────┘    from the SAME shared feature map)
         │
         ▼
┌─────────────────────────┐
│  Detection Head           │
│  - Classification (softmax │
│    over C classes + bg)    │
│  - Box regression (per-class│
│    refinement)             │
└────────┬─────────────────┘
         │
         ▼
   Final detections (after per-class NMS)
```

### RoI Align vs. RoI Pooling

```
RoI Pooling (Fast R-CNN): snaps the proposal's boundaries to the nearest
  feature-map grid cell — introduces small MISALIGNMENTS due to rounding,
  especially noticeable for small objects.

RoI Align (Faster R-CNN improvements / Mask R-CNN paper): uses BILINEAR
  INTERPOLATION to sample feature values at exact (non-rounded) sub-pixel
  locations within the proposal — avoids the quantization artifacts of
  RoI Pooling, improving localization accuracy, especially for tasks
  requiring pixel-level precision (e.g. instance segmentation, Topic 4).
```

### The Four-Part Multi-Task Loss

```
L = L_rpn_cls + L_rpn_box + L_head_cls + L_head_box

L_rpn_cls:  binary cross-entropy — "does this anchor contain an object?"
L_rpn_box:  smooth-L1 (Huber) regression loss on anchor→proposal offsets
L_head_cls: categorical cross-entropy — "which of C+1 classes (incl. background)?"
L_head_box: smooth-L1 regression loss on proposal→final-box offsets

Both box regression losses ONLY apply to POSITIVE (matched) anchors/proposals
— there's no meaningful "box refinement" target for background samples.
```

---

## 7. YOLO: You Only Look Once

**Paper:** Redmon et al. (2016) — "You Only Look Once: Unified, Real-Time
Object Detection"

### The Core Philosophy: Detection as Direct Regression

Instead of a two-stage propose-then-classify pipeline, YOLO frames detection
as a SINGLE regression problem: divide the image into an `S×S` grid, and have
EACH grid cell directly predict bounding boxes, confidence, and class
probabilities in ONE forward pass.

```
Input Image (e.g. 448×448)
    │
    ▼
   CNN Backbone
    │
    ▼
S×S×(B×5+C) output tensor

where:
  S×S  = grid dimensions (e.g. 7×7)
  B    = number of boxes predicted per cell (e.g. 2)
  5    = each box predicts (x, y, w, h, confidence)
  C    = number of classes (class probabilities, shared across the B boxes
         in the original YOLOv1; later versions predict per-box classes)
```

### Grid Cell Responsibility

```
Each ground-truth object is assigned to EXACTLY ONE grid cell — the cell
containing the object's CENTER point. That cell's predictor(s) are trained
to predict this object; ALL OTHER cells are trained to predict "no object"
(confidence → 0).

(x, y) predicted RELATIVE to the cell's top-left corner, normalized to [0,1]
(w, h) predicted RELATIVE to the WHOLE IMAGE, normalized to [0,1]
```

### Confidence Score Definition

```
confidence = P(object) × IoU(predicted_box, ground_truth_box)

This is a SINGLE learned scalar meant to capture BOTH "is there an object
here" AND "how accurate is my box" simultaneously — at inference, this
confidence directly serves as a detection score for NMS ranking.
```

### Why "You Only Look Once" — The Key Speed Advantage

Unlike two-stage detectors (which run a SEPARATE classification network pass
for EACH of ~300 proposals), YOLO's entire grid of predictions comes from a
SINGLE forward pass of a SINGLE network over the SINGLE input image — hence
"you only look once." This makes YOLO dramatically faster at inference time,
the primary motivation for its design (real-time detection, e.g. for video).

---

## 8. YOLO Loss Function

The original YOLOv1 loss is a carefully weighted sum of squared errors:

```
L = λ_coord Σᵢ₌₀^{S²} Σⱼ₌₀^B 𝟙_{ij}^{obj} [(xᵢ−x̂ᵢ)² + (yᵢ−ŷᵢ)²]
  + λ_coord Σᵢ₌₀^{S²} Σⱼ₌₀^B 𝟙_{ij}^{obj} [(√wᵢ−√ŵᵢ)² + (√hᵢ−√ĥᵢ)²]
  + Σᵢ₌₀^{S²} Σⱼ₌₀^B 𝟙_{ij}^{obj} (Cᵢ−Ĉᵢ)²
  + λ_noobj Σᵢ₌₀^{S²} Σⱼ₌₀^B 𝟙_{ij}^{noobj} (Cᵢ−Ĉᵢ)²
  + Σᵢ₌₀^{S²} 𝟙ᵢ^{obj} Σ_{c∈classes} (pᵢ(c)−p̂ᵢ(c))²
```

### Term-by-Term Breakdown

```
Term 1 (coordinate xy):    penalizes (x,y) center prediction error
                            ONLY for cells/boxes RESPONSIBLE for an object
                            (𝟙_{ij}^{obj} = 1 only for the matched box)

Term 2 (coordinate wh):    penalizes width/height error, using √w, √h
                            (NOT raw w, h) — see rationale below

Term 3 (object confidence): penalizes confidence error for boxes that
                            SHOULD contain an object

Term 4 (no-object conf.):  penalizes confidence error for boxes that should
                            NOT contain an object — weighted by λ_noobj < 1
                            (typically 0.5) since the VAST majority of grid
                            cells contain no object, and without down-
                            weighting this term would dominate the loss and
                            push EVERY confidence toward zero, including for
                            cells that should detect something

Term 5 (classification):   penalizes class probability error, only for
                            cells responsible for an object
```

### Why √w, √h Instead of Raw w, h

```
A fixed absolute error in width (e.g. Δw=10 pixels) matters A LOT for a
small object (w=20 → 50% relative error) but BARELY AT ALL for a large
object (w=300 → 3% relative error).

Using √w means: d(√w)/dw = 1/(2√w) — the GRADIENT of the square-root
transform is LARGER for small w than for large w, automatically giving
proportionally MORE importance to size errors on small objects relative to
large ones, without needing a separate explicit per-object-size reweighting
term.
```

### λ_coord and λ_noobj — Why These Specific Weightings Matter

```
λ_coord = 5    (typical value — UPWEIGHTS localization loss)
λ_noobj = 0.5  (typical value — DOWNWEIGHTS no-object confidence loss)

Without these adjustments, the loss is dominated by the the high COUNT of
"no object" cells (since most cells are background) — the model would
quickly learn to predict near-zero confidence everywhere (minimizing the
dominant term) at the expense of ever learning accurate localization for
the rare cells that DO contain objects.
```

---

## 9. Two-Stage vs One-Stage: Trade-offs

```
                       Two-Stage (Faster R-CNN)     One-Stage (YOLO)
─────────────────────────────────────────────────────────────────────
Speed                  Slower (multiple stages)      Faster (single pass)
Accuracy (small objs)  Generally better                Historically weaker
                                                        (improved a lot in
                                                         later YOLO versions)
Localization precision Higher (RoI Align refinement)   Lower (coarser grid)
Training complexity     More complex (RPN + head)        Simpler (single network)
Real-time capability    Historically NO (5-10 FPS)       YES (45+ FPS, YOLOv1)
Class imbalance handling RPN's fixed pos/neg sampling    λ_noobj reweighting
```

**Why both paradigms persist today:** the choice depends on the deployment
constraint. Surveillance/autonomous-driving systems requiring REAL-TIME
inference on edge hardware favor YOLO-family one-stage detectors. Offline
or high-accuracy-critical applications (e.g., medical imaging, satellite
imagery analysis) where inference speed is less critical often favor
two-stage detectors for their typically higher localization precision.

---

## 10. Evaluation: mean Average Precision (mAP)

### Precision and Recall at a Given IoU/Confidence Threshold

```
True Positive  (TP): predicted box matches a ground truth (IoU ≥ threshold,
                      correct class) and hasn't already been matched
False Positive (FP): predicted box doesn't match any ground truth
False Negative (FN): a ground truth box with no matching prediction

Precision = TP / (TP + FP)     "of my predictions, how many were correct?"
Recall    = TP / (TP + FN)     "of all true objects, how many did I find?"
```

### Average Precision (AP) — Area Under the Precision-Recall Curve

```
1. Rank all predictions (across the whole dataset, for one class) by
   confidence score, descending
2. Walk down this ranked list, computing cumulative precision/recall at
   each point as more predictions are considered
3. AP = area under the resulting precision-recall curve
   (in practice, often computed via 11-point or all-point interpolation)
```

### mean Average Precision (mAP)

```
mAP = (1/C) Σ_c AP_c        average AP across all C classes

mAP@0.5:       AP computed using IoU threshold 0.5 for "correct" match
mAP@0.5:0.95:  average of mAP computed at IoU thresholds 0.5, 0.55, ..., 0.95
               (the stricter, more comprehensive COCO-style metric)
```

mAP is the standard benchmark metric for comparing detector accuracy across
the entire field — both Faster R-CNN and YOLO-family papers report results
in mAP to allow direct comparison despite their very different architectures.

---

## Key Equations Summary

| Concept | Formula |
|---|---|
| IoU | Area(A∩B) / Area(A∪B) |
| Anchor offset (x,y) | tx=(x*−xa)/wa, ty=(y*−ya)/ha |
| Anchor offset (w,h) | tw=log(w*/wa), th=log(h*/ha) |
| NMS keep condition | keep highest-conf box; discard others with IoU≥τ against it |
| YOLO confidence | P(object) × IoU(pred,gt) |
| YOLO coord loss | λ_coord·𝟙^obj·[(x−x̂)²+(y−ŷ)²+(√w−√ŵ)²+(√h−√ĥ)²] |
| Precision / Recall | TP/(TP+FP),  TP/(TP+FN) |
| mAP | (1/C)Σ_c AP_c |
