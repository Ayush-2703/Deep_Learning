# Theory: Segmentation — U-Net & Mask R-CNN

**Deep Learning Mastery Repository**

---

## Table of Contents
1. [Segmentation Task Taxonomy](#1-segmentation-task-taxonomy)
2. [U-Net Architecture](#2-u-net-architecture)
3. [Skip Connections in U-Net vs ResNet](#3-skip-connections-in-u-net-vs-resnet)
4. [Segmentation Loss Functions](#4-segmentation-loss-functions)
5. [Mask R-CNN Architecture](#5-mask-r-cnn-architecture)
6. [RoIAlign for Mask Prediction](#6-roialign-for-mask-prediction)
7. [Evaluation Metrics for Segmentation](#7-evaluation-metrics-for-segmentation)

---

## 1. Segmentation Task Taxonomy

```
Classification:          "What is in this image?"          → 1 label/image
Object Detection:        "What and WHERE (box)?"            → N boxes+labels/image
Semantic Segmentation:    "What class is EVERY PIXEL?"       → 1 class-map/image
                          (does NOT distinguish between two
                           instances of the same class —
                           "all pixels belonging to ANY car"
                           get the same single "car" label)
Instance Segmentation:    "What class AND which INSTANCE is
                           every pixel?"                      → N masks+labels/image
                          (distinguishes "car #1" from "car #2"
                           — each gets its own separate mask)
Panoptic Segmentation:    Combines semantic (for background/
                           "stuff" classes like sky, road) +
                           instance (for countable "things" like
                           cars, people) into one unified output
```

### Visual Comparison

```
Original Image:        🚗 🚗  (two cars)

Semantic Segmentation:  [car][car]   ← both cars get the SAME "car" mask region,
                                        merged into one connected blob if touching

Instance Segmentation:  [car#1][car#2] ← each car gets its OWN distinct mask,
                                          even if they're touching/overlapping
```

This topic covers ONE representative architecture for each major paradigm:
**U-Net** for semantic segmentation, **Mask R-CNN** for instance segmentation.

---

## 2. U-Net Architecture

**Paper:** Ronneberger, Fischer, Brox (2015) — "U-Net: Convolutional Networks
for Biomedical Image Segmentation"

### The Encoder-Decoder Structure

```
Input Image (H×W)
       │
       ▼
┌──────────────┐                                   ┌──────────────┐
│  Encoder      │                                   │  Decoder      │
│  (Contracting)│                                   │  (Expanding)  │
│               │                                   │               │
│  Conv-Conv     │──skip──────────────────────────►│  Concat-Conv   │
│  MaxPool(2)    │  connection 1                    │  Upsample      │
│  (H/2 × W/2)   │                                   │  (H/2 × W/2)   │
│                │                                   │               │
│  Conv-Conv     │──skip──────────────────────►     │  Concat-Conv   │
│  MaxPool(2)    │  connection 2                    │  Upsample      │
│  (H/4 × W/4)   │                                   │  (H/4 × W/4)   │
│                │                                   │               │
│       ⋮        │                                   │       ⋮        │
│                │                                   │               │
│  Conv-Conv      │   (bottleneck — smallest          │  Conv-Conv     │
│  (H/16×W/16)    │    spatial resolution, most        │  (H/16×W/16)   │
│                 │    abstract features)              │               │
└────────┬────────┘                                   └────────┬──────┘
         │                                                     │
         └─────────────────────► bottleneck ◄───────────────────┘
                                       │
                                       ▼
                            Final Conv (1×1) → per-pixel class scores
                                       │
                                       ▼
                              Output: (num_classes, H, W)
```

### Why "U"-Shaped?

The architecture's name comes directly from its diagram shape: the
contracting encoder path (left side of the "U") progressively REDUCES
spatial resolution while INCREASING channel depth (extracting increasingly
abstract features), then the expanding decoder path (right side of the "U")
progressively RESTORES spatial resolution back to the original input size —
visually forming a "U" when drawn with the bottleneck at the bottom and
skip connections arcing across the top.

### Why a Decoder Is Necessary (Unlike Classification CNNs)

Classification CNNs (Topic 2) only need a SINGLE label per image — they can
progressively shrink spatial resolution all the way down to global average
pooling, discarding spatial detail entirely. Segmentation requires a
PER-PIXEL prediction at the ORIGINAL input resolution — the network must
therefore "undo" the encoder's downsampling, which is exactly the decoder's
job: a sequence of UPSAMPLING operations (transposed convolution or
nearest/bilinear interpolation + convolution) that progressively restore
spatial resolution while refining the predicted segmentation map.

### Upsampling Methods

```
Transposed Convolution ("deconvolution"):
  Learnable upsampling — the kernel weights are trained, similar to a
  regular convolution but with the input/output roles swapped (each input
  pixel scatters its value across MULTIPLE output positions, weighted by
  the learned kernel) — can introduce "checkerboard" artifacts if not
  carefully configured.

Nearest/Bilinear Interpolation + Regular Convolution:
  Fixed, NON-learnable upsampling (just resizes), followed by a normal
  convolution to refine the upsampled result — often preferred in practice
  for avoiding checkerboard artifacts, at the cost of one fewer "free"
  learnable transformation.
```

---

## 3. Skip Connections in U-Net vs ResNet

Both U-Net and ResNet (Topic 2) use "skip connections," but they serve
FUNDAMENTALLY DIFFERENT purposes and use DIFFERENT combination operations:

```
                  ResNet                           U-Net
─────────────────────────────────────────────────────────────────────
Combination:      ADDITION (y = F(x) + x)          CONCATENATION (channel-wise)
Purpose:           Ease optimization of VERY        Recover FINE-GRAINED spatial
                   DEEP networks (gradient flow)     detail LOST during downsampling
Shapes involved:   SAME shape (x and F(x) match)     DIFFERENT origin, matched via
                                                       cropping/padding to align spatially
Spans:             Within a single residual BLOCK     Across the ENTIRE encoder→decoder,
                   (local, short-range)                at EACH resolution level (long-range)
```

### Why Concatenation (Not Addition) for U-Net

The encoder's high-resolution early-layer features contain precise spatial
/boundary information (exactly WHERE an edge is) but limited semantic
abstraction (WHAT the edge belongs to). The decoder's upsampled features
contain rich semantic information (learned through the bottleneck) but have
LOST precise spatial detail during the encoder's downsampling — interpolation
during upsampling can only approximately reconstruct fine boundaries.

Concatenating the encoder's matching-resolution feature map directly into
the decoder gives the decoder DIRECT access to the precise spatial detail
it would otherwise have to (imperfectly) infer — critically important for
producing sharp, accurate segmentation boundaries rather than blurry,
imprecise ones.

---

## 4. Segmentation Loss Functions

### 4.1 Pixel-wise Cross-Entropy

```
L_CE = -(1/N) Σᵢ Σ_c yᵢ,c log(ŷᵢ,c)

Sum over EVERY pixel i, and every class c — essentially classification
loss applied independently at each pixel location.
```

**Limitation:** treats every pixel equally, regardless of class frequency.
If 95% of pixels are background and 5% are the object of interest, the loss
is dominated by getting the easy 95% right, providing weak gradient signal
for correctly segmenting the rare foreground class.

### 4.2 Dice Loss

The Dice coefficient measures overlap between predicted and ground-truth
masks, directly addressing class imbalance:

```
Dice(A, B) = 2|A ∩ B| / (|A| + |B|)

For predicted probability map p̂ and binary ground truth mask y:

Dice = (2 Σᵢ p̂ᵢyᵢ + ε) / (Σᵢ p̂ᵢ + Σᵢ yᵢ + ε)

Dice Loss = 1 - Dice
```

**Why Dice handles class imbalance better than pixel-wise CE:**
Dice is computed as a RATIO relative to the total foreground area
(`Σp̂+Σy` in the denominator) rather than summing independently over EVERY
pixel equally. A model that perfectly predicts a small foreground region
gets full credit (Dice→1) regardless of how large the (correctly-predicted)
background is — the metric inherently FOCUSES on the foreground overlap,
rather than being diluted by the easy, abundant background pixels.

### 4.3 Combined BCE + Dice Loss (Standard Practice)

```
L = L_BCE + L_Dice
```

Combining both losses is extremely common in practice: BCE provides
stable, well-behaved PIXEL-LEVEL gradients (especially helpful early in
training), while Dice directly optimizes the REGION-OVERLAP metric we
actually care about — empirically, this combination often outperforms
either loss alone.

### 4.4 IoU Loss (Jaccard)

```
IoU = |A ∩ B| / |A ∪ B| = |A∩B| / (|A|+|B|-|A∩B|)

IoU Loss = 1 - IoU
```

Closely related to Dice (in fact, `Dice = 2·IoU/(1+IoU)`), but penalizes
errors slightly differently — Dice is generally smoother and more commonly
used for the TRAINING loss, while IoU (also called the Jaccard Index) is
more commonly used for FINAL EVALUATION reporting.

---

## 5. Mask R-CNN Architecture

**Paper:** He, Gkioxari, Dollár, Girshick (2017) — "Mask R-CNN"

### Extending Faster R-CNN With a Mask Head

Mask R-CNN takes the EXACT Faster R-CNN pipeline (Topic 3 §6) and adds ONE
additional parallel output branch: a small per-RoI mask-prediction head,
running alongside the existing classification and box-regression heads.

```
                  Shared Backbone Feature Map
                            │
                            ▼
                   Region Proposal Network
                            │
                            ▼
                   RoI Align (per proposal)
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
    Classification    Box Regression    Mask Head (NEW)
    (which class?)     (refine box)     (small FCN predicting
                                          a binary mask PER CLASS,
                                          at fixed resolution e.g 28×28)
```

### The Mask Head

```
RoI-Aligned features (e.g. 14×14×256)
       │
       ▼
  Conv → Conv → Conv → Conv  (a small fully-convolutional network)
       │
       ▼
  Transposed Conv (upsample to 28×28)
       │
       ▼
  1×1 Conv → K channels (one binary mask PER CLASS, K=num_classes)
       │
       ▼
  Sigmoid → K independent binary masks (28×28 each)

At inference: take ONLY the mask channel corresponding to the
PREDICTED class (from the classification head) — discard the other
K-1 class-specific masks for this RoI.
```

**Why predict ONE mask PER CLASS rather than a single class-agnostic mask?**
This DECOUPLES mask prediction from classification — the mask head doesn't
need to "decide" what class an object is; it only needs to predict "if this
WERE class c, what would its mask look like" for every class independently,
and the (separately-computed) classification head's decision determines
which of these K candidate masks is actually used. Empirically, this
decoupling significantly improves mask quality compared to forcing a single
mask head to ALSO implicitly solve classification.

### The Multi-Task Loss

```
L = L_cls + L_box + L_mask

L_mask = average binary cross-entropy, evaluated ONLY on the mask channel
         corresponding to the RoI's GROUND-TRUTH class (during training)
```

---

## 6. RoIAlign for Mask Prediction

As introduced in Topic 3 §6, RoIAlign uses bilinear interpolation (instead of
RoI Pooling's coordinate rounding) to extract features at PRECISE, non-
quantized sub-pixel locations. This precision was a relatively MINOR
improvement for Faster R-CNN's box-classification task (where rounding
errors of a pixel or two are often tolerable), but is CRITICAL for Mask
R-CNN — predicting a coherent, well-aligned per-pixel binary mask requires
exact spatial correspondence between the extracted features and the
underlying image, and even small misalignments significantly degrade mask
boundary quality. This is precisely why Mask R-CNN's original paper
INTRODUCED RoIAlign as a key contribution, even though the technique applies
equally well (and is now used by default) in plain Faster R-CNN too.

---

## 7. Evaluation Metrics for Segmentation

### Semantic Segmentation: Mean IoU (mIoU)

```
For each class c:
  IoU_c = TP_c / (TP_c + FP_c + FN_c)
          (pixel-level TP/FP/FN for class c specifically)

mIoU = (1/C) Σ_c IoU_c
```

### Pixel Accuracy

```
Pixel Accuracy = (correctly classified pixels) / (total pixels)
```

**Why mIoU is preferred over raw pixel accuracy:** pixel accuracy can be
misleadingly high simply by correctly predicting the (often large, easy)
background class everywhere, even if the model completely fails on rare
foreground classes. mIoU's per-class averaging (BEFORE averaging across
classes) ensures every class — common or rare — contributes EQUALLY to the
final reported metric, directly analogous to why Dice/IoU losses (§4) are
preferred over raw pixel-wise CE during training.

### Instance Segmentation: Mask AP

Identical in spirit to detection's mAP (Topic 3 §10), but using MASK IoU
(overlap between predicted and ground-truth BINARY MASKS) rather than BOX
IoU as the matching criterion for determining true/false positives.

---

## Key Equations Summary

| Concept | Formula |
|---|---|
| U-Net skip connection | concat(encoder_features, decoder_features) |
| Dice coefficient | 2\|A∩B\| / (\|A\|+\|B\|) |
| Dice loss | 1 − Dice |
| Combined loss | L_BCE + L_Dice |
| IoU (Jaccard) | \|A∩B\| / \|A∪B\| |
| Mask R-CNN total loss | L_cls + L_box + L_mask |
| Mean IoU | (1/C)Σ_c TP_c/(TP_c+FP_c+FN_c) |
