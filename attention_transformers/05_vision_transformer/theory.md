# Theory: Vision Transformer (ViT)

**Deep Learning Mastery Repository**

---

## Table of Contents
1. [Applying Transformers to Images](#1-applying-transformers-to-images)
2. [Patch Embedding](#2-patch-embedding)
3. [Patch Embedding as a Strided Convolution](#3-patch-embedding-as-a-strided-convolution)
4. [The [CLS] Token for Image Classification](#4-the-cls-token-for-image-classification)
5. [Position Embeddings for Patches](#5-position-embeddings-for-patches)
6. [The ViT Encoder: Identical to BERT's](#6-the-vit-encoder-identical-to-berts)
7. [Inductive Bias: CNNs vs Vision Transformers](#7-inductive-bias-cnns-vs-vision-transformers)
8. [Patch Size Trade-off](#8-patch-size-trade-off)
9. [Hybrid Architectures](#9-hybrid-architectures)

---

## 1. Applying Transformers to Images

**Paper:** Dosovitskiy et al. (2020) — "An Image is Worth 16x16 Words:
Transformers for Image Recognition at Scale"

Every Transformer covered so far (Topics 2-4) operates on a SEQUENCE of
discrete tokens. An image is a 2D grid of continuous pixel values — not
naturally a sequence at all. ViT's central idea: FORCE an image into
sequence form by splitting it into a grid of fixed-size PATCHES, then
treating each patch as if it were a single "token" in a sequence — after
which EVERYTHING else about the Transformer (self-attention, position
information, encoder stack) applies completely unchanged.

```
Image (H×W×C)  →  split into patches (P×P×C each)  →  flatten each patch
    →  linearly project each flattened patch to d_model  →  sequence of
       "patch embeddings," treated EXACTLY like token embeddings (Topic 3 §4)
```

---

## 2. Patch Embedding

```
Image: H×W×C  (e.g. 32×32×3)
Patch size: P×P  (e.g. 8×8)
Number of patches: (H/P)×(W/P)  (e.g. 4×4 = 16 patches)

Each patch: P×P×C values → FLATTENED to a single vector of length P²·C
            (e.g. 8×8×3 = 192)

Linear projection: flattened_patch (P²·C,) → d_model  via a SHARED,
                   learned linear layer (the SAME projection weights
                   applied to every patch, exactly analogous to how a
                   token embedding table's lookup is the SAME regardless
                   of WHERE in the sequence a given token appears)
```

---

## 3. Patch Embedding as a Strided Convolution

**This construction is mathematically IDENTICAL to a single convolutional
layer with `kernel_size=P` and `stride=P`** (non-overlapping patches):

```
Conv2d(in_channels=C, out_channels=d_model, kernel_size=P, stride=P)

For each non-overlapping P×P output position, this convolution computes
EXACTLY: flatten the P×P×C input patch, apply a learned linear
projection to d_model output channels — precisely patch embedding's
definition above.
```

This is the SAME "convolution as linear projection over a receptive
field" insight explored via im2col in Phase 2 Topic 1 §11 — patch
embedding is simply the SPECIAL CASE where the convolution's stride
EQUALS its kernel size (non-overlapping patches, rather than the
overlapping sliding windows typical of standard CNN layers). Implementing
patch embedding via `nn.Conv2d(..., kernel_size=P, stride=P)` and via
"manually flatten + `nn.Linear`" are two equally valid, mathematically
EXACT-MATCH implementations of the same operation, differing only in
computational convenience — a Conv2d layer is typically slightly more
efficient in practice since it avoids an explicit reshape/flatten step.

---

## 4. The [CLS] Token for Image Classification

Directly borrowing BERT's design (Topic 3 §6): prepend a LEARNED `[CLS]`
embedding to the sequence of patch embeddings. After the full encoder
stack, the `[CLS]` token's final representation — having attended to
EVERY patch via self-attention — serves as the aggregate image
representation, fed into a final classification head.

```
Sequence fed to encoder: [CLS] patch₁ patch₂ ... patch_N
                            ↓ (after full encoder stack)
Classification: Linear([CLS]_final_representation) → class logits
```

---

## 5. Position Embeddings for Patches

Just as BERT needs position embeddings to distinguish "token at position
3" from "token at position 7" (Topic 3 §4), ViT needs position embeddings
to distinguish "patch at grid location (0,0)" from "patch at grid location
(3,3)" — WITHOUT position information, self-attention would treat the
patches as a completely unordered SET, discarding all spatial layout
information (the SAME permutation-invariance problem from Topic 1 §8,
now applied to 2D spatial patches rather than 1D sequential tokens).

**Simplification used by the original ViT (and this implementation):** a
single LEARNED 1D position embedding per patch INDEX (flattening the 2D
patch grid into a 1D sequence in, e.g., row-major order) — the model must
implicitly learn whatever 2D spatial relationships matter (e.g., "patch
index 5 is directly BELOW patch index 1" in a 4-patches-per-row grid)
purely from data, since the embedding itself carries no explicit 2D
structure. More sophisticated variants (2D-factored position embeddings)
exist but the simple 1D approach is what the original paper found
sufficient in practice.

---

## 6. The ViT Encoder: Identical to BERT's

Once patches are embedded and position information is added, ViT's
encoder is a STANDARD, unmodified bidirectional Transformer encoder stack
— structurally and mathematically IDENTICAL to Topic 3's `BERTEncoder`
(multi-head self-attention + FFN, Pre-LN, residual connections). There is
NOTHING vision-specific about the encoder itself — all of ViT's
image-specific design lives entirely in the PATCH EMBEDDING step (§2-3)
and the final classification head; everything after patch embedding is
pure, unmodified sequence-Transformer machinery.

---

## 7. Inductive Bias: CNNs vs Vision Transformers

### What CNNs Get "For Free" Architecturally

```
Locality:               a CNN kernel only looks at a small local
                        neighborhood (Phase 2 Topic 1 §9's receptive
                        field) -- the ARCHITECTURE ITSELF assumes nearby
                        pixels are more likely to be related than distant
                        ones.

Translation equivariance: the SAME kernel is applied at every spatial
                        position (Phase 2 Topic 1 §1) -- the architecture
                        ITSELF assumes a learned feature detector should
                        behave identically regardless of WHERE in the
                        image it fires.
```

### What ViT Must Learn From Scratch

```
ViT's self-attention has NO built-in locality bias -- from the very first
layer, EVERY patch can attend to EVERY other patch equally, regardless of
spatial distance. There is also no inherent translation equivariance --
different position embeddings for different patch locations mean the
model's behavior CAN legitimately differ based on absolute position,
with no architectural pressure toward consistency.

Both of these useful, image-specific "assumptions" that a CNN gets for
free must be LEARNED PURELY FROM DATA by a Vision Transformer, if they
are useful for the task at all.
```

### The Practical Consequence

The original ViT paper's key empirical finding: trained from scratch on
small-to-medium datasets, ViT tends to UNDERPERFORM comparable CNNs — the
lack of built-in inductive biases means ViT needs to DISCOVER, from data
alone, patterns that a CNN's architecture already assumes. However, when
pretrained on VERY large datasets (hundreds of millions of images), ViT
matches or EXCEEDS CNN performance — with enough data, the model can learn
whatever spatial patterns are actually useful, unconstrained by (and
potentially exceeding) the specific assumptions baked into convolution's
fixed local, translation-equivariant structure.

---

## 8. Patch Size Trade-off

```
Smaller patches:  MORE patches (longer sequence) → finer-grained spatial
                  detail, but O(L²) attention cost grows accordingly
                  (Topic 1 §9)

Larger patches:   FEWER patches (shorter sequence) → cheaper computation,
                  but each patch embedding must summarize a LARGER
                  image region as a single vector, potentially losing
                  fine detail within that patch
```

This is directly analogous to the resolution/receptive-field trade-offs
in choosing CNN kernel sizes and strides (Phase 2 Topic 1), now expressed
through sequence length rather than spatial feature-map size.

---

## 9. Hybrid Architectures

A natural middle ground, used by several follow-up works: use a SMALL
CNN backbone (a few convolutional layers) to extract an initial feature
map, THEN feed that feature map's spatial positions as "patches" into a
Transformer encoder — combining CNN's beneficial early-layer inductive
biases (locality, translation equivariance, useful for low-level features
like edges/textures) with the Transformer's strength at modeling
LONG-RANGE relationships between higher-level features, once the CNN has
already done useful local preprocessing.

---

## Key Equations Summary

| Concept | Formula / Rule |
|---|---|
| Patch embedding | flatten(P×P×C patch) → Linear → d_model |
| Patch embed = strided conv | Conv2d(kernel_size=P, stride=P) |
| Number of patches | (H/P)×(W/P) |
| ViT input sequence | [CLS] + patch₁ + ... + patch_N (+ position embeddings) |
| Classification | Linear([CLS]_final) → class logits |
