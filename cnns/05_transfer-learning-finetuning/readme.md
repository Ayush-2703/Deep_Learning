# Theory: Transfer Learning & Fine-tuning

**Deep Learning Mastery Repository**

---

## Table of Contents
1. [Why Transfer Learning Works](#1-why-transfer-learning-works)
2. [The Feature Hierarchy Argument](#2-the-feature-hierarchy-argument)
3. [Transfer Learning Strategies](#3-transfer-learning-strategies)
4. [Layer Freezing Mechanics](#4-layer-freezing-mechanics)
5. [Learning Rate Strategies for Fine-tuning](#5-learning-rate-strategies-for-fine-tuning)
6. [Domain Shift and When Transfer Learning Helps (or Hurts)](#6-domain-shift-and-when-transfer-learning-helps-or-hurts)
7. [Practical Decision Framework](#7-practical-decision-framework)

---

## 1. Why Transfer Learning Works

### The Core Idea

Instead of training a network from random initialization on your specific
(often small) target dataset, transfer learning starts from a network ALREADY
TRAINED on a different (typically large) source dataset, and adapts it to
the new task.

```
Traditional approach:
  Random Init → Train on Target Dataset (small) → Deployed Model
  Problem: small dataset + random init = high variance, easy to overfit,
           network must learn EVERYTHING (edges, textures, shapes, AND
           the final task) from scratch using limited data

Transfer learning approach:
  Random Init → Train on Source Dataset (large) → Pretrained Model
                                                          │
                                                          ▼
                                            Adapt to Target Dataset (small)
  Benefit: the network ALREADY KNOWS generic visual features (edges, textures,
           shapes) from the large source dataset; it only needs to learn the
           NEW task-specific mapping using the small target dataset
```

### The Formal Justification

This works because of an empirical (and increasingly theoretically
understood) property of deep networks trained on diverse visual data: their
EARLY layers learn features that are **highly general and task-agnostic**
(edge detectors, color blob detectors, simple textures), while only their
LATER layers learn features that are **increasingly task-specific** (e.g.,
"detector for a specific dog breed's ear shape").

---

## 2. The Feature Hierarchy Argument

### Layer-by-Layer Specificity

```
Layer 1 (early):   Edge/color detectors      — UNIVERSAL across almost any
                                                  visual task (faces, cars,
                                                  shapes, medical images...)

Layer 2-3 (mid):    Textures, simple shapes    — STILL fairly general, useful
                                                  for most natural/structured
                                                  images

Layer 4-5 (late):   Object parts, complex      — INCREASINGLY task-specific;
                    co-occurring patterns        a "wheel detector" is useful
                                                  for vehicles but useless for
                                                  classifying medical scans

Final layers:        Task-specific class        — ENTIRELY task-specific;
                     decision boundaries          must always be re-learned
                                                  for a new task
```

This hierarchy was empirically demonstrated by Zeiler & Fergus (2014) and
Yosinski et al. (2014), who visualized and quantified exactly how
transferable each layer's learned features are across different target
tasks — finding that EARLY layers transfer almost universally well, while
transferability progressively decreases (and task-specific fine-tuning
becomes progressively more necessary) toward the network's output.

### Why This Justifies Freezing Early Layers

If early layers have ALREADY learned near-optimal generic features (because
they were trained on a large, diverse source dataset), there is little to
gain — and some risk of HARM via overfitting on limited target data — from
continuing to update them. Freezing these layers (§4) preserves their
already-good generic features while letting only the later, more
task-specific layers adapt to the new problem.

---

## 3. Transfer Learning Strategies

### 3.1 Feature Extraction (Frozen Backbone)

```
┌──────────────────────────┐     ┌───────────────────┐
│  Pretrained Backbone     │ ──► │  NEW Classifier   │
│  (FROZEN — no gradient   │     │  Head (trainable) │
│   updates at all)        │     │                   │
└──────────────────────────┘     └───────────────────┘

Only the new head's parameters receive gradient updates.
The backbone acts as a FIXED feature extractor.
```

**When to use:** Very small target dataset, target task is SIMILAR to the
source task, or computational budget for fine-tuning is limited (frozen
layers need no gradient computation, reducing both memory and compute cost
during training).

### 3.2 Full Fine-tuning (All Layers Trainable)

```
┌──────────────────────────┐     ┌───────────────────┐
│  Pretrained Backbone     │ ──► │  NEW Classifier   │
│  (UNFROZEN — all layers  │     │  Head (trainable) │
│   receive gradients,     │     │                   │
│   typically at LOW LR)   │     │                   │
└──────────────────────────┘     └───────────────────┘

EVERY parameter in the network can adapt to the target task.
```

**When to use:** Larger target dataset available, target task/domain
differs MORE substantially from the source, or maximum possible accuracy is
required and the computational budget allows full backpropagation through
the entire network.

### 3.3 Partial Fine-tuning (Freeze Early, Unfreeze Late)

```
┌───────────────┐  ┌───────────────┐     ┌───────────────────┐
│  Early Layers │  │  Late Layers  │ ──► │  NEW Classifier   │
│  (FROZEN)     │  │  (UNFROZEN,   │     │  Head (trainable) │
│               │  │   low LR)     │     │                   │
└───────────────┘  └───────────────┘     └───────────────────┘
```

A middle-ground compromise: freeze the most generic (early) layers entirely,
but allow the more task-specific (later) layers to adapt — balancing
overfitting risk against the network's ability to adjust to a genuinely
different target task.

### 3.4 Strategy Comparison Table

```
                       Feature Extraction   Full Fine-tuning   Partial Fine-tuning
─────────────────────────────────────────────────────────────────────────────────
Trainable parameters    Fewest (head only)   All                Some (late layers + head)
Overfitting risk         Lowest                 Highest            Medium
Compute/memory cost       Lowest                 Highest            Medium
Best for                  Tiny target data,       Large target data,  Medium target data,
                          similar domain          different domain    moderately different domain
Adaptation capacity       Limited                 Maximum             Moderate
```

---

## 4. Layer Freezing Mechanics

### How Freezing Works in PyTorch

```python
for param in model.backbone.parameters():
    param.requires_grad = False
```

Setting `requires_grad = False` tells PyTorch's autograd engine to SKIP
gradient computation for that parameter entirely during `.backward()` —
the parameter's `.grad` remains `None` (or unchanged from any previous
value), and the optimizer (even if mistakenly given this parameter) will
have nothing to update it with.

### Why Also Exclude Frozen Parameters From the Optimizer

```python
optimizer = optim.Adam(
    [p for p in model.parameters() if p.requires_grad],   # only trainable params
    lr=1e-4
)
```

**Why filter `if p.requires_grad` when constructing the optimizer, given
that frozen parameters won't receive gradients anyway?**
Technically, Adam (and most optimizers) would simply do nothing useful for a
parameter with `grad=None` — but explicitly excluding frozen parameters
from the optimizer's parameter list is still best practice: it avoids any
optimizer-internal state (e.g., Adam's per-parameter momentum/variance
buffers) being needlessly allocated for parameters that will never update,
saving memory, and makes the code's INTENT explicit and self-documenting —
anyone reading the optimizer construction immediately sees which parameters
are meant to be trainable.

### BatchNorm Layers Require Special Care When Freezing

```python
# Freezing conv/linear weights is straightforward, but BatchNorm layers
# ALSO maintain running statistics (running_mean, running_var) that update
# during forward passes in TRAIN mode — even if requires_grad=False on the
# BatchNorm's weight/bias!

model.backbone.eval()    # forces BatchNorm to use FIXED running stats,
                          # preventing them from drifting during fine-tuning
                          # of a notionally "frozen" backbone
```

This is a common, subtle bug: merely setting `requires_grad=False` on
BatchNorm's learnable `weight`/`bias` does NOT stop its running
mean/variance from continuing to update during forward passes if the layer
remains in `.train()` mode — for a TRULY frozen backbone, the BatchNorm
layers specifically must also be kept in `.eval()` mode throughout training.

---

## 5. Learning Rate Strategies for Fine-tuning

### Why Fine-tuning Uses a LOWER Learning Rate Than Training From Scratch

```
From-scratch training:  lr ≈ 1e-3 (typical Adam default)
Fine-tuning:             lr ≈ 1e-4 to 1e-5 (often 10-100× smaller)
```

The pretrained weights already encode useful, carefully-tuned information.
A LARGE learning rate applied to these weights could cause large, disruptive
updates that destroy this useful pretrained structure before the network has
a chance to make small, beneficial adjustments — colloquially called
"catastrophic forgetting" in the extreme case. A small learning rate allows
gentle, incremental adaptation, preserving most of the valuable pretrained
knowledge while still permitting useful task-specific adjustment.

### Discriminative (Layer-wise) Learning Rates

A more sophisticated strategy assigns DIFFERENT learning rates to different
parts of the network, reflecting each layer's differing need for adaptation:

```
Early layers (most generic):    lowest LR  (e.g., 1e-5) — barely adjust
Middle layers:                   medium LR  (e.g., 1e-4)
Late layers (most task-specific): higher LR  (e.g., 1e-3)
New classifier head:              highest LR (e.g., 1e-3, or even higher,
                                              since it starts from random
                                              init and needs the most
                                              adjustment)
```

```python
optimizer = optim.Adam([
    {"params": model.early_layers.parameters(), "lr": 1e-5},
    {"params": model.late_layers.parameters(),  "lr": 1e-4},
    {"params": model.classifier.parameters(),    "lr": 1e-3},
])
```

This directly encodes the feature-hierarchy argument (§2) into the
optimization process itself: layers that are already "mostly right" get
gentle nudges, while the entirely-new classifier head gets full-strength
updates.

### Warmup for the New Head Before Unfreezing the Backbone

A common two-PHASE fine-tuning recipe:
```
Phase 1: Freeze backbone entirely. Train ONLY the new head for a few epochs
         (using a normal/higher LR) until it reaches reasonable performance.
Phase 2: Unfreeze some or all of the backbone. Continue training the WHOLE
         network at a LOW learning rate for further refinement.
```

**Why this order matters:** at the very start of fine-tuning, the NEW head
is randomly initialized and will produce essentially random, large gradient
signals back into the backbone if unfrozen immediately — this large, noisy
early-training signal could disrupt the pretrained backbone's useful weights
before the head has stabilized. Training the head ALONE first lets it reach
a sensible starting point, producing more refined, less disruptive gradients
once the backbone is later unfrozen.

---

## 6. Domain Shift and When Transfer Learning Helps (or Hurts)

### The Domain Similarity Spectrum

```
Source & Target IDENTICAL distribution:
  → Transfer learning trivially helps (it's just more training data)

Source & Target SIMILAR domain, SAME task:
  → Transfer learning typically helps SUBSTANTIALLY (classic ImageNet→
    medical-imaging-adjacent-task transfer scenarios)

Source & Target SIMILAR domain, DIFFERENT task:
  → Transfer learning often helps (shared low-level features remain useful
    even though the final task-specific decision differs)

Source & Target VERY DIFFERENT domain:
  → Transfer learning MAY help less, or even HURT compared to training from
    scratch with sufficient target data — early-layer features tuned for
    natural images (e.g., ImageNet photos) may be POORLY suited to very
    different data (e.g., raw radar signals, abstract synthetic patterns)
```

### Negative Transfer

In rare cases, starting from pretrained weights can actually perform WORSE
than random initialization — this happens when the source domain's learned
features actively MISLEAD the optimization on a sufficiently different
target domain, and the network must first "unlearn" inappropriate biases
before it can learn useful target-specific features — a strictly harder
optimization path than starting from neutral random initialization.

### The Role of Target Dataset Size

```
Target dataset size:      Recommended strategy:
─────────────────────────────────────────────────────────────
Very small (10s-100s)      Feature extraction (frozen backbone) — minimize
                            overfitting risk; there's not enough data to
                            safely update many parameters
Small-medium (1000s)        Partial or full fine-tuning at LOW learning rate
Large (10,000s+)             Full fine-tuning; with enough target data,
                            the benefit of pretrained initialization
                            diminishes (though it usually still HELPS and
                            rarely hurts, often providing faster convergence
                            even when final accuracy would eventually
                            converge similarly either way)
```

---

## 7. Practical Decision Framework

```
                    START
                       │
                       ▼
        Is the target dataset LARGE (10,000s+)?
              │                          │
             YES                         NO
              │                          │
              ▼                          ▼
     Full fine-tuning           Is target domain SIMILAR
     likely best                to source domain?
                                       │           │
                                      YES           NO
                                       │           │
                                       ▼           ▼
                          Feature extraction   Partial/full fine-tuning
                          OR partial            at LOW learning rate,
                          fine-tuning            monitor closely for
                          (try both, compare      negative transfer
                          validation results)
```

This is a starting heuristic, not a rigid rule — in practice, trying
MULTIPLE strategies and comparing validation performance (exactly what this
topic's implementation does) is the most reliable way to determine the best
approach for any SPECIFIC source/target pairing.

---

## Key Equations Summary

| Concept | Formula / Rule |
|---|---|
| Freezing a parameter | `param.requires_grad = False` |
| Fine-tuning LR (typical) | 10-100× smaller than from-scratch LR |
| Discriminative LR | early layers: low LR, late layers: high LR |
| Frozen BatchNorm | must ALSO call `.eval()`, not just freeze weight/bias |
| Strategy by data size | tiny→feature extraction, large→full fine-tuning |
