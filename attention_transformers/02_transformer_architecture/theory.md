# Theory: The Transformer Architecture

**Deep Learning Mastery Repository**

---

## Table of Contents
1. [The Full Architecture Overview](#1-the-full-architecture-overview)
2. [The Encoder Block](#2-the-encoder-block)
3. [The Decoder Block](#3-the-decoder-block)
4. [Position-wise Feed-Forward Networks](#4-position-wise-feed-forward-networks)
5. [Residual Connections in the Transformer](#5-residual-connections-in-the-transformer)
6. [Layer Normalization](#6-layer-normalization)
7. [Pre-LN vs Post-LN](#7-pre-ln-vs-post-ln)
8. [Embeddings and Scaling](#8-embeddings-and-scaling)
9. [Training Details](#9-training-details)
10. [Why "Attention Is All You Need"](#10-why-attention-is-all-you-need)

---

## 1. The Full Architecture Overview

**Paper:** Vaswani et al. (2017) — "Attention Is All You Need"

```
                    ENCODER STACK (N layers)          DECODER STACK (N layers)
Source tokens               │                                    │           Target tokens
      │                     ▼                                    ▼               │
      ▼             ┌─────────────────┐                   ┌───────────────┐      ▼
 Embedding          │ Self-Attention  │                   │ Masked Self-  │  Embedding
      +             │  (bidirectional)│                   │  Attention    │      +
Positional          └───────┬─────────┘                   │  (causal)     │  Positional
 Encoding                   │  +residual, norm            └───────┬───────┘   Encoding
      │                     ▼                                    │  +residual, norm
      │              ┌───────────────┐                            ▼
      │              │  Feed-Forward │                   ┌─────────────────┐
      │              └───────┬───────┘                   │ Cross-Attention │◄──── ENCODER
      │                       │  +residual, norm         │ (Q=decoder,     │      OUTPUT
      │                       ▼                          │  K,V=encoder)   │
      │              (repeat N times)                    └─────────┬───────┘
      │                       │                                    │  +residual, norm
      │                       ▼                                    ▼
      │              Encoder Output ─────────────────────►┌────────────────┐
      │                                                   │  Feed-Forward  │
      │                                                   └────────┬───────┘
      │                                                            │  +residual, norm
      │                                                            ▼
      │                                                    (repeat N times)
      │                                                            │
      │                                                            ▼
      │                                                    Linear + Softmax
      │                                                            │
      │                                                            ▼
      │                                                    Output token probabilities
```

**The single most important architectural claim of this paper:** NO
recurrence, NO convolution — the entire model is built from attention
mechanisms (Topic 1) and simple feed-forward layers, achieving full
parallelism across the sequence dimension during training.

---

## 2. The Encoder Block

Each of the `N` identical encoder layers consists of TWO sub-layers:

```
1. Multi-Head Self-Attention (bidirectional — every source position can
   attend to EVERY other source position, no masking)
2. Position-wise Feed-Forward Network (§4)

Each sub-layer wrapped with a RESIDUAL connection (§5) and LAYER
NORMALIZATION (§6):

  x = x + SelfAttention(x)     (+ LayerNorm, placement discussed in §7)
  x = x + FFN(x)                (+ LayerNorm)
```

The encoder's job: transform the source sequence into a rich, contextualized
representation where EVERY position's output vector has "seen" the ENTIRE
source sequence (via self-attention) — directly generalizing what a
bidirectional RNN encoder (Phase 3 Topic 3) achieves, but with DIRECT,
unattenuated access to every other position (no vanishing-gradient-style
decay across distance).

---

## 3. The Decoder Block

Each of the `N` identical decoder layers consists of THREE sub-layers:

```
1. Masked Multi-Head Self-Attention (CAUSAL — position t can only attend to
   target positions ≤ t, exactly as in Topic 1 §7)
2. Multi-Head CROSS-Attention (Q from the decoder, K/V from the ENCODER's
   output — directly generalizing Phase 3 Topic 3's Bahdanau attention into
   the Q/K/V framework introduced in Topic 1)
3. Position-wise Feed-Forward Network

Each sub-layer wrapped with residual connection + LayerNorm:

  x = x + MaskedSelfAttention(x)         (+ LayerNorm)
  x = x + CrossAttention(Q=x, K=V=enc_out) (+ LayerNorm)
  x = x + FFN(x)                          (+ LayerNorm)
```

### Why Masked Self-Attention BEFORE Cross-Attention (Not After)

The decoder's masked self-attention lets each target position build a
representation informed by PREVIOUS target positions FIRST — establishing
"what have I generated so far" — before that representation is used as
the QUERY for cross-attention into the source ("given what I've generated
so far, what part of the source should I look at next"). This ordering
mirrors the natural generation process: you need SOME notion of decoding
progress before it makes sense to decide what source information is
currently relevant.

---

## 4. Position-wise Feed-Forward Networks

```
FFN(x) = max(0, xW₁+b₁)W₂+b₂          (a 2-layer MLP with ReLU)

Typically: W₁ ∈ ℝ^(d_model × d_ff),  W₂ ∈ ℝ^(d_ff × d_model)
           d_ff is usually 4× larger than d_model (e.g., d_model=512, d_ff=2048)
```

**"Position-wise" means the SAME FFN (identical weights) is applied
INDEPENDENTLY to each sequence position** — no mixing of information ACROSS
positions happens inside the FFN (that's entirely the attention sub-layer's
job). This division of labor is clean: attention sub-layers handle
ACROSS-POSITION information mixing; FFN sub-layers handle PER-POSITION
non-linear transformation/feature processing, applied identically and
in parallel at every position (implemented efficiently as a single batched
matrix multiply treating the sequence dimension as an extra batch
dimension).

---

## 5. Residual Connections in the Transformer

Every sub-layer (self-attention, cross-attention, FFN) is wrapped in a
residual connection: `output = x + Sublayer(x)`.

This serves the SAME fundamental purpose as ResNet's residual connections
(Phase 2 Topic 2 §6): providing a "gradient superhighway" that lets
gradients flow backward through the (potentially very deep, `N`-layer)
stack with minimal attenuation, since `∂output/∂x` always includes a `+1`
term from the identity path regardless of how small the sub-layer's own
gradient contribution becomes. This is ESSENTIAL for training deep
Transformer stacks (modern large language models can have 50-100+ layers)
— without residual connections, such deep stacks would suffer the same
degradation problem that motivated ResNet in the first place.

---

## 6. Layer Normalization

### The Formula

```
LayerNorm(x) = γ · (x-μ)/√(σ²+ε) + β

μ, σ²: computed ACROSS THE FEATURE DIMENSION, for EACH individual sample
       (and each individual sequence position) INDEPENDENTLY
γ, β:  learned scale and shift parameters (same shape as the feature dim)
```

### Why LayerNorm, Not BatchNorm, for Transformers

```
BatchNorm (Phase 1 Topic 5): normalizes ACROSS THE BATCH dimension, for
  each feature independently. Requires a reasonably large, stable batch
  size, and its statistics depend on OTHER samples in the same batch.

LayerNorm: normalizes ACROSS THE FEATURE dimension, for each individual
  sample (and sequence position) INDEPENDENTLY of every other sample.
```

**Two key reasons LayerNorm fits sequence models better:**

```
1. VARIABLE SEQUENCE LENGTHS: BatchNorm's statistics would need special
   handling for padded positions within a batch (padding tokens
   shouldn't contribute to the normalization statistics) — LayerNorm
   sidesteps this entirely, since it never aggregates across the batch
   or sequence dimension at all.

2. INFERENCE-TIME CONSISTENCY: BatchNorm requires tracking running
   statistics for use at inference (Phase 1 Topic 5) and behaves
   differently in train vs eval mode. LayerNorm computes its statistics
   fresh from each individual input, identically in train and eval mode —
   no running-statistics bookkeeping needed, and no train/eval behavioral
   mismatch risk (a subtlety Phase 2 Topic 5 flagged for frozen-backbone
   fine-tuning).
```

---

## 7. Pre-LN vs Post-LN

### Post-LN (Original 2017 Paper)

```
x = LayerNorm(x + Sublayer(x))          (normalize AFTER the residual addition)
```

### Pre-LN (Modern Standard, e.g. GPT-2 onward)

```
x = x + Sublayer(LayerNorm(x))          (normalize BEFORE the sublayer, residual
                                          path stays UN-normalized)
```

### Why This Ordering Matters for Training Stability

In Post-LN, the residual path itself passes THROUGH a LayerNorm at every
layer — over many stacked layers, this repeated normalization can distort
the gradient magnitudes flowing through the "gradient superhighway"
(§5), making VERY deep Post-LN Transformers notoriously difficult to train
without a careful learning-rate WARMUP schedule (§9).

In Pre-LN, the residual path (`x` itself) is NEVER normalized — it flows
through the entire network UNCHANGED except for the additive contributions
from each sub-layer, preserving a cleaner, more directly ResNet-like
gradient path. Empirically, Pre-LN Transformers train more stably and are
LESS sensitive to the learning-rate warmup schedule, which is why most
modern large-scale Transformer implementations (GPT-2/3, most current LLMs)
use Pre-LN despite the original paper using Post-LN.

**Trade-off:** Post-LN can achieve SLIGHTLY better final performance in
some settings (when training succeeds) because every layer's OUTPUT is
kept normalized, but this benefit only materializes if training doesn't
destabilize first — Pre-LN's robustness is usually the more practical
choice, especially as depth increases.

---

## 8. Embeddings and Scaling

```
Input: token indices → Embedding lookup → embeddings ∈ ℝ^(L × d_model)
Scaled: embeddings × √d_model                (BEFORE adding positional encoding)
Then:   + PositionalEncoding(position)        (Topic 1 §8)
```

**Why multiply embeddings by `√d_model` before adding positional
encoding?** The original paper's stated rationale: this keeps the
embedding values at a comparable SCALE to the positional encoding values
(which are bounded in `[-1,1]`, per Topic 1 §8's "bounded" property) —
without this scaling, for large `d_model`, the (typically small-magnitude,
post-Xavier-init) embedding values could be completely dominated by the
additive positional encoding signal, making it harder for the network to
preserve token-identity information distinctly from position information
in the earliest layers.

---

## 9. Training Details

### Learning Rate Warmup (the "Noam" Schedule)

```
lr(step) = d_model^(-0.5) · min(step^(-0.5), step·warmup_steps^(-1.5))

This INCREASES linearly for the first `warmup_steps`, then DECREASES
proportionally to the inverse square root of the step number.
```

This directly echoes Phase 1 Topic 6 §4.4's warmup+decay schedule
discussion — the original Transformer paper's specific schedule was one of
the first widely-influential applications of this now-standard pattern,
motivated by the SAME concern (avoid large, destabilizing early updates
before the optimizer's adaptive estimates and the (Post-LN, in the
original paper) normalization statistics have stabilized).

### Label Smoothing

```
Instead of a one-hot target (probability 1.0 on the correct class, 0 on
all others), label smoothing uses:
  target_smoothed = (1-ε)·one_hot + ε/K   (K = vocabulary size, ε≈0.1)

This prevents the model from becoming OVERCONFIDENT (predicting
probability →1.0 on the correct token), which empirically improves
generalization and BLEU scores, at the cost of slightly higher training
perplexity (since the model is explicitly discouraged from ever reaching
the theoretical minimum loss of a hard one-hot target).
```

---

## 10. Why "Attention Is All You Need"

```
                        RNN Seq2Seq+Attn      Transformer
                        (Phase 3 Topic 3)
─────────────────────────────────────────────────────────────
Cross-position mixing    Sequential RNN         Self-attention
                         recurrence + Bahdanau   (parallel, direct)
                         attention for cross-seq

Training parallelism     Sequential per          FULLY parallel across
                         time-step (even with     sequence positions
                         teacher forcing, the      (causal mask lets ALL
                         decoder RNN must           positions' losses be
                         still unroll step-by-      computed in ONE
                         step)                      forward pass)

Long-range dependency    Attenuated by             Direct, unattenuated
                         recurrence depth           (single attention hop
                         (mitigated by gating,      regardless of distance)
                         Phase 3 Topic 2)

Positional information   Implicit (via              Explicit (must be
                         recurrence order)           injected via positional
                                                     encoding, Topic 1 §8)
```

The Transformer's core insight: RECURRENCE was never fundamentally
NECESSARY for sequence modeling — it was simply the mechanism available
before attention was generalized (Topic 1) into a standalone,
recurrence-free tool for cross-position information mixing. Removing
recurrence ENTIRELY (replacing it with self-attention + explicit
positional encoding) trades RNNs' inherent sequentiality for full
training parallelism, at the cost of `O(L²)` attention complexity (Topic 1
§9) — a trade that has proven overwhelmingly favorable at the sequence
lengths and hardware (massively parallel GPUs/TPUs) used for most modern
large-scale sequence modeling.

---

## Key Equations Summary

| Concept | Formula |
|---|---|
| Encoder sublayer | x = x + SelfAttn(x), then x = x + FFN(x) (each +LayerNorm) |
| Decoder sublayers | x=x+MaskedSelfAttn(x); x=x+CrossAttn(x,enc); x=x+FFN(x) |
| Position-wise FFN | FFN(x) = max(0,xW₁+b₁)W₂+b₂ |
| LayerNorm | γ·(x-μ)/√(σ²+ε)+β, stats over FEATURE dim |
| Post-LN | x = LayerNorm(x + Sublayer(x)) |
| Pre-LN | x = x + Sublayer(LayerNorm(x)) |
| Embedding scaling | embed(x) × √d_model, then + PE |
| Noam LR schedule | d_model⁻⁰·⁵·min(step⁻⁰·⁵, step·warmup⁻¹·⁵) |
