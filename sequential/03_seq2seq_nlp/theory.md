# Theory: Seq2Seq, Attention & Teacher Forcing

**Deep Learning Mastery Repository**

---

## Table of Contents
1. [The Encoder-Decoder Framework](#1-the-encoder-decoder-framework)
2. [Teacher Forcing](#2-teacher-forcing)
3. [Bahdanau Attention (Additive)](#3-bahdanau-attention-additive)
4. [Luong Attention (Multiplicative)](#4-luong-attention-multiplicative)
5. [Attention as Soft Alignment](#5-attention-as-soft-alignment)
6. [Beam Search Decoding](#6-beam-search-decoding)
7. [Evaluation: BLEU Score](#7-evaluation-bleu-score)

---

## 1. The Encoder-Decoder Framework

### Motivation

Vanilla RNNs handle sequences of ONE FIXED role (all input, or all output).
Many real tasks require mapping a sequence of one length/domain to a sequence
of a DIFFERENT length/domain:

```
Source: "How are you?"              (English, 3 tokens)
Target: "Comment allez-vous?"       (French, 3 tokens)

Source: "Summarize this paragraph"  (100 tokens)
Target: "It describes..."           (15 tokens)
```

The **encoder-decoder** (Seq2Seq) architecture handles this by separating the
network into two roles:

```
Encoder RNN:   reads the SOURCE sequence, compresses it into a fixed-size
               context vector c (the final hidden state hₑₙc_T)

Decoder RNN:   generates the TARGET sequence token-by-token, conditioned
               on c and all previously generated tokens

              ┌────────────────────┐    c    ┌───────────────────────┐
SOURCE ──────►│    Encoder RNN      │──────►│    Decoder RNN          │──► TARGET
              │  h₁,h₂,...,hₜ      │        │  generates y₁,y₂,...,yₜ│
              └────────────────────┘        └───────────────────────┘
```

### The Information Bottleneck Problem

The encoder's FINAL hidden state `c = hₑₙc_T` must compress the ENTIRE source
sequence into a FIXED-SIZE vector. For long sequences, this is a severe bottleneck:

```
Short source (5 tokens):  5 tokens → d-dim vector    (manageable)
Long source (50 tokens):  50 tokens → same d-dim vector  (much harder!)

The fixed-size bottleneck is why basic Seq2Seq performance degrades
significantly for long sequences — directly motivating attention (§3).
```

---

## 2. Teacher Forcing

### The Training vs Inference Discrepancy

**During inference (generation):**
```
At each decoding step t, the decoder's input is its OWN PREVIOUS PREDICTION ŷ_{t-1}.
If ŷ_{t-1} is wrong, the error PROPAGATES — each wrong token corrupts all subsequent
decoding steps, causing "exposure bias" (the model never learns to recover from
its own mistakes during training).
```

**Teacher forcing (used during training):**
```
At each decoding step t, ALWAYS feed the GROUND-TRUTH PREVIOUS TOKEN y_{t-1}
as the decoder's input — regardless of what the model would have actually predicted.

This provides clean, correct context at every step, making training fast and
stable. But the model never learns to handle its own mistakes.
```

### Scheduled Sampling (Curriculum Learning)

```
A practical compromise: begin training with 100% teacher forcing (stable, fast),
gradually decrease to 0% over training (increasing "real" generation exposure).

At each step, with probability p: use ground-truth token  (teacher forcing)
                  with probability (1-p): use model's own prediction

p starts near 1.0, decays toward 0.0 as training progresses.
```

---

## 3. Bahdanau Attention (Additive)

**Paper:** Bahdanau, Cho, Bengio (2015) — "Neural Machine Translation by Jointly
Learning to Align and Translate"

### The Core Idea

Instead of compressing the entire source into ONE fixed vector, attention allows
the decoder to ATTEND TO DIFFERENT PARTS of the source at each decoding step:

```
For decoding step t (generating token yₜ):

  1. For each encoder hidden state hⱼ (j=1,...,T_src):
       eₜⱼ = vₐᵀ tanh(Wₐhₜ₋₁ + Uₐhⱼ)      ← alignment score (scalar)

  2. Normalize scores:
       αₜⱼ = exp(eₜⱼ) / Σⱼ exp(eₜⱼ)         ← attention weights (sum to 1)

  3. Weighted sum of encoder states:
       cₜ = Σⱼ αₜⱼ hⱼ                          ← context vector (decoder-step-specific)

  4. Combine context with decoder state:
       sₜ = tanh(Wc[sₜ₋₁ ; cₜ] + bc)           ← updated decoder state

  5. Generate output:
       P(yₜ | y<ₜ, X) = softmax(Wsₜ + b)
```

**Why "additive"?** The alignment function `eₜⱼ = vₐᵀ tanh(W·hₜ₋₁ + U·hⱼ)` ADDS
(after linear projection) the decoder state `hₜ₋₁` and encoder state `hⱼ`, then
applies `tanh` — a learnable compatibility function.

---

## 4. Luong Attention (Multiplicative / Dot-Product)

**Paper:** Luong, Pham, Manning (2015) — "Effective Approaches to Attention-based
Neural Machine Translation"

```
Dot-product attention:    eₜⱼ = sₜᵀ hⱼ
General attention:        eₜⱼ = sₜᵀ Wₐ hⱼ
Concat attention:         eₜⱼ = vₐᵀ tanh(Wₐ[sₜ ; hⱼ])

Most common variant: scaled dot-product  eₜⱼ = (sₜᵀ hⱼ) / √d
```

**Why scale by `√d`?** For large dimensions `d`, dot products grow large in
magnitude → softmax saturates (nearly one-hot) → gradients vanish. Scaling by
`√d` keeps dot products in a stable range regardless of dimension.

This scaled dot-product attention is the building block of the Transformer
(Phase 4 Topic 1).

---

## 5. Attention as Soft Alignment

The attention weights `αₜⱼ` can be interpreted as a soft ALIGNMENT between
each target token (at decoding step `t`) and each source token (at encoder
position `j`).

```
For a translation "The cat sat on the mat" → "Le chat s'est assis sur le tapis":

When generating "chat" (French for "cat"), the model should attend strongly
to "cat" in the source — αₜⱼ should be large for j corresponding to "cat".

The alignment matrix (T_target × T_source) visualized as a heatmap should
show a roughly diagonal pattern for simple translation, with deviations
corresponding to reordering (e.g., adjective-noun order differs between
French and English).
```

---

## 6. Beam Search Decoding

### Why Greedy Decoding Is Suboptimal

Greedy decoding: at each step, pick the single HIGHEST-PROBABILITY next token.

```
Step 1: "The" (prob=0.8) ← greedy picks this
Step 2: "cat" (prob=0.6)
Step 3: "sat" (prob=0.3)
Total sequence probability: 0.8×0.6×0.3 = 0.144

But a different first choice:
Step 1: "A" (prob=0.2)
Step 2: "very" (prob=0.9)
Step 3: "small" (prob=0.8)
Total: 0.2×0.9×0.8 = 0.144 ← same, but greedy would never explore this
```

### Beam Search

```
Maintain K "hypotheses" (partial sequences) at each step:

At each decoding step:
  For each current hypothesis: expand it by ONE token, keeping top-K
  candidates by cumulative log-probability

  From the K×|Vocab| candidates, keep only the top-K best hypotheses
  (by total log-probability so far)

Continue until all hypotheses have produced <EOS>, or a max length is reached.
Return the highest-probability complete sequence.
```

**Typical K (beam size) = 4 or 5** for most applications. K=1 reduces to
greedy decoding; K=|Vocab| is exhaustive search (computationally infeasible
for large vocabularies).

---

## 7. Evaluation: BLEU Score

BLEU (Bilingual Evaluation Understudy) measures n-gram overlap between a
generated sequence and one or more reference translations.

```
BLEU = BP × exp(Σₙ wₙ log pₙ)

where:
  pₙ    = modified n-gram precision (fraction of generated n-grams that
           appear in the reference, clipped to avoid repetition gaming)
  wₙ    = weight for each n-gram order (typically 1/N for uniform)
  N     = max n-gram order (typically 4)
  BP    = brevity penalty = exp(1 - len_ref/len_hyp) if len_hyp < len_ref
                                                    else 1
```

**BLEU range: [0, 1]** (or reported as 0-100). Higher is better. Rough interpretation:
```
> 0.4 (40): Very good, approaching human quality for some tasks
0.3-0.4:    Good quality
0.2-0.3:    Understandable
< 0.2:      Poor quality
```

**Note on BLEU's limitations:** BLEU only measures n-gram overlap, not semantic
quality. A paraphrase that means the same thing but uses different words scores
zero. BLEU is widely used but increasingly complemented by learned metrics
(BERTScore, BLEURT) in current research.

---

## Key Equations Summary

| Concept | Formula |
|---|---|
| Encoder context (basic) | c = h_T (last hidden state) |
| Bahdanau score | eₜⱼ = vₐᵀ tanh(Wₐh_{t-1} + Uₐhⱼ) |
| Attention weights | αₜⱼ = exp(eₜⱼ)/Σⱼ exp(eₜⱼ) |
| Context vector | cₜ = Σⱼ αₜⱼhⱼ |
| Scaled dot-product | eₜⱼ = sₜᵀhⱼ/√d |
| BLEU brevity penalty | BP = min(1, exp(1-ref_len/hyp_len)) |
