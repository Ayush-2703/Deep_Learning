# Theory: GPT — Decoder-Only Autoregressive Generation

**Phase 4 — Topic 4 | Deep Learning Mastery Repository**

---

## Table of Contents
1. [The Decoder-Only Architecture](#1-the-decoder-only-architecture)
2. [Next-Token Prediction: The Pretraining Objective](#2-next-token-prediction-the-pretraining-objective)
3. [GPT's Decoder Block vs the Full Transformer's Decoder](#3-gpts-decoder-block-vs-the-full-transformers-decoder)
4. [Autoregressive Generation](#4-autoregressive-generation)
5. [Sampling Strategies](#5-sampling-strategies)
6. [Perplexity: Evaluating Language Models](#6-perplexity-evaluating-language-models)
7. [Why GPT Excels at Generation, BERT Does Not](#7-why-gpt-excels-at-generation-bert-does-not)
8. [KV-Caching: Efficient Autoregressive Inference](#8-kv-caching-efficient-autoregressive-inference)
9. [A Brief Note on Scaling](#9-a-brief-note-on-scaling)

---

## 1. The Decoder-Only Architecture

**Paper:** Radford et al. (2018) — "Improving Language Understanding by
Generative Pre-Training" (GPT-1), with GPT-2/3/4 scaling the same core
architecture substantially.

GPT uses ONLY causally-masked self-attention blocks — no encoder, no
cross-attention (contrast with Topic 2's full encoder-decoder Transformer,
and Topic 3's encoder-only BERT):

```
                  GPT DECODER STACK (N layers)
Token sequence          │
      │                  ▼
      ▼           ┌───────────────┐
 Embedding         │ Masked (causal)│
      +             │ Self-Attention │
Positional          └───────┬───────┘
 Encoding                    │  +residual, norm
      │                       ▼
      │              ┌───────────────┐
      │              │  Feed-Forward  │
      │              └───────┬───────┘
      │                       │  +residual, norm
      │              (repeat N times)
      │                       │
      │                       ▼
      │              Linear + Softmax
      │                       │
      │                       ▼
      │              Next-token probabilities
```

Every position attends ONLY to itself and earlier positions (Topic 1 §7's
causal mask) — there is no bidirectional attention anywhere in the model,
and no separate cross-attention sub-layer, since there is no second
sequence (like an encoder's source) to attend to.

---

## 2. Next-Token Prediction: The Pretraining Objective

```
Given a sequence x₁, x₂, ..., xₜ, predict xₜ₊₁ at EVERY position
SIMULTANEOUSLY (via the causal mask, exactly as Topic 2 §10's parallel
teacher-forcing training):

L = -Σₜ log P(xₜ₊₁ | x₁,...,xₜ)

This is a SINGLE, uniform training objective applied identically at every
position — unlike BERT's MLM (Topic 3 §2), which only computes loss at
a SUBSET of specially-corrupted positions.
```

**Why this objective is a NATURAL fit for causal masking:** at position
`t`, the model has access to EXACTLY the information a real generation
process would have (everything up to and including position `t`) when
predicting position `t+1` — training and actual autoregressive generation
use the IDENTICAL information-availability pattern, unlike BERT's MLM
objective, which trains with bidirectional context that would NEVER be
available during genuine left-to-right generation.

---

## 3. GPT's Decoder Block vs the Full Transformer's Decoder

```
                      Full Transformer Decoder    GPT Decoder
                      (Topic 2 §3)                (this topic)
──────────────────────────────────────────────────────────────
Masked self-attention  YES                          YES
Cross-attention         YES (attends to a            NO (no separate
                        separate ENCODER              source sequence
                        output)                       exists at all)
Feed-forward            YES                          YES
Sub-layers per block    3                             2
```

GPT's decoder block is simply the full Transformer decoder block (Topic 2
§3) with the CROSS-ATTENTION sub-layer removed entirely — there is nothing
for it to attend to, since GPT processes a SINGLE sequence throughout, not
a source-target PAIR of sequences.

---

## 4. Autoregressive Generation

```
1. Start with a PROMPT (or just a beginning-of-sequence token)
2. Run the model forward, get a probability distribution over the next token
3. SELECT a token from this distribution (§5 discusses HOW)
4. APPEND the selected token to the sequence
5. Repeat from step 2, now with the LONGER sequence (including the just-
   generated token) as input
6. Continue until an end-of-sequence token is generated or a maximum
   length is reached
```

This is architecturally IDENTICAL to the greedy/beam decoding procedures
already implemented for the RNN-based Seq2Seq decoder (Phase 3 Topic 3)
and the Transformer's decoder (Topic 2) — the only difference is that GPT
generates from a SINGLE sequence's own continuation, rather than
conditioning on a separate source sequence via cross-attention.

---

## 5. Sampling Strategies

### Greedy Decoding

```
xₜ₊₁ = argmax_v P(v | x₁,...,xₜ)

Deterministic: the SAME prompt always produces the SAME continuation.
Can produce repetitive, overly "safe" text in practice (always picking
the single most likely token at each step doesn't necessarily produce the
most likely FULL SEQUENCE, and tends to lack diversity).
```

### Temperature Sampling

```
P_T(v) = softmax(logits(v) / T)

T=1:     standard softmax (unmodified distribution)
T→0:     approaches greedy/argmax (probability mass concentrates entirely
         on the single highest-logit token)
T→∞:     approaches UNIFORM random sampling over the vocabulary
         (all tokens become equally likely regardless of their logits)
0<T<1:   SHARPENS the distribution (more confident/conservative,
         between standard and greedy)
T>1:     FLATTENS the distribution (more diverse/random, more risk of
         incoherent output)
```

**Why dividing by `T` has this effect:** dividing logits by a small `T`
(e.g., `T=0.5`) DOUBLES their effective magnitude before the softmax —
per Topic 1 §4's variance argument, larger-magnitude logits produce a
MORE peaked (lower-entropy) softmax distribution. The reverse holds for
`T>1`.

### Top-k Sampling

```
1. Compute the full probability distribution over the vocabulary
2. Keep ONLY the k highest-probability tokens, discard the rest (set
   their probability to 0)
3. RENORMALIZE the remaining k probabilities to sum to 1
4. Sample from this restricted, renormalized distribution
```

**Why top-k helps:** it prevents sampling from the "long tail" of very
low-probability, often nonsensical tokens that pure temperature sampling
(especially at higher `T`) can still occasionally select, while still
preserving genuine diversity AMONG the plausible candidates (unlike greedy
decoding's complete determinism).

### Top-p (Nucleus) Sampling — Brief Mention

A related, adaptive alternative: instead of a FIXED count `k`, keep the
SMALLEST set of tokens whose CUMULATIVE probability exceeds a threshold
`p` (e.g., `p=0.9`) — this adapts the effective candidate pool size to the
model's actual confidence at each step (a very peaked distribution needs
few tokens to reach `p`; a very flat distribution needs many), often
preferred over fixed-`k` top-k sampling in practice.

---

## 6. Perplexity: Evaluating Language Models

```
Perplexity = exp( -1/N · Σᵢ log P(xᵢ | x<ᵢ) )
           = exp(average per-token cross-entropy loss)

Equivalently: perplexity is the EXPONENTIAL of the average negative
log-likelihood per token.
```

**Intuition:** perplexity can be interpreted as "the model's effective
branching factor" — how many roughly-equally-likely choices the model
feels it's choosing between, on average, at each position. A perplexity
of `10` roughly means the model is "as confused as if choosing uniformly
among 10 equally likely options" at each step; a PERFECT model (always
100% confident in the correct next token) achieves a perplexity of
exactly `1`; a model with UNIFORM output (no learned structure at all)
over a vocabulary of size `V` achieves a perplexity of exactly `V`.

**Why exponentiate the loss, rather than just reporting the raw
cross-entropy?** Perplexity's exponential scale gives a more intuitively
interpretable number directly tied to "effective vocabulary size," making
it easier to compare across different experimental configurations at a
glance than raw log-loss values.

---

## 7. Why GPT Excels at Generation, BERT Does Not

```
                          GPT                          BERT
──────────────────────────────────────────────────────────────
Training objective        Next-token prediction         Masked LM
                          (matches generation exactly)   (bidirectional,
                                                          mismatches
                                                          generation)
Attention pattern          Causal (left-to-right only)   Full bidirectional
Natural generation         YES -- training and            NO -- would need
                           inference use IDENTICAL         significant
                           information availability        architectural
                                                           adaptation
                                                           (e.g., iterative
                                                           unmasking schemes)
```

This is the DIRECT continuation of Topic 3 §8's comparison table — this
topic empirically demonstrates WHY the "natural generation" column matters
so much: GPT's causal training objective means the model has NEVER seen a
training example where FUTURE context was available, so its behavior at
GENERATION time (where future context is, by definition, unavailable) is
never out-of-distribution relative to training — unlike BERT, whose MLM
training ALWAYS provided bidirectional context, making naive left-to-right
generation with a BERT-style model a fundamentally different (and
typically much lower-quality) regime than what it was trained for.

---

## 8. KV-Caching: Efficient Autoregressive Inference

A brief but important practical note: naive autoregressive generation
(re-running the ENTIRE sequence through the model at every new token, as
our from-scratch implementation does for clarity) is wasteful — the
Key and Value projections for all PREVIOUS positions don't change as new
tokens are appended (since causal masking means earlier positions'
representations never depend on LATER tokens). Production implementations
CACHE these Key/Value tensors across generation steps, only computing the
Query/Key/Value for the SINGLE newest token at each step — reducing
per-step generation cost from `O(L)` (re-processing the whole sequence)
to `O(1)` relative to sequence length, restoring the SAME constant
per-step inference cost that made RNNs attractive (Phase 3 Topic 4 §1),
while retaining the FULL training parallelism advantage that motivated
moving away from RNNs in the first place.

---

## 9. A Brief Note on Scaling

An empirical finding (Kaplan et al. 2020, "Scaling Laws for Neural Language
Models") that has profoundly shaped the field: language model performance
(measured via loss/perplexity) improves SMOOTHLY and PREDICTABLY as a
power-law function of model size, dataset size, and compute budget,
across many orders of magnitude — with no clear sign of the "diminishing
returns" one might naively expect. This empirical regularity was the
primary motivation behind the GPT-2→GPT-3→GPT-4 scaling progression:
rather than requiring qualitatively new architectural breakthroughs, simply
scaling up the SAME decoder-only Transformer architecture (with proportionally
more data and compute) reliably produced better language models — an
observation this repository's necessarily small-scale, CPU-bound
experiments cannot directly demonstrate, but which is essential context
for understanding why the specific architecture covered in this topic
became the foundation for essentially all major large language models.

---

## Key Equations Summary

| Concept | Formula |
|---|---|
| Next-token loss | L = -Σₜ log P(xₜ₊₁\|x₁,...,xₜ) |
| Temperature sampling | P_T(v) = softmax(logits(v)/T) |
| Top-k sampling | restrict to top-k tokens, renormalize, sample |
| Perplexity | exp(-1/N · Σᵢ log P(xᵢ\|x<ᵢ)) = exp(avg cross-entropy) |
| KV-cache generation cost | O(1) per step (vs O(L) naive re-processing) |
