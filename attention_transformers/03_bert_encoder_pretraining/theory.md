# Theory: BERT — Bidirectional Encoder Pretraining

**Deep Learning Mastery Repository**

---

## Table of Contents
1. [BERT's Core Idea: Encoder-Only, Bidirectional](#1-berts-core-idea-encoder-only-bidirectional)
2. [Masked Language Modeling (MLM)](#2-masked-language-modeling-mlm)
3. [The 80/10/10 Masking Strategy](#3-the-801010-masking-strategy)
4. [Input Representation: Token + Position + Segment](#4-input-representation-token--position--segment)
5. [Next Sentence Prediction (NSP)](#5-next-sentence-prediction-nsp)
6. [The [CLS] Token as a Sequence Representation](#6-the-cls-token-as-a-sequence-representation)
7. [Pretraining Then Fine-Tuning](#7-pretraining-then-fine-tuning)
8. [BERT vs GPT: Bidirectional vs Unidirectional Context](#8-bert-vs-gpt-bidirectional-vs-unidirectional-context)

---

## 1. BERT's Core Idea: Encoder-Only, Bidirectional

**Paper:** Devlin et al. (2018) — "BERT: Pre-training of Deep Bidirectional
Transformers for Language Understanding"

BERT uses ONLY the Transformer's ENCODER stack (Topic 2 §2) — no decoder,
no causal masking. Every position's self-attention can see EVERY other
position in the sequence, both to its LEFT and its RIGHT:

```
Standard left-to-right LM (e.g. GPT, Topic 4):
  "The cat sat on the ___"  → predict using ONLY left context

BERT (bidirectional):
  "The cat ___ on the mat"  → predict using BOTH left AND right context
                              ("cat" AND "on the mat" both inform the guess)
```

This bidirectionality is BERT's central advantage for language
UNDERSTANDING tasks (classification, extraction, similarity) — but it
comes at a direct cost: a standard bidirectional encoder CANNOT be used
for autoregressive left-to-right GENERATION (predicting the next token
given only previous tokens), since every position already has access to
the very future tokens a generative model would need to predict. This is
precisely why BERT and GPT (Topic 4) diverged into different architectural
families suited to different task types (§8).

---

## 2. Masked Language Modeling (MLM)

### The Problem With Naively Bidirectional Language Modeling

If BERT simply tried to predict EVERY token using bidirectional context
(as opposed to GPT's next-token prediction), the task would be TRIVIAL —
each position could "cheat" by directly copying its own (unmasked) input
token via a shortcut path in self-attention, since that exact token is
literally present in the input it can see.

### The MLM Solution

```
1. Randomly select ~15% of input token positions
2. REPLACE the token at each selected position according to the 80/10/10
   strategy (§3)
3. Train the model to predict the ORIGINAL (pre-replacement) token at
   EACH selected position, using the (partially corrupted) bidirectional
   context

Loss is computed ONLY over the selected (masked) positions -- NOT over
every position in the sequence (unlike GPT's next-token loss, which
applies to every position).
```

This is directly analogous to a DENOISING AUTOENCODER objective: the model
must reconstruct corrupted/missing information from surrounding context,
forcing it to build genuinely useful representations of language structure
rather than relying on a trivial identity shortcut.

---

## 3. The 80/10/10 Masking Strategy

For each of the ~15% selected positions, the ACTUAL replacement follows:

```
80% of the time:  replace with the special [MASK] token
10% of the time:  replace with a RANDOM token from the vocabulary
10% of the time:  leave the token UNCHANGED (but still predict it!)
```

### Why Not Simply Always Use [MASK] (100% of the Time)?

**The train/inference mismatch problem:** the special `[MASK]` token NEVER
appears in real downstream fine-tuning data (§7) — it's an artifact
specific to the PRETRAINING objective. If the model were trained to expect
`[MASK]` at every position it needs to reason carefully about, it might
learn to ONLY apply its full "careful reasoning" machinery when it SEES
`[MASK]`, and behave differently (worse) on ordinary, un-masked tokens
during downstream fine-tuning — where `[MASK]` never appears at all.

### Why the Other 20% (Random Token / Unchanged)

```
Random token (10%):  forces the model to VERIFY every input token against
                     context, rather than blindly trusting that any
                     non-[MASK] token it sees must be "correct" -- since
                     now a SEEMINGLY ordinary token might actually be
                     wrong and need correcting.

Unchanged (10%):     ensures the model still receives a training signal
                     to predict the CORRECT token even when the input
                     already appears (correctly) unmodified -- biasing the
                     model only mildly toward the observed input while
                     still requiring genuine contextual reasoning at
                     these positions.
```

This specific ratio was determined empirically by the BERT authors as an
effective balance — the general PRINCIPLE (avoid an exact match between
the pretraining corruption process and any single simple downstream
signature) is the reusable insight, more than the precise 80/10/10 split
itself.

---

## 4. Input Representation: Token + Position + Segment

```
Input embedding = TokenEmbedding(token) + PositionEmbedding(position) + SegmentEmbedding(segment)

TokenEmbedding:    standard learned lookup table, indexed by token identity
PositionEmbedding: LEARNED (not fixed sinusoidal, Topic 1 §8) embedding,
                   indexed by absolute position in the sequence
SegmentEmbedding:  indicates which of TWO segments ("sentence A" or
                   "sentence B") each token belongs to -- relevant for the
                   NSP objective (§5); for single-sentence tasks, every
                   token gets the SAME segment embedding
```

All three embeddings are SUMMED (not concatenated) before being fed into
the encoder stack — the network learns to disentangle token-identity,
position, and segment information from this combined signal through
training, the same summation strategy used for combining token embeddings
with positional encoding in Topic 2 §8.

---

## 5. Next Sentence Prediction (NSP)

BERT's ORIGINAL second pretraining objective, alongside MLM:

```
Input: [CLS] Sentence A [SEP] Sentence B [SEP]

Task: binary classification -- is Sentence B the ACTUAL next sentence
      following Sentence A in the original text (50% of training examples),
      or a RANDOM, unrelated sentence (the other 50%)?

Prediction made from the [CLS] token's final representation (§6).
```

**An important historical nuance:** later research (notably the RoBERTa
paper, Liu et al. 2019) found that REMOVING the NSP objective entirely
(training with MLM alone, but on LONGER contiguous text spans) matched or
IMPROVED downstream performance compared to the original BERT's MLM+NSP
combination — suggesting NSP's specific formulation was not as valuable
as originally believed, though the broader idea of incorporating
SENTENCE-level (not just token-level) structure into pretraining remains
an active area explored by later work.

---

## 6. The [CLS] Token as a Sequence Representation

```
Input:  [CLS] token₁ token₂ ... tokenₙ [SEP]

After passing through the full encoder stack, the [CLS] token's FINAL
hidden state serves as an aggregate representation of the ENTIRE
sequence -- used directly as input to a classification head for
sequence-level tasks (sentiment classification, NSP, entailment, etc.)
```

**Why does a token with no inherent "meaning" of its own end up encoding
useful sequence-level information?** Because self-attention lets `[CLS]`
attend to EVERY other position in the sequence (bidirectionally, per §1),
and because it is the DESIGNATED input to the sequence-level pretraining
objective (NSP originally, or any downstream classification head during
fine-tuning), gradient descent shapes `[CLS]`'s representation SPECIFICALLY
to aggregate whatever sequence-level information the training objective
requires — it's a LEARNED aggregation point, not an inherently special
token architecturally.

---

## 7. Pretraining Then Fine-Tuning

```
Pretraining:  Large, UNLABELED corpus + MLM (+NSP) objective
              → learns general-purpose bidirectional language representations

Fine-tuning:  Small, LABELED downstream dataset (e.g. sentiment
              classification, question answering) + task-specific head
              → adapts the pretrained representations to the specific task
```

**This is EXACTLY the transfer learning methodology from Phase 2 Topic 5**,
applied to NLP instead of computer vision: MLM pretraining plays the role
of "training on the large SOURCE task" (Phase 2 Topic 5's ImageNet-style
large source dataset), and downstream fine-tuning plays the role of
"adapting to the small TARGET task." The SAME strategic choices apply
directly:

```
Feature extraction (frozen encoder + new head):  fastest, least
   overfitting risk, best for TINY downstream datasets

Full fine-tuning (unfrozen encoder, LOW learning rate):  typically
   achieves the BEST downstream accuracy given sufficient labeled data,
   at higher compute/overfitting risk

Discriminative learning rates:  lower LR for earlier (more generic)
   encoder layers, higher LR for later layers and the new task head
```

Every one of these strategies, and their trade-offs, transfers DIRECTLY
from Phase 2 Topic 5's vision-domain experiment to this NLP-domain
setting — the transfer learning PRINCIPLE is domain-agnostic, even though
the specific pretrained representations (visual features vs. contextual
token representations) differ completely.

---

## 8. BERT vs GPT: Bidirectional vs Unidirectional Context

```
                           BERT (Encoder-only)          GPT (Decoder-only, Topic 4)
────────────────────────────────────────────────────────────────────────────────────────
Attention direction     Bidirectional (full)            Causal (left-to-right only)
Pretraining objective   Masked Language Modeling        Next-token prediction
Natural use case        Understanding tasks             Generation tasks
                        (classification, extraction,    (open-ended text
                         similarity, QA)                completion, dialogue)
Can generate text        NOT directly (no natural       YES (autoregressive
 autoregressively?       left-to-right generation        generation is the
                         order; would need special       NATIVE training
                         adaptation)                     objective)
```

This is not a strict hierarchy where one architecture is "better" — they
are optimized for FUNDAMENTALLY different task shapes, directly
foreshadowing Topic 4's detailed exploration of the decoder-only,
autoregressive-generation family.

---

## Key Equations Summary

| Concept | Formula / Rule |
|---|---|
| MLM masking rate | ~15% of positions selected |
| 80/10/10 strategy | 80% [MASK], 10% random token, 10% unchanged |
| MLM loss | CrossEntropy computed ONLY over masked positions |
| Input embedding | TokenEmb + PositionEmb(learned) + SegmentEmb |
| NSP task | Binary classification from [CLS]: IsNext vs NotNext |
| Fine-tuning strategies | Feature extraction / Full fine-tune / Discriminative LR (Phase 2 Topic 5) |
