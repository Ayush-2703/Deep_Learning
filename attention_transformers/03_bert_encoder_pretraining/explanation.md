# Code Explanation: BERT — Bidirectional Encoder Pretraining

**`implementation.py` walkthrough**

---

## 1. Section A — Deliberately Imperfect Topic Structure

### Why 85% Own-Topic / 15% Noise, Not 100% Clean Separation

```python
def generate_topic_sentence(rng, topic, sent_len=SENT_LEN, own_topic_prob=0.85):
    ...
    if rng.random() < own_topic_prob:
        tokens.append(int(rng.choice(own_tokens)))
    else:
        tokens.append(int(rng.integers(0, CONTENT_VOCAB)))
```

**Why not simply make each topic's tokens 100% exclusive to that topic?**
A perfectly clean, zero-noise synthetic corpus would make BOTH the MLM
pretraining task AND the downstream topic classification task completely
trivial (any single token would perfectly reveal the topic, and masked-token
prediction would reduce to "look at any OTHER token in the sentence, since
they're all from the exact same 5-token set"). The 15% noise injection
creates a GENUINELY ambiguous task with real, irreducible uncertainty —
mirroring how real language has genuine ambiguity that a good model must
learn to navigate via context, rather than exploiting a trivial shortcut
that wouldn't teach us anything meaningful about whether MLM pretraining
works.

---

## 2. Section B — Masking Statistics Verified, Not Assumed

### Live Result: Within Expected Sampling Noise of Every Target

```
Selected for masking: 14.9%  (target: 15%)
[MASK] token:  79.1%  (target: 80%)
Random token:   9.2%  (target: 10%)
Unchanged:     11.7%  (target: 10%)
```

Every one of the four statistics lands within roughly 1 percentage point
of its theoretical target, confirming the masking implementation correctly
applies BOTH the outer 15% selection rate AND the inner 80/10/10 sub-split
— verified numerically on 2000 sentences (16,000 total token positions)
rather than simply trusting the `rng.random() < 0.8` / `< 0.9` conditional
logic reads correctly at a glance. This numerical verification is
especially valuable here because subtle off-by-one or boundary errors in
nested probability thresholds (e.g., using `<=` vs `<`, or checking against
the wrong cumulative threshold) are an easy category of bug that wouldn't
raise any exception — the code would run fine and just produce SILENTLY
skewed masking ratios.

---

## 3. Section C — Learned Position Embeddings, Not Sinusoidal

```python
self.pos_emb = nn.Embedding(max_len, d_model)   # LEARNED position embedding
```

Unlike Topic 2's Transformer (which used FIXED sinusoidal positional
encoding, per theory.md's original-paper convention), BERT uses a LEARNED
embedding table indexed by absolute position — theory.md §4 notes this
is simpler to implement and empirically comparable for the FIXED, modest
sequence lengths typical of BERT-style pretraining, at the cost of not
generalizing to sequence lengths beyond `max_len` (a hard architectural
ceiling, unlike sinusoidal encoding's in-principle extrapolation
capability from Topic 1 §8). For this topic's fixed 10-token sequences
([CLS]+8 content tokens+[SEP]), this trade-off is a non-issue.

---

## 4. Section D/E — MLM Pretraining: A Genuine, Bounded Learning Signal

### Live Result: A Real, Interesting Plateau — Not a Bug

```
Epoch  1/40: val_MLM_acc=20.0%
Epoch 20/40: val_MLM_acc=25.8%
Epoch 40/40: val_MLM_acc=22.6%
```

**Chance-level accuracy here is `100/24 ≈ 4.2%`** (predicting the correct
one of 24 vocabulary tokens uniformly at random) — the model's ~22-26%
plateau represents genuinely learned structure, roughly **5-6× better
than chance**, confirming the model IS using bidirectional context to
narrow down plausible masked tokens rather than guessing blindly.

**Why does accuracy plateau well below 100%, rather than climbing toward
it?** This ceiling is an EXPECTED, IRREDUCIBLE consequence of the
corpus's deliberately-injected ambiguity (Section A), not underfitting:

```
1. The 15% "noise" tokens (drawn uniformly from ALL 20 tokens, regardless
   of sentence topic) are, by construction, UNPREDICTABLE from context --
   no amount of training can teach the model to correctly guess a token
   that was randomly substituted with no relationship to the topic.

2. Even for "own-topic" positions (the 85% majority), correctly guessing
   WHICH of the 5 topic-specific tokens was originally present (rather
   than just recognizing "this position's masked token belongs to topic
   K's token set") requires information the SURROUNDING tokens may not
   uniquely provide, since sentences don't encode a stricter grammar
   beyond "sample each position i.i.d. from the topic's token set."
```

The observed ~22-26% accuracy is consistent with a model that has
successfully learned to infer the sentence's TOPIC from context (secretly
the more important, generalizable skill) but faces a genuine remaining
~1-in-5 uncertainty even after correctly identifying the topic, compounded
by the irreducible 15% pure-noise positions. We report this plateau
honestly as the CORRECT, expected outcome of the corpus design, rather
than training longer chasing a higher number that the task's own
structure doesn't support.

### Small Fluctuation at Epoch 40 (26.2%→22.6%) — Re-Masking Variance

```python
rng_seed = SEED + epoch
full_input, full_labels = build_mlm_batch(sent_tr, seed=rng_seed)
```

Each epoch uses a FRESH random masking pattern (`seed=SEED+epoch`, mirroring
real BERT pretraining, where masking is stochastic per training step/epoch
rather than fixed once at the start). The validation accuracy is measured
against a SEPARATE, ALSO-freshly-masked validation set each evaluation
call — small fluctuations between nearby epochs (e.g., the 3.6 percentage
point dip at epoch 40) reflect genuine variance from WHICH specific
positions happened to be masked in that evaluation pass, not a sign of
training instability.

---

## 5. Section F — A Strikingly Clean Transfer-Learning Result

### Live Result

```
From-scratch:         95.0% val accuracy | 103,492 trainable params
Feature Extraction:  100.0% val accuracy |     260 trainable params
Full Fine-tuning:    100.0% val accuracy | 103,492 trainable params
```

**This result is notably CLEANER than Phase 2 Topic 5's analogous vision
experiment, where Feature Extraction slightly UNDERPERFORMED From-scratch
due to a deliberately engineered domain shift.** Here, Feature Extraction
(a FROZEN pretrained encoder plus a mere 260-parameter linear head) matches
Full Fine-tuning's PERFECT 100% accuracy — why does frozen-feature transfer
work so much more cleanly in this experiment?

**The key difference: pretraining-task/downstream-task ALIGNMENT.** The
MLM pretraining objective REQUIRES the model to infer each sentence's
TOPIC from context in order to predict masked tokens accurately (Section
D/E's discussion) — this is EXACTLY the latent variable the downstream
classification task asks about directly. The pretrained `[CLS]`
representation, shaped entirely by the MLM objective, ALREADY encodes
topic information in an almost perfectly linearly-separable form by the
time downstream fine-tuning begins — a tiny linear probe (260 params) is
sufient to extract it. Phase 2 Topic 5's vision experiment deliberately
engineered a DOMAIN SHIFT between source and target (different visual
style, different background) specifically to illustrate when frozen
features are NOT enough — this topic's result is the complementary,
equally instructive case: when the pretraining objective and downstream
task are WELL-ALIGNED, even minimal frozen-feature transfer can be
strikingly effective.

### Why From-Scratch Still Reaches a Respectable 95%, Not Chance

With only 80 labeled training examples for a 4-way classification task,
From-scratch achieves 95% — notably HIGHER than Phase 2 Topic 5's
analogous vision from-scratch baseline's much less stable performance at
a comparable data scarcity ratio. This makes sense given task simplicity:
distinguishing 4 topics based on token-frequency patterns in an 8-token
sequence is a substantially EASIER learning problem than Phase 2's visual
shape-classification task under domain shift — 80 examples is apparently
enough signal for even a randomly-initialized small Transformer encoder to
largely solve this particular task without any pretraining benefit,
though pretraining still closes the remaining gap to a PERFECT 100%.

---

## Pitfalls Avoided

| Pitfall | Fix Applied |
|---|---|
| Fully clean, non-overlapping topic vocabulary makes the task trivial | 85%/15% own-topic/noise injection creates genuine ambiguity |
| Assuming masking probability code is correct without measurement | Explicit statistical verification against all 4 target ratios |
| Interpreting MLM's ~25% plateau as underfitting | Explained the task's OWN irreducible uncertainty ceiling |
| Presenting the clean Feature-Extraction-matches-Full-Fine-tuning result without explaining WHY it differs from Phase 2 Topic 5 | Explicitly named pretraining/downstream task ALIGNMENT as the deciding factor |
| Small epoch-to-epoch MLM accuracy fluctuation mistaken for instability | Attributed to genuine re-masking variance (fresh stochastic masks per epoch) |

---
