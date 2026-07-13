# Code Explanation: Seq2Seq, Attention & Teacher Forcing

**`implementation.py` walkthrough**

---

## 1. Section A — Two Synthetic Tasks, Not One

```python
def make_reversal_data(...):   # Task A: "abcde" -> "edcba"
def make_sorting_data(...):     # Task B: "3143" -> "1334"
```

**Why build TWO tasks instead of one?** Reversal requires the decoder to
attend to a single, PREDICTABLE source position at each step (output
position `t` always needs source position `L-1-t`) — a relatively easy
alignment pattern to learn. Sorting is structurally harder: the correct
source position to attend to at each decoding step depends on the VALUES
in the source sequence, not just their position, requiring the attention
mechanism to do genuine content-based lookup rather than a fixed positional
pattern. Training on both gives a much richer picture of what the attention
mechanism has actually learned than either task alone.

### The Padding Convention: `PAD=0, SOS=1, EOS=2`

```python
PAD, SOS, EOS = 0, 1, 2
```

Reserving the LOWEST indices for special tokens (rather than appending them
at the end) is a common convention — it keeps `PAD=0` aligned with
PyTorch's typical default padding behavior in various utilities, and makes
`ignore_index=PAD` in `CrossEntropyLoss` (Section E) immediately readable.

---

## 2. Section B — Bidirectional Encoder

### Merging Forward and Backward Final States

```python
h_fwd = h[0]; h_bwd = h[1]                        # each: (B,hidden)
h_combined = torch.tanh(self.fc_h(torch.cat([h_fwd, h_bwd], dim=-1)))
```

A bidirectional GRU produces TWO final hidden states — one from reading the
sequence left-to-right, one from right-to-left. Neither alone represents
"the end of the sequence" in a directionally-neutral way. Concatenating both
and passing through a learned linear projection (`fc_h`) lets the network
learn how to best COMBINE both directions' summaries into a single initial
hidden state for the decoder — this is a standard, principled way to bridge
a bidirectional encoder to a unidirectional decoder (the decoder itself must
remain unidirectional, since at generation time future target tokens don't
exist yet).

### Why `pack_padded_sequence` / `pad_packed_sequence`

```python
packed = nn.utils.rnn.pack_padded_sequence(
    x, src_lens.cpu(), batch_first=True, enforce_sorted=False)
enc_out_packed, h = self.gru(packed)
enc_out, _ = nn.utils.rnn.pad_packed_sequence(enc_out_packed, batch_first=True)
```

Within a batch, sequences have DIFFERENT lengths (reversal task samples
range 5-10 characters). Padding shorter sequences to the batch's max length
lets them share a single tensor, but naively running a GRU over the padded
tensor would let the network process meaningless PAD tokens as if they were
real input, corrupting the final hidden state. `pack_padded_sequence`
tells the GRU each sequence's TRUE length, so it processes only the real
tokens and stops exactly at each sequence's actual end — `enforce_sorted=False`
allows this without needing to pre-sort the batch by length, which the
`DataLoader`'s random shuffling would otherwise break.

---

## 3. Section C — Bahdanau Attention With Masking

### Why the Attention Mask Uses `float('-inf')`, Not `0`

```python
if mask is not None:
    scores = scores.masked_fill(mask, float('-inf'))
alpha = F.softmax(scores, dim=1)
```

**Why `-inf` rather than simply zeroing out padded positions' scores?**
`softmax` is computed as `exp(score)/Σexp(score)`. If a padded position's
score were set to `0` (rather than `-inf`), `exp(0)=1` would still
contribute meaningfully to the softmax denominator and could receive a
non-trivial attention weight — incorrectly allowing the decoder to "attend"
to a meaningless PAD position. Setting the score to `-inf` makes
`exp(-inf)=0` exactly, guaranteeing padded positions receive EXACTLY zero
attention weight regardless of the other (real) positions' score
magnitudes — the standard, numerically clean way to exclude padding from
attention.

---

## 4. Section D — The Decoder's Three-Way Concatenation for Output Projection

```python
pred_in = torch.cat([gru_out.squeeze(1), context, emb.squeeze(1)], dim=-1)
logits = self.fc_out(pred_in)
```

**Why does the final output projection see THREE concatenated vectors
(GRU output, attention context, AND the input embedding), rather than just
the GRU's output alone?**
This is a well-established refinement from the original Bahdanau/Luong
attention papers: the GRU's hidden output already incorporates SOME
information from the context vector (since `context` was part of the GRU's
INPUT this step), but concatenating the RAW context vector again directly
into the output projection gives the final classification layer more
DIRECT, undiluted access to what the attention mechanism selected — without
requiring the GRU to perfectly preserve that information through its own
internal gating. Similarly, re-including the raw input embedding gives the
output layer direct access to "what token was just generated," which can
help the network avoid immediately repeating the same token. This
"deep output" pattern consistently improves NMT-style seq2seq models
empirically.

---

## 5. Section E — Scheduled Teacher Forcing

```python
tf_ratio = 1.0 - (epoch / n_epochs)
```

This linearly decays the teacher-forcing ratio from `1.0` (100% teacher
forcing at epoch 0) down to nearly `0.0` (almost pure free-running
generation by the final epoch) — exactly the curriculum-learning strategy
described in theory.md §2. The LIVE training log shows this decay directly
alongside the loss:

```
Epoch  1/30 | tf=1.00 | train=3.1492 | val=2.8828
Epoch 20/30 | tf=0.37 | train=0.0152 | val=0.0035
Epoch 30/30 | tf=0.03 | train=0.0200 | val=0.0094
```

**Why does validation loss BRIEFLY increase around epoch 25 (0.0035→0.0376)
before recovering by epoch 30?** As `tf_ratio` drops below ~0.3-0.4, the
model increasingly must condition on its OWN (possibly imperfect)
predictions rather than the ground truth — a harder training regime that
can transiently INCREASE loss even as the underlying model continues to
improve, before the network adapts to this harder "free-running" objective
and recovers. This is the expected, documented behavior of scheduled
sampling — not a sign of training failure — and is precisely why the
decay is GRADUAL rather than an abrupt switch from 100%→0% teacher forcing.

---

## 6. Section F — BLEU Score Implementation

### Why `min(0, 1 - len(reference)/len(hypothesis))` for the Brevity Penalty

```python
bp = math.exp(min(0, 1 - len(reference)/len(hypothesis)))
```

Per theory.md §7, the brevity penalty should equal `1` (no penalty) when
the hypothesis is AT LEAST as long as the reference, and should shrink
below `1` only when the hypothesis is TOO SHORT. Wrapping the exponent in
`min(0, ...)` enforces exactly this: when `len(hypothesis) ≥
len(reference)`, the term `1 - ref/hyp` is `≥0`, and `min(0, positive) = 0`,
giving `exp(0)=1` (no penalty). Only when the hypothesis is shorter does the
term go negative, correctly triggering `exp(negative) < 1`.

### Why `max(prec, 1e-10)` Before Taking `log`

```python
log_prec += math.log(max(prec, 1e-10)) / max_n
```

If an n-gram precision `prec` is EXACTLY `0` (no matching n-grams at all —
common for higher `n` on short or very wrong sequences), `math.log(0)`
raises a `ValueError` (mathematically, `log(0) = -∞`, which Python doesn't
represent gracefully via `math.log`). Clamping to a tiny positive floor
(`1e-10`) avoids this crash while still producing a very large negative
log-precision contribution, correctly reflecting "essentially zero
precision at this n-gram order" without crashing the whole evaluation loop.

---

## 7. Live Results — Both Tasks Solved Near-Perfectly

```
Task A (Reversal):  Greedy: exact_match=99.0%  BLEU=0.9979
Task B (Sorting):   Greedy: exact_match=100.0%  BLEU=1.0000
```

Both tasks converge to near-perfect exact-match accuracy and BLEU scores
essentially at the ceiling. This confirms the full architecture — bidirectional
encoder, Bahdanau attention with masking, scheduled teacher forcing, and the
deep-output decoder — is correctly implemented and sufficient for both the
POSITION-based alignment task (reversal) and the VALUE-based alignment task
(sorting), which is a meaningfully harder generalization test of the
attention mechanism's genuine content-based lookup capability.

**Why did BEAM SEARCH provide no improvement over GREEDY decoding here
(identical scores on both tasks)?** Beam search's advantage over greedy
decoding is most pronounced when the model is UNCERTAIN at some
decoding steps — situations where a locally-suboptimal token choice could
lead to a better overall sequence. Given the near-perfect BLEU/exact-match
scores, the trained model has become highly CONFIDENT and ACCURATE at
essentially every decoding step for both tasks — there is little
ambiguity left for beam search's broader exploration to meaningfully
exploit. Beam search would be expected to show a clearer advantage on a
harder task where the model retains genuine step-by-step uncertainty.

---

## Pitfalls Avoided

| Pitfall | Fix Applied |
|---|---|
| GRU processing meaningless PAD tokens, corrupting final hidden state | `pack_padded_sequence`/`pad_packed_sequence` |
| Softmax assigning non-zero attention weight to padded positions | `masked_fill(mask, float('-inf'))` before softmax |
| Bidirectional encoder's two final states with no combination rule | Learned `tanh(fc_h([h_fwd;h_bwd]))` projection |
| Random batch order breaking `pack_padded_sequence`'s length-sort assumption | `enforce_sorted=False` |
| Abrupt 100%→0% teacher forcing switch destabilizing training | Linear scheduled decay across all epochs |
| `math.log(0)` crashing BLEU computation on zero-precision n-grams | `max(prec, 1e-10)` floor before `log` |

---
