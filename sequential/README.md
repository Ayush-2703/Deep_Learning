<div align="center">

![Phase 3: Sequential Modeling](https://capsule-render.vercel.app/api?type=waving&color=0:0B0C0E,50:363B45,100:586174&height=200&section=header&text=Phase%203:%20Sequential%20Modeling%20Fields&fontSize=30&fontColor=ffffff&fontAlignY=25&animation=fadeIn&desc=Deep%20Learning&descSize=25&descAlignY=58)
</br>
**Made with ❤️ by [Ayush Kumar Singh](https://github.com/Ayush-2703)**

</div>

---

Four topics tracing sequence modeling from first principles to the current
state of the art: vanilla RNNs → gated LSTM/GRU → attention-based Seq2Seq →
State Space Models (S4 and Mamba). Everything runs CPU-only on synthetic
data (marker sequences, parity strings, sine waves, string reversal/sorting
pairs) — no internet dataset downloads. Every `implementation.py` was
actually executed end-to-end during this build, including a manual NumPy
backprop-through-time implementation cross-checked against PyTorch's
autograd to floating-point precision. Each topic's `explanation.md` reports
the real numbers that came out — including three separate results across
this phase where a gated or "smarter" architecture unexpectedly failed to
beat a simpler one, kept in and explained rather than hidden (see
"Notable engineering detours" below).

Every topic follows the repository's 3-file structure:

```
0X-topic-name/
├── readme.md           ← Full derivations, ASCII diagrams, historical context
├── implementation.py   ← Runnable PyTorch/NumPy code
└── explanation.md      ← Line-by-line walkthrough + live results
```

## 📌 Table of Contents

- [Topics](#topics)
- [Architecture Cheat Sheet](#architecture-cheat-sheet)
- [The Recurring Benchmark](#the-recurring-benchmark-one-task-four-architectures)
- [Notable Engineering Detours and Honest Findings](#notable-engineering-detours-and-honest-findings-see-each-topics-explanationmd-for-full-detail)
- [Running the Code](#running-the-code)

---

## Topics

| # | Topic | Data | Core result |
|---|-------|------|-------------|
| 01 | RNNs — Vanilla Recurrence & BPTT | NumPy AND-style marker sequences, parity strings, sine waves | Manual NumPy backprop-through-time matches PyTorch's autograd exactly (float64, exact-match verified); gradient norm collapses from 2.40e-02 at length 5 to a literal float32 zero by length 200 — a 9+ order-of-magnitude exponential decay; long-range signal-detection accuracy is **bimodal, not smoothly degrading** — 100% at lengths 5–30 and again at 75, but 54–55% (chance level) at lengths 50 and 100 |
| 02 | LSTM & GRU — Gated Recurrence | Same marker/parity suite plus the classic Adding Problem | At length 200, gradient norms are LSTM=2.59e-43 and GRU=1.61e-36 versus vanilla RNN's exact 0.0 — gating measurably **delays**, but doesn't eliminate, vanishing gradients; on the Adding Problem, GRU decisively wins at long range (MSE 0.0011–0.0018 vs. RNN/LSTM stuck near the baseline-mean MSE of 0.14–0.18); yet on implicit signal detection (no explicit "remember this" marker), **all three architectures** still collapse to 47–57% — chance level |
| 03 | Seq2Seq, Attention & Teacher Forcing | Two synthetic tasks: string reversal (position-based alignment) and digit sorting (value-based alignment) | A bidirectional-GRU encoder with Bahdanau attention, padding masks, and scheduled teacher forcing reaches **99.0% exact-match / BLEU 0.998** on reversal and a perfect **100% exact-match / BLEU 1.000** on sorting — the structurally harder, content-dependent alignment task; beam search (k=5) matches greedy decoding exactly once the model is this confident, adding zero measurable improvement |
| 04 | State Space Models — S4 & Mamba | Continuous ↔ discrete ↔ convolutional-view validation, HiPPO stability check, and the same signal-detection benchmark from Topics 1–2 | Zero-order-hold discretization validated to **0.0074% relative error** against a 1000×-finer Euler reference; the recurrent and convolutional views of a linear SSM are proven mathematically identical, matching to **2.78e-16** — literal float64 machine epsilon; but the Selective SSM (Mamba) is **no exception** to Topic 2's finding — it too lands near chance (49.5–57%) on implicit signal detection, extending the "architectural capacity ≠ guaranteed learning" conclusion to a fourth architecture family |

---

## Architecture Cheat Sheet

Derived and empirically checked across this phase's four topics — not a
textbook table copied in from elsewhere:

| | RNN / LSTM / GRU | *(Transformer, for context — Phase 4)* | S4 (LTI SSM) | Mamba (Selective SSM) |
|---|---|---|---|---|
| **Training parallelism** | None — strictly sequential | Full — all positions at once | Full — via FFT convolution | Full — via parallel associative scan |
| **Training complexity** | O(L) sequential steps | O(L²) | O(L log L) | O(L) |
| **Inference complexity** | O(1)/step, constant state | O(L)/step — grows with context | O(1)/step, constant state | O(1)/step, constant state |
| **Content-based selection** | Limited (via gates, LSTM/GRU only) | Native (attention) | None — fixed A, B, C per layer | Yes — Δ, B, C are functions of the input |
| **Long-range memory** | Poor (vanilla), better with gating | Strong (direct attention to any position) | Excellent in principle (HiPPO init) | Good in principle; empirically no better than gated RNNs on *implicit* signals in this phase's tests |

No single row dominates every column — which is exactly why RNNs, attention,
and SSMs remain three genuinely different tools rather than a strict
progression from "worse" to "better."

---

## The Recurring Benchmark: One Task, Four Architectures

The same **signal-detection task** — remember a single bit planted at
position 0 across an otherwise uninformative noisy sequence, with no
explicit marker flagging it as important — was run against all four
architectures in this phase, specifically to make results comparable:

| Length | Vanilla RNN | LSTM | GRU | Selective SSM (Mamba) |
|---|---|---|---|---|
| 50  | 100.0% *(reliability-cliff spike)* | 51% | 53% | 53.5% |
| 100 | 55%  | 53% | 53% | 57.0% |
| 150 | 57%  | 53% | 47% | 49.5% |

Every architecture in this repository — gated or not, attention-free or
selective — struggles on this exact task once the sequence gets long,
because the difficulty here isn't vanishing gradients or fixed dynamics;
it's that **nothing in the input tells the network what to remember**. The
Adding Problem (Topic 2), by contrast, has an explicit marker channel and
is solved cleanly by every gated architecture. Read together, Topics 1, 2,
and 4 make the same point from three different angles: architectural
capacity for long-range memory is necessary but not sufficient — the
training signal's clarity matters just as much as the mechanism.

---

## Notable Engineering Detours and Honest Findings (see each topic's `explanation.md` for full detail)

1. **Topic 01**: PyTorch's `nn.RNN` splits its bias into two separately
   stored, internally-summed terms (`bias_ih_l0`, `bias_hh_l0`). Verifying
   the manual NumPy implementation against PyTorch required zeroing one
   term explicitly and casting both sides to `float64`, so precision
   couldn't masquerade as a logic bug.
2. **Topic 01**: signal-detection accuracy turned out **bimodal**, not a
   smooth decay curve — length 75 hits 100% sandwiched between two
   near-chance failures at 50 and 100. Reported exactly as measured, since
   it reflects real sensitivity to random initialization under vanishing
   gradients rather than a cherry-picked curve.
3. **Topic 02**: PyTorch's actual GRU formula applies the reset gate
   *after* the hidden-to-hidden projection (`r ⊙ (W_hh·h + b_hh)`), not
   before it as most textbook derivations present (`W_hh·(r⊙h)`) — caught
   empirically when the textbook version failed the `np.allclose`
   weight-injection check against PyTorch.
4. **Topic 02**: GRU beats LSTM decisively on the long-range Adding
   Problem (MSE 0.0011 vs. 0.1832 at length 100) — attributed to GRU's
   simpler, less-squashed gradient path, and reported as a genuine finding
   rather than an anomaly to explain away.
5. **Topic 03**: the attention mask uses `float('-inf')` rather than `0`
   before softmax, so padded positions receive *exactly* zero attention
   weight regardless of the magnitude of real positions' scores elsewhere.
6. **Topic 03**: validation loss briefly rises around epoch 25 as the
   scheduled-sampling ratio drops the model onto its own (imperfect)
   predictions — a known, expected side effect of curriculum-style teacher
   forcing decay, not a sign of a broken run, and it recovers by epoch 30.
7. **Topic 04**: Mamba's state matrix is parametrized as `A = -exp(A_log)`,
   which guarantees strict negativity — and therefore stability — for any
   real-valued learnable parameter, rather than learning `A` as a raw,
   unconstrained value that gradient descent could push positive.
8. **Topic 04**: the same signal-detection benchmark from Topics 1–2 was
   rerun against the Selective SSM and produced the same near-chance result
   (49.5–57%) at longer lengths — reported as a shared limitation across
   all four architecture families in this phase, not adjusted until Mamba
   "won."

---

## Running the Code

```bash
cd 0X-topic-name/
python3 implementation.py
```

Requires: `torch`, `numpy`, `matplotlib`. CPU-only, no GPU/CUDA needed, no
internet dataset downloads — every task in this phase is procedurally
generated (markers, parity strings, sine waves, reversal/sorting pairs).

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0B0C0E,50:363B45,100:586174&height=70&section=footer" width="100%"/>

</div>
