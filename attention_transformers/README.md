<div align="center">

![Phase 4: Attention & Transformers](https://capsule-render.vercel.app/api?type=waving&color=0:0B0C0E,50:363B45,100:586174&height=200&section=header&text=Phase%204:%20Attention%20%26%20Transformers&fontSize=30&fontColor=ffffff&fontAlignY=25&animation=fadeIn&desc=Deep%20Learning&descSize=25&descAlignY=58)

**Made with ❤️ by [Ayush Kumar Singh](https://github.com/Ayush-2703)**

</div>

---

Five topics tracing the attention family from the raw mechanism to its
three dominant architectural descendants: scaled dot-product & multi-head
attention → the full encoder-decoder Transformer → BERT (encoder-only,
bidirectional) → GPT (decoder-only, autoregressive) → the Vision
Transformer (attention applied to images). Everything runs CPU-only on
synthetic data — no internet dataset downloads, no pretrained checkpoints.
Every `implementation.py` was actually executed end-to-end during this
build, including several verifications against PyTorch's own internals
(`nn.MultiheadAttention`, `F.scaled_dot_product_attention`) rather than
just against hand-written reference code. Each topic's `explanation.md`
reports the real numbers that came out — including a genuine overfitting
curve, an inconclusive Pre-LN vs Post-LN comparison, and a near-uniform
attention-weight result — kept in and explained rather than hidden (see
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
- [The Recurring Crossover](#the-recurring-crossover-attention-vs-recurrence-measured-twice)
- [Notable Engineering Detours and Honest Findings](#notable-engineering-detours-and-honest-findings-see-each-topics-explanationmd-for-full-detail)
- [Running the Code](#running-the-code)

---

## Topics

| # | Topic | Data | Core result |
|---|-------|------|-------------|
| 01 | Attention Mechanisms — Scaled Dot-Product & Multi-Head | Synthetic Q/K/V variance sweeps, a max-value-finding sequence task | Multi-head self-attention matches PyTorch's `nn.MultiheadAttention` exactly on **both** output and attention weights; empirically confirms `Var(q·k)=d_k` growing linearly with dimension, and shows an unscaled dot-product collapsing softmax to a hard one-hot `[0,0,1]` vs. the scaled version staying genuinely soft (`[0.19, 0.10, 0.71]`); a direct wall-clock crossover measurement shows attention ~1.3× faster than an LSTM at length 16, but **~11× slower** by length 512 |
| 02 | The Transformer Architecture | Synthetic digit-reversal sequences | Full encoder-decoder Transformer verified sub-layer-by-sub-layer against PyTorch internals; two real bugs found and fixed (batched greedy decoding not stopping per-sequence at EOS; a positional-encoding buffer sized too small for the eval-time decode budget); exact-match accuracy degrades sharply with target length under a fixed 80-epoch budget (76% at length 5 → 2.7% at length 25); a Pre-LN vs. Post-LN comparison came back **genuinely inconclusive** at this shallow 4-layer scale, reported as such rather than forced to match the textbook expectation |
| 03 | BERT — Bidirectional Encoder Pretraining | Synthetic 4-topic sentence corpus with 15% injected noise | Masking statistics (15% selection, 80/10/10 sub-split) verified numerically to within ~1 percentage point of every target; MLM validation accuracy plateaus at 22–26% against a 4.2% chance baseline — explained as the corpus's own deliberately irreducible ambiguity, not underfitting; downstream topic classification hits a clean **100%** with just a 260-parameter frozen linear probe on the pretrained `[CLS]` token, matching full fine-tuning exactly |
| 04 | GPT — Decoder-Only Autoregressive Generation | Synthetic Markov "counting" sequences (`(prev+1) % 10` with 80% probability) | Causal masking verified to exact `0.00e+00` attention on future positions; a genuine, honestly-reported overfitting curve (train loss `1.09→0.69` while validation perplexity *worsens* `2.62→3.35` over 40 epochs); trained perplexity (3.35) lands **75.7%** of the way from a uniform-random baseline (10.0) to the derived theoretical-best bound (1.22); GPT trains slower *and* scores worse validation perplexity than an LSTM at this task's short 15-token length — a direct, quantitative confirmation of Topic 01's own measured attention/RNN crossover, not a contradiction of Transformer theory |
| 05 | Vision Transformer (ViT) | Synthetic 32×32 RGB shape-classification images (same task as Phase 2's CNNs) | Patch embedding proven mathematically identical to a stride-`P` convolution, verified to exact floating-point match; trained ViT reaches **97.5%** validation accuracy — squarely inside Phase 2's CNN range (93–100%) — but only with adequate data (800 images); a dedicated data-efficiency sweep (5/15/40/100 images/class) shows CNNs saturating at 100% by 40/class while ViT still trails at 89.3% even at 100/class, a clean confirmation of ViT's missing locality/translation-equivariance inductive bias |

---

## Architecture Cheat Sheet

Five different Transformer configurations sit inside this phase alone —
here's what actually distinguishes them, verified rather than assumed:

| | Full Transformer (02) | BERT (03) | GPT (04) | ViT (05) |
|---|---|---|---|---|
| **Stack** | Encoder + Decoder | Encoder only | Decoder only (no cross-attention) | Encoder only |
| **Attention pattern** | Bidirectional (enc.) + causal (dec.) + cross-attention | Bidirectional | Causal (left-to-right) only | Bidirectional |
| **Pretraining objective** | Sequence-to-sequence (translation-style) | Masked Language Modeling | Next-token prediction | *(none — supervised classification here)* |
| **Input tokens** | Text tokens | Text tokens | Text tokens | Image patches, via a strided convolution |
| **Natural use case** | Sequence-to-sequence tasks | Understanding (classification, extraction) | Open-ended generation | Image classification |
| **Native autoregressive generation?** | Yes (via the decoder) | No — bidirectional context never matches generation | Yes — training and inference share the identical information pattern | N/A |

The throughline: every one of these is the *same* self-attention +
feed-forward building block from Topic 01. What changes across rows is
never the core mechanism — only which positions are allowed to attend to
which, and what objective shapes the result.

---

## The Recurring Crossover: Attention vs. Recurrence, Measured Twice

Topic 01 measured attention's wall-clock cost against an LSTM directly and
found a crossover somewhere between length 32 and 64 — attention faster on
short sequences, dramatically slower (~11×) by length 512, a direct
consequence of `O(L²)` vs. `O(L)` complexity. Topic 04 then trained an
actual GPT model against an LSTM on 15-token sequences — well below that
crossover point — and found GPT was **both slower to train and scored
worse validation perplexity** than the LSTM, with less than a quarter of
its parameter efficiency. Read together, these aren't two isolated
results: Topic 04's outcome is exactly what Topic 01's own measurement
already predicted for a task this short. "Transformers train faster than
RNNs" is true asymptotically and at the sequence lengths real large
language models use — not a universal law true at every scale, and this
phase demonstrates precisely where that claim's applicability begins
rather than just asserting it.

---

## Notable Engineering Detours and Honest Findings (see each topic's `explanation.md` for full detail)

1. **Topic 01**: before writing the from-scratch multi-head attention
   implementation, `nn.MultiheadAttention`'s internals were probed
   empirically (weight-stacking layout, contiguous head-splitting,
   boolean mask convention) rather than guessed from documentation alone
   — turning what could have been an iterative guess-and-check loop into
   a correct implementation on the first attempt.
2. **Topic 01**: the `√d_k` scaling argument was confirmed empirically,
   not just derived — raw dot-product variance tracks `d_k` almost
   exactly across three orders of magnitude (`d_k=4→Var≈4`,
   `d_k=256→Var≈247`), while the scaled variance stays pinned near `1.0`
   regardless of dimension.
3. **Topic 02**: batched greedy decoding originally only stopped once
   *every* sequence in a batch hit EOS — already-finished sequences kept
   generating extra, meaningless tokens for several more steps. Fixed by
   truncating each sequence independently at its own first EOS during
   evaluation, rather than adding per-sequence stop-state tracking to the
   decoding loop itself.
4. **Topic 02**: an initial 5.3% exact-match result at 25 epochs was
   investigated systematically (60, then 100 epochs at the same length)
   before concluding it was a training-budget issue, not an architecture
   bug — confirmed once 60 epochs alone lifted exact-match to 68%.
5. **Topic 02**: the Pre-LN vs. Post-LN comparison came back with Pre-LN
   winning on final loss but *losing* on training-loss stability — the
   opposite of the textbook expectation — attributed honestly to the
   experiment running at a shallow 4-layer depth, well below where the
   Pre-LN/Post-LN stability literature's claims are actually about
   (training failure at much greater depths), not treated as a
   refutation of the theory.
6. **Topic 03**: the 80/10/10 MLM masking sub-split was verified
   numerically on 16,000 token positions (79.1% / 9.2% / 11.7% against
   80/10/10 targets) rather than trusted from reading the conditional
   logic — this class of nested-probability bug produces no exception,
   just silently skewed ratios.
7. **Topic 03**: BERT's frozen-feature transfer (260-parameter linear
   probe) matched full fine-tuning at a perfect 100%, noticeably cleaner
   than Phase 2 Topic 5's analogous vision experiment — traced to
   pretraining/downstream-task *alignment*: MLM pretraining already
   forces the model to infer sentence topic, which is exactly what the
   downstream classifier asks for directly.
8. **Topic 04**: training loss improved every epoch (`1.09→0.69`) while
   validation perplexity *worsened* (`2.62→3.35`) — a textbook
   overfitting curve, reported in full and used as the final model for
   every downstream section rather than silently swapping in an
   earlier, better-validation checkpoint.
9. **Topic 04**: temperature-1.5 sampling's rule-adherence (69.2%) came
   in *below* the training data's own native 80% noise rate — flagged
   explicitly, since high-temperature sampling compounds the model's
   residual uncertainty with additional flattening rather than merely
   reproducing the data's inherent randomness.
10. **Topic 05**: patch embedding was verified as an *exact* numerical
    match to a strided convolution (not merely "conceptually similar"),
    reusing the same weight-reshape verification discipline as Phase 2's
    im2col check.
11. **Topic 05**: final-layer `[CLS]` attention weights came back nearly
    uniform (max weight 0.066 vs. a uniform baseline of 0.0625) on a
    correctly-classified example — reported honestly as a genuine
    "attention is not explanation" finding rather than replaced with a
    more dramatic-looking cherry-picked example.

---

## Running the Code

```bash
cd 0X-topic-name/
python3 implementation.py
```

Requires: `torch`, `numpy`, `matplotlib`, `Pillow` (Topic 05 only, for
synthetic shape-image generation). CPU-only, no GPU/CUDA needed, no
internet dataset downloads or pretrained checkpoints — every task in this
phase is procedurally generated.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0B0C0E,50:363B45,100:586174&height=70&section=footer" width="100%"/>

</div>
