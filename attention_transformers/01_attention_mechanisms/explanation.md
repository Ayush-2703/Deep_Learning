# Code Explanation: Attention Mechanisms

**`implementation.py` walkthrough**

---

## 1. Section A — Scaled Dot-Product Attention From Scratch

### Numerical Stability: Subtracting the Max Before `exp`

```python
scores = scores - scores.max(axis=-1, keepdims=True)   # numerical stability
exp_scores = np.exp(scores)
```

This is the same softmax stability trick from Phase 1 Topic 2's theory —
subtracting each row's maximum before exponentiating doesn't change the
final softmax result (the subtraction cancels in the normalization), but
prevents `np.exp` from overflowing on large positive scores. This matters
MORE here than in a typical classifier, since attention scores can span a
wide range depending on `d_k` (exactly the problem Section B explores).

### Why Verify Against `F.scaled_dot_product_attention`, Not a Manual Torch Loop

```python
out_torch = F.scaled_dot_product_attention(Q_t, K_t, V_t)[0,0].numpy()
```

PyTorch 2.x ships a built-in, heavily-optimized `scaled_dot_product_attention`
function (used internally by `nn.MultiheadAttention` and modern Transformer
implementations) that implements EXACTLY the formula from theory.md §3.
Verifying against this built-in function — rather than another hand-written
PyTorch loop that could share the same bug as our NumPy version — gives
genuine, independent confirmation that our from-scratch formula matches
the field's standard, battle-tested implementation.

---

## 2. Section B — The Variance Argument, Confirmed Empirically

### Live Result: Theory Matches Measurement Almost Exactly

```
d_k=4:    raw Var=4.04    (theory: 4)
d_k=64:   raw Var=65.18   (theory: 64)
d_k=256:  raw Var=246.74  (theory: 256)
```

Across three orders of magnitude in dimension, the empirically measured
variance of the raw dot product tracks the theoretical prediction
`Var(q·k)=d_k` almost exactly (small deviations are expected sampling
noise from finite `n_trials=5000`). The SCALED variance, meanwhile, stays
pinned near `1.0` regardless of `d_k` — confirming theory.md §4's claim
that dividing by `√d_k` makes the attention score distribution
DIMENSION-INDEPENDENT.

### The Illustrative Softmax Saturation Example

```
Raw scores:    [ 6.81 -3.3  28.13]  -> softmax: [0. 0. 1.]
Scaled scores: [ 0.43 -0.21  1.76]  -> softmax: [0.1879 0.0999 0.7122]
```

**Why does the raw-score softmax collapse to an exact one-hot `[0,0,1]`
while the scaled version remains a genuinely soft distribution?** At
`d_k=256`, an UNSCALED dot product of `28.13` is roughly `1.76×√256=28.16`
— i.e., the RAW score is `√d_k` times LARGER than the scaled score. Since
softmax involves `exp()`, this multiplicative blow-up in the input
translates to an exponential blow-up in the OUTPUT ratio between the
largest and second-largest scores — `exp(28.13)/exp(6.81)` is astronomically
larger than `exp(1.76)/exp(0.43)`, causing the unscaled softmax to
effectively become a hard `argmax` (one weight ≈1, all others ≈0). This
directly illustrates WHY unscaled attention would produce near-useless
gradients: a near-one-hot distribution means the gradient with respect to
all the NON-selected keys/values is nearly zero, starving those pathways
of any learning signal — exactly the softmax-saturation failure mode from
Phase 1 Topic 2's theory, now shown concretely in the attention context.

---

## 3. Section C — Multi-Head Attention, Verified to Match PyTorch Exactly

### Why We Probed `nn.MultiheadAttention`'s Internals BEFORE Writing This Code

Before writing `MultiHeadAttentionScratch`, we ran small standalone probes
against `nn.MultiheadAttention` to empirically determine THREE
implementation details not fully pinned down by reading the class
documentation alone:

```
1. Weight layout:  in_proj_weight stacks [Wq;Wk;Wv] as ONE (3*embed_dim,
                    embed_dim) matrix (confirmed via .named_parameters())
2. Head splitting: heads take CONTIGUOUS chunks of embed_dim, not an
                    interleaved or otherwise-permuted split (confirmed by
                    injecting identity weights and tracing which output
                    dimensions responded to which one-hot input dimensions)
3. Mask convention: attn_mask is a BOOLEAN mask where True=forbidden,
                    matching the additive -inf convention from Phase 3
                    Topic 3's Bahdanau attention masking
```

This mirrors the exact debugging discipline used for LSTM/GRU's internal
gate-order conventions in Phase 3 Topic 2 — rather than guessing an
implementation detail and discovering a mismatch only after the
verification assertion fails, we determined PyTorch's actual convention
FIRST via small, targeted experiments, then wrote the from-scratch
implementation to match it directly. This turned what could have been an
iterative guess-and-check debugging loop into a single correct
implementation on the first attempt.

### Live Result: Exact Match on BOTH Output and Attention Weights

```
Output match: True
Attention weights match: True
```

This is a stronger verification than Section A's — it confirms not only
that the FINAL output matches, but that the INTERMEDIATE attention weight
matrices (shape `(batch, heads, Lq, Lk)`) are identical, meaning every
architectural detail (weight stacking order, head-splitting axis, per-head
scaling) was reproduced correctly, not just coincidentally producing the
same final numbers through a different (but mathematically equivalent)
computation path.

---

## 4. Section D — Causal Masking, Verified to Exact Zero

### Why Check `future_weights.max()` Rather Than the Full Matrix

```python
future_weights = attn_avg[causal_mask.numpy()]
print(f"Max attention weight assigned to any FUTURE position: {future_weights.max():.2e}")
```

Using the boolean `causal_mask` array to directly INDEX the attention
weight matrix extracts EXACTLY the upper-triangular (forbidden) entries as
a flat array, regardless of the mask's specific shape — checking the
MAXIMUM of this flat array is a strict, single-number test: if even ONE
forbidden position received ANY non-zero weight (a masking bug), the
maximum would be greater than zero and the assertion would fail
immediately. The live result `0.00e+00` confirms EVERY forbidden position
received exactly zero weight, not merely a very small one.

---

## 5. Section E — Positional Encoding's Three Properties, All Verified

### The Relative-Position Linearity Test: Train/Test Split on POSITIONS

```python
pos_train = np.arange(0, 60)
pos_test = np.arange(60, 90)
M_k, _, _, _ = np.linalg.lstsq(PE_pos_train, PE_posk_train, rcond=None)
...
fit_err = np.max(np.abs(PE_posk_test_true - PE_posk_test_pred))
```

**Why fit the linear transformation `M_k` on ONE range of positions
(0-59) and test it on a COMPLETELY DIFFERENT, held-out range (60-89)?**
This is the critical test for whether the linear relationship
`PE(pos+k) ≈ M_k·PE(pos)` is a genuine property of the encoding SCHEME
itself (true for ANY `pos`, with the SAME `M_k`), versus merely an
artifact of overfitting `M_k` to the SPECIFIC positions it was fit on. If
`M_k` were fit and tested on the SAME positions, a sufficiently flexible
linear fit could trivially achieve near-zero error even for an encoding
scheme where the relationship DOESN'T actually hold at other positions —
similar in spirit to why Phase 1 Topic 6 insists on separate train/validation
splits for hyperparameter selection. The held-out test achieving `2.22e-07`
error (essentially floating-point noise) confirms the SAME `M_k`
transformation correctly predicts `PE(pos+5)` from `PE(pos)` at positions
the fit never saw — direct empirical proof of theory.md §8's claimed
linearity property, not just a restatement of the trigonometric identity
that predicts it.

---

## 6. Section F — Content-Based Lookup: A Clean, Verifiable Attention Demo

### Why No Positional Encoding Is Used in This Task, Deliberately

```python
class MaxFinderAttention(nn.Module):
    def __init__(self, d_model=16):
        ...
        # No positional encoding added anywhere
```

**Why omit positional encoding here, given Section E just demonstrated its
value?** This task — "find the position holding the maximum VALUE" — is
DELIBERATELY designed to be solvable using PURE content-based lookup: the
correct answer depends ENTIRELY on the VALUES in the sequence, never on
WHERE they appear. Including positional encoding would muddy this
demonstration by giving the model an alternative (positional) signal it
could potentially exploit instead of learning genuine content-based
attention. Omitting it deliberately isolates and stress-tests the
core query-key matching mechanism from theory.md §2 in its purest form.

### The Learned Query Token: A Single Fixed "Question" Vector

```python
self.query_token = nn.Parameter(torch.randn(1, 1, d_model))
...
q = self.query_token.expand(B, -1, -1)
```

**Why is there only ONE learned query vector, shared across every input
sequence and every batch, rather than deriving a query from the input
itself (as in typical self-attention)?** This task has a FIXED "question"
— "which position has the maximum value?" — that doesn't change across
examples. A single, globally-shared, LEARNED query vector represents
exactly this fixed question; through training, gradient descent shapes
this vector's direction so that its dot product with each position's KEY
(derived from that position's VALUE) is HIGHEST precisely when that
position holds the maximum. This is a minimal, clean illustration of
`Attention(Q,K,V)` (theory.md §2) where `Q` need not come from the same
place as `K,V` — foreshadowing exactly the cross-attention structure
Topic 2's full Transformer decoder will use.

### Live Result: 100% Attention-Argmax Match — Genuine Content-Based Lookup Confirmed

```
Validation value-prediction MSE: 0.000069
Attention-argmax matches TRUE argmax position: 100.0%
```

**Why check attention-argmax accuracy SEPARATELY from value-prediction
MSE, rather than relying on MSE alone?** A model could, in principle,
achieve low MSE through some OTHER shortcut (e.g., learning a rough
statistical estimate of "the expected maximum of 10 uniform(-1,1) values"
without genuinely locating any specific position) — MSE alone cannot
distinguish "correctly attending to the true max position" from
"outputting a numerically close value via an entirely different
mechanism." Directly checking whether `argmax(attention_weights)` matches
the TRUE max-value position provides mechanistic, not just outcome-level,
verification that the model learned the INTENDED content-based lookup
strategy — and the live 100% match rate confirms it did, precisely and
completely.

---

## 7. Section G — The Complexity Crossover, Empirically Confirmed

### Live Result: The O(L²) vs O(L) Crossover, Visible in Real Numbers

```
Length=16:  Attention faster (ratio 1.29 -- LSTM takes 1.29x longer)
Length=64:  LSTM faster (ratio 0.56 -- LSTM takes only 0.56x as long)
Length=512: LSTM much faster (ratio 0.09 -- LSTM takes only 9% as long)
```

**Why does attention start out FASTER than LSTM at short lengths, then
become dramatically SLOWER at long lengths — exactly crossing over
somewhere between L=32 and L=64?** At short sequence lengths, LSTM's
INHERENTLY SEQUENTIAL computation (each time step must wait for the
previous one — Phase 3 Topic 1 §3) dominates its cost, while attention's
fully-parallel matrix multiplications execute efficiently regardless of
the (still small) `L²` term. As `L` grows, attention's `O(L²)` cost
eventually overtakes LSTM's `O(L)` cost, and by `L=512`, attention takes
roughly **11× longer** than LSTM per forward pass (`24.2ms` vs `2.3ms`) —
a direct, empirical demonstration of theory.md §9's complexity table, not
merely a restatement of the Big-O notation. This crossover point is
CPU-specific and implementation-specific (GPU parallelism, which
efficiently exploits attention's embarrassingly-parallel structure across
BOTH the batch and sequence dimensions, shifts this crossover to much
longer sequences in practice) — but the qualitative trend (attention's
relative cost growing faster than RNN's as sequences lengthen) is the
fundamental, hardware-independent trade-off Phase 3 Topic 4 §1's original
comparison table was built around.

---

## Pitfalls Avoided

| Pitfall | Fix Applied |
|---|---|
| Guessing PyTorch's internal weight-stacking/head-splitting convention | Probed `nn.MultiheadAttention` empirically FIRST, before writing scratch code |
| Softmax overflow for large attention scores | Subtract row-max before `exp`, same as Phase 1 Topic 2's stability trick |
| Testing positional-encoding linearity on the SAME positions used to fit it | Strict train/test split across DIFFERENT position ranges |
| Confusing "low MSE" with "learned the intended mechanism" | Separately verified attention-argmax matches the TRUE max position |
| Letting positional encoding confound a pure content-based-lookup demo | Deliberately omitted PE from the max-finding task |
| Presenting Big-O complexity as if it were empirically self-evident | Actually measured wall-clock time across 6 sequence lengths to show the real crossover |

---

*Next: [Topic 2 — The Transformer Architecture](../02-transformer-architecture/explanation.md)*
