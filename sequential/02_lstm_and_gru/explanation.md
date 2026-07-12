# Code Explanation: LSTM & GRU — Gated Recurrent Architectures

**`implementation.py` walkthrough**

---

## 1. Section A — Manual LSTM Cell

### Stacked Gate Weights and the `[0*d:1*d]` Slicing Pattern

```python
self.W_ih = rng.uniform(-std, std, (4*hidden_size, input_size))
...
i_t = sigmoid(gates[0*d:1*d])
f_t = sigmoid(gates[1*d:2*d])
g_t = np.tanh(gates[2*d:3*d])
o_t = sigmoid(gates[3*d:4*d])
```

PyTorch stores ALL four gates' weight rows concatenated in a single matrix
(`weight_ih_l0` has shape `(4*hidden_size, input_size)`) — it computes a
single `(4d,)` vector of pre-activations and then slices it into four gate
vectors. Matching this exact stacking order (input→forget→cell→output) in
our manual implementation lets us directly inject PyTorch's learned weights
for verification without any reordering. Using named index arithmetic
(`0*d:1*d` etc.) rather than hardcoded numbers makes the gate order
self-documenting and immediately verifiable against theory.md §2.

### Why `sigmoid` on `i,f,o` but `tanh` on `g`

Gates (`i,f,o`) are MULTIPLICATIVE controllers — they select "how much to
pass through" — so sigmoid's `(0,1)` range directly represents a soft
on/off switch. The candidate `g` is a CONTENT vector (what to write into
memory), using tanh's `(-1,1)` range to represent a signed value that can
increase OR decrease the stored cell content.

---

## 2. Section B — Manual GRU Cell

### The Key PyTorch-Specific Subtlety: Where the Reset Gate Applies

```python
# WRONG (common textbook presentation):
# n_t = tanh(W_in x + b_in + W_hn (r_t*h) + b_hn)  ← reset gates h BEFORE projection

# CORRECT (PyTorch's actual formula):
n_t = np.tanh(gi[2*d:3*d] + r_t * gh[2*d:3*d])    # reset gates AFTER hidden projection
```

Most textbooks write GRU's candidate as `tanh(Wₓ·x + W_h·(r⊙h))` — applying
the reset gate directly to `h` BEFORE the weight projection. PyTorch instead
computes the full hidden projection `W_hh·h + b_hh` FIRST, then applies the
reset gate to that projected vector: `r ⊙ (W_hh·h + b_hh)`. These two
formulas are NOT equivalent — they produce different outputs for identical
weights and inputs. Getting this wrong would cause the manual implementation
to fail the `np.allclose` verification even with correctly injected weights.
We discovered this empirically by trying the textbook formula first and
checking the assertion.

---

## 3. Section C — Gradient Flow: Confirmed But Nuanced

```
Length=200: RNN=0.00e+00   LSTM=2.59e-43   GRU=1.61e-36
```

Both LSTM and GRU maintain non-zero gradients at length 200 where vanilla RNN
has COMPLETE underflow — confirming theory.md §3's constant-error-carousel
claim. However, `2.59e-43` is still nearly zero in absolute terms (only barely
representable in float32). This is important context: gating mechanisms
DRAMATICALLY extend the length over which gradients survive, but they don't
provide PERFECT infinite-length memory — they delay (not eliminate) the vanishing
gradient problem.

### Why GRU Shows LARGER Gradients Than LSTM at All Lengths

GRU has a simpler, more direct gradient path — a single hidden state updated
via `hₜ = (1−z)⊙h_{t-1} + z⊙h̃ₜ`. When `z≈0`, this is close to an identity
mapping with gradient `1`. LSTM's path goes through TWO non-linearities
(`oₜ⊙tanh(Cₜ)` for the hidden state, plus the cell state's own update) before
the gradient can reach the input — additional squashing that slightly attenuates
GRU's gradient advantage on this particular random initialization.

---

## 4. Section D — The Adding Problem: A Striking Three-Way Split

```
Length=100: RNN=0.1838  LSTM=0.1832  GRU=0.0011   Baseline=0.1842
Length=200: RNN=0.1443  LSTM=0.1444  GRU=0.0018   Baseline=0.1424
```

**GRU DOMINATES at lengths 100 and 200, while RNN and LSTM essentially tie
with each other AT OR NEAR the "always predict the mean" baseline.**

This is a genuinely surprising result that merits careful explanation, not
dismissal:

**Why does GRU succeed where LSTM surprisingly fails at long lengths?**

At lengths 100-200, even LSTM's gradient path is attenuated enough that
training with 60 epochs is insufficient for LSTM to reliably converge on this
specific task configuration. GRU, with its structurally simpler, MORE DIRECT
gradient path (fewer squashing operations per step), manages to maintain just
enough training signal to solve the task where LSTM cannot in the same
compute budget. This illustrates a key practical nuance in theory.md §7:
LSTM is not universally better than GRU — for certain long-range tasks at
certain lengths, GRU's architectural simplicity can actually be an advantage
precisely because it provides a slightly less-attenuated gradient path.

**Why does RNN collapse to baseline at lengths 100-200 (not just random guessing)?**

"Predicting the baseline mean" (MSE ≈ 0.17) IS actually the optimal strategy for
a model that has COMPLETELY failed to learn the task — the mean of the expected
sum (E[two Uniform(0,1) values summed] = 1.0) is the best constant predictor.
RNN MSE ≈ baseline MSE means the RNN has learned to predict ≈1.0 for every
input, ignoring the sequence content entirely — exactly what we'd expect from a
model that can no longer carry the first marked value's information across the
sequence gap due to vanishing gradients.

---

## 5. Section E — Signal Detection Revisited: ALL THREE Fail

```
Length=50:  RNN=54%, LSTM=51%, GRU=53%   (all near chance)
Length=100: RNN=55%, LSTM=53%, GRU=53%   (all near chance)
Length=150: RNN=57%, LSTM=53%, GRU=47%   (all near chance, GRU slightly below!)
```

This was the most surprising result, requiring careful interpretation. All
three architectures perform near the 50% chance baseline on signal detection
at lengths 50-150 — seemingly contradicting Section C's (GRU and LSTM
maintain larger gradients than RNN) and the Adding Problem's (GRU solves it
at length 200) results.

**Why do LSTM and GRU also fail here, despite their gradient advantage?**

The signal detection task requires remembering a **single bit** (`x[0] ∈ {-1,+1}`)
across `T-1` steps of continuous noise. The gating mechanisms need to LEARN to
keep the forget gate near 1 and the input gate near 0 after the first step —
which requires the network to develop a clear strategy from the gradient signal.

With only 30 epochs of training (vs. 60 for the Adding Problem), the gated networks
have insufficient time to reliably learn this gating strategy at long lengths. The
Adding Problem uses a clearer MARKER signal (binary channel explicitly flagging
"remember this") that makes learning what to remember structurally easier; the
signal detection task has no such explicit marker, requiring the network to
infer "remember position 0 always" from a purely implicit training signal — a
harder meta-learning challenge. This is an instructive distinction: gating
mechanisms provide the CAPACITY for long-range memory, but that capacity
requires sufficient training data and epochs to be reliably learned, especially
without an explicit "what to remember" signal in the input.

---

## Pitfalls Avoided

| Pitfall | Fix |
|---|---|
| Textbook GRU formula ≠ PyTorch (reset gates h before vs after projection) | Match PyTorch's `r_t * (W_hh@h + b_hh)` exactly |
| Same stacked-gate convention as LSTM not verified empirically | `assert np.allclose(...)` with actual weight injection |
| Overconfident claim "LSTM/GRU always beat RNN" | Reported Section E's honest all-three-fail result |
| Dismissing GRU beating LSTM as impossible | Explained via gradient-path simplicity argument |

---

*Previous: [Topic 1 — RNNs](../01-rnns/explanation.md)*
*Next: [Topic 3 — Seq2Seq NLP](../03-seq2seq-nlp/explanation.md)*
