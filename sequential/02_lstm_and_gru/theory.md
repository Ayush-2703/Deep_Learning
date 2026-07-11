# Theory: LSTM & GRU — Gated Recurrent Architectures

**Phase 3 — Topic 2 | Deep Learning Mastery Repository**

---

## 1. Why Gating Solves the Vanishing Gradient Problem

Vanilla RNN update: `hₜ = tanh(Wₕₕh_{t-1} + ...)` — purely multiplicative.
LSTM cell update:  `Cₜ = fₜ ⊙ C_{t-1} + iₜ ⊙ C̃ₜ` — **additive pathway!**

When fₜ≈1 and iₜ≈0: `Cₜ ≈ C_{t-1}` — near-identity, gradient ≈ 1.
This is the "Constant Error Carousel" — gradient flows backward with NO decay.

---

## 2. The LSTM Cell

```
Forget gate:    fₜ = σ(Wf·[h_{t-1}, xₜ] + bf)      "keep old cell state?"
Input gate:     iₜ = σ(Wi·[h_{t-1}, xₜ] + bi)      "write new info?"
Candidate cell: C̃ₜ = tanh(Wc·[h_{t-1}, xₜ] + bc)   "what to write?"
Output gate:    oₜ = σ(Wo·[h_{t-1}, xₜ] + bo)       "expose how much?"

Cell update:    Cₜ = fₜ ⊙ C_{t-1} + iₜ ⊙ C̃ₜ
Hidden output:  hₜ = oₜ ⊙ tanh(Cₜ)
```

Two separate state vectors: **Cₜ** (long-term memory, additive updates) and
**hₜ** (working representation, filtered and exposed to the rest of the network).

Gradient through cell: `∂Cₜ/∂C_{t-1} = fₜ` — when fₜ≈1, no decay across steps.

---

## 3. The GRU Cell (Cho et al. 2014)

```
Reset gate:  rₜ = σ(Wr·[h_{t-1}, xₜ])     "how much past affects candidate?"
Update gate: zₜ = σ(Wz·[h_{t-1}, xₜ])     "how much to interpolate?"
Candidate:   h̃ₜ = tanh(Wh·[rₜ⊙h_{t-1}, xₜ])
Update:      hₜ = (1−zₜ)⊙h_{t-1} + zₜ⊙h̃ₜ
```

GRU merges cell and hidden states into ONE vector. The update equation
IS the additive gradient-preserving pathway — no separate output gate needed.

---

## 4. LSTM vs GRU

```
                LSTM            GRU
Gates:          3               2
State vectors:  2 (C,h)         1 (h)
Parameters:     4× hidden²      3× hidden²  (25% fewer)
```

GRU has 75% of LSTM's parameter count for the same hidden_size.
In practice: LSTM often better on complex long-range tasks; GRU comparable
and faster on many tasks — good default choice.

---

## 5. The Adding Problem (Hochreiter & Schmidhuber 1997)

Classic benchmark: sequence of (value, marker) pairs. Exactly TWO markers=1
(one per half). Target = sum of values at marked positions.

Requires: identify which positions are marked, REMEMBER the first value
across a potentially long gap, produce a precise regression output.
Vanilla RNNs essentially never solve this beyond length~100; LSTMs do reliably.

---

## Key Equations

| Formula | Description |
|---|---|
| Cₜ = fₜ⊙C_{t-1} + iₜ⊙C̃ₜ | LSTM cell update (additive) |
| hₜ = oₜ⊙tanh(Cₜ) | LSTM hidden output |
| hₜ = (1−zₜ)⊙h_{t-1} + zₜ⊙h̃ₜ | GRU hidden update |
| ∂Cₜ/∂C_{t-1} = fₜ | Gradient preserves when fₜ≈1 |

