# Theory: Recurrent Neural Networks (RNNs)

**Phase 3 — Topic 1 | Deep Learning Mastery Repository**

---

## Table of Contents
1. [Why Sequences Need a Different Architecture](#1-why-sequences-need-a-different-architecture)
2. [The Vanilla RNN Cell](#2-the-vanilla-rnn-cell)
3. [Unrolling Through Time](#3-unrolling-through-time)
4. [Backpropagation Through Time (BPTT)](#4-backpropagation-through-time-bptt)
5. [The Vanishing/Exploding Gradient Problem](#5-the-vanishingexploding-gradient-problem-in-rnns)
6. [RNN Architectural Variants](#6-rnn-architectural-variants)
7. [Bidirectional RNNs](#7-bidirectional-rnns)
8. [Truncated BPTT](#8-truncated-bptt)

---

## 1. Why Sequences Need a Different Architecture

Feedforward networks (Phase 1) and CNNs (Phase 2) assume FIXED-size input.
Sequential data has THREE properties that break this:

1. **Variable length**: sentences can be 3 or 300 words
2. **Order matters**: "dog bites man" ≠ "man bites dog"
3. **Long-range dependencies**: resolving "it" may require context from 30 words back

An RNN processes ONE ELEMENT AT A TIME, maintaining a **hidden state** vector
updated at each step — a learned, compressed memory. The SAME cell (weights)
is reused at every step, analogous to a CNN's spatial parameter sharing.

---

## 2. The Vanilla RNN Cell

```
hₜ = tanh(Wₕₕ h_{t-1} + Wₓₕ xₜ + bₕ)
yₜ = Wₕᵥ hₜ + b_y

Wₕₕ ∈ ℝ^(d×d)  hidden-to-hidden (recurrent) weight matrix
Wₓₕ ∈ ℝ^(d×k)  input-to-hidden weight matrix
```

**Same matrices at EVERY time step** — this is what makes it "recurrent."

---

## 3. Unrolling Through Time

An RNN processing length-T sequence = depth-T feedforward net with TIED weights.
This enables standard backpropagation (BPTT).

---

## 4. Backpropagation Through Time (BPTT)

```
∂L/∂Wₕₕ = Σₜ ∂Lₜ/∂Wₕₕ   (sum contributions from every time step)

∂hₜ/∂hₖ = ∏ᵢ₌ₖ₊₁ᵗ diag(tanh'(zᵢ)) · Wₕₕᵀ   (product of Jacobians)
```

The product of many matrices is the root cause of vanishing/exploding gradients.

---

## 5. The Vanishing/Exploding Gradient Problem in RNNs

```
‖∂hₜ/∂hₖ‖ ≈ ‖Wₕₕ‖^(t-k) · ‖diag(tanh')‖^(t-k)

If ‖Wₕₕ‖·max(tanh') < 1 → gradient VANISHES exponentially
If ‖Wₕₕ‖·max(tanh') > 1 → gradient EXPLODES exponentially
```

A vanilla RNN reliably learns dependencies spanning ~5-10 steps in practice.
This motivates gating mechanisms (Topic 2: LSTM, GRU).

**Partial mitigations:** gradient clipping (for exploding), orthogonal init,
ReLU activation, truncated BPTT.

---

## 6. RNN Architectural Variants

```
ONE-TO-MANY:     image → caption
MANY-TO-ONE:     sentence → sentiment label
MANY-TO-MANY (synced):      per-step output (POS tagging)
MANY-TO-MANY (encoder-decoder): machine translation (Topic 3)
```

---

## 7. Bidirectional RNNs

```
Forward RNN:   →h₁ →h₂ ... →hₜ   (sees past context)
Backward RNN:  ←h₁ ←h₂ ... ←hₜ   (sees future context, separate weights)
Output:        hₜ = [→hₜ ; ←hₜ]   (concatenated)
```

Cannot be used for streaming/autoregressive generation (future tokens unavailable).

---

## 8. Truncated BPTT

For very long sequences: split into chunks, carry `h.detach()` between chunks.
```python
h = h.detach()  # cuts gradient graph; value still flows forward
```
Trades some gradient accuracy for computational tractability.

---

## Key Equations Summary

| Concept | Formula |
|---|---|
| RNN cell | hₜ = tanh(Wₕₕh_{t-1} + Wₓₕxₜ + bₕ) |
| BPTT gradient | ∂L/∂Wₕₕ = Σₜ ∂Lₜ/∂Wₕₕ |
| Jacobian product | ∂hₜ/∂hₖ = ∏ᵢ diag(tanh'(zᵢ))·Wₕₕᵀ |
| Bidirectional | hₜ = [→hₜ ; ←hₜ] |
