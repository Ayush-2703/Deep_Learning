# Theory: Attention Mechanisms — Scaled Dot-Product & Multi-Head Attention

**Phase 4 — Topic 1 | Deep Learning Mastery Repository**

---

## Table of Contents
1. [From Bahdanau/Luong Attention to Self-Attention](#1-from-bahdanaulu-attention-to-self-attention)
2. [The Query-Key-Value Framework](#2-the-query-key-value-framework)
3. [Scaled Dot-Product Attention](#3-scaled-dot-product-attention)
4. [Why Scale by √d_k](#4-why-scale-by-dk)
5. [Self-Attention vs Cross-Attention](#5-self-attention-vs-cross-attention)
6. [Multi-Head Attention](#6-multi-head-attention)
7. [Causal (Masked) Attention](#7-causal-masked-attention)
8. [Positional Encoding](#8-positional-encoding)
9. [Computational Complexity](#9-computational-complexity)

---

## 1. From Bahdanau/Luong Attention to Self-Attention

Phase 3 Topic 3 introduced attention as a mechanism letting a Seq2Seq
DECODER look back at ALL encoder hidden states, rather than compressing
everything into one fixed context vector. That attention was fundamentally
CROSS-attention: queries came from the decoder, keys/values came from a
DIFFERENT sequence (the encoder).

The Transformer's key generalization: apply this SAME query-key-value
mechanism WITHIN a single sequence, letting every position attend to every
OTHER position in that same sequence. This is **self-attention** — and,
critically, the Transformer discards recurrence ENTIRELY, relying on
self-attention (plus position information, §8) as the sole mechanism for
modeling sequential relationships.

```
Bahdanau/Luong (Phase 3):    decoder position t  →  attends to  →  ALL encoder positions
                              (cross-sequence attention, still wrapped in an RNN)

Self-Attention (Transformer): EVERY position in a sequence → attends to →  EVERY
                              position in the SAME sequence (no RNN at all)
```

---

## 2. The Query-Key-Value Framework

Attention is best understood through a retrieval-system analogy:

```
Query (Q):  "What am I looking for?"     — a representation of the current position
Key (K):    "What do I contain?"          — a representation each position offers to be matched against
Value (V):  "What do I actually provide?" — the content actually retrieved if matched

Retrieval process:
  1. Compare the Query against every Key (compute a similarity/compatibility score)
  2. Convert scores to a probability distribution (softmax) — "how much to retrieve from each position"
  3. Return a WEIGHTED SUM of Values, weighted by that distribution
```

This is directly analogous to a soft, differentiable dictionary lookup: instead
of retrieving ONE exact-match value (as in a hash table), attention retrieves
a WEIGHTED BLEND of all values, with weights determined by how well each
key matches the query.

---

## 3. Scaled Dot-Product Attention

### The Formula

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V

Q ∈ ℝ^(L_q × d_k)    L_q queries, each dimension d_k
K ∈ ℝ^(L_k × d_k)    L_k keys, each dimension d_k (same d_k as queries — required for the dot product)
V ∈ ℝ^(L_k × d_v)    L_k values, each dimension d_v (can differ from d_k)

QKᵀ ∈ ℝ^(L_q × L_k)           raw compatibility scores (one row per query, one column per key)
softmax(QKᵀ/√d_k) ∈ ℝ^(L_q × L_k)   normalized attention weights (each ROW sums to 1)
Output ∈ ℝ^(L_q × d_v)         weighted combination of values, per query
```

### Step-by-Step

```
1. Score:   S = QKᵀ                     (dot product = a similarity measure between vectors)
2. Scale:   S' = S / √d_k                (see §4 for why)
3. Normalize: A = softmax(S', dim=-1)    (row-wise softmax — each query's weights sum to 1)
4. Aggregate: Output = A V               (weighted sum of values per query)
```

**Why dot product for similarity?** For two vectors, `q·k = ‖q‖‖k‖cos(θ)` — the
dot product is large when vectors point in similar directions (small angle
θ) and both have substantial magnitude. This is a computationally cheap
(single matrix multiply), differentiable similarity measure — ideal for a
mechanism that needs to be both fast (GPU-friendly matmul) and trainable
via backpropagation.

---

## 4. Why Scale by √d_k

### The Variance Argument

Assume `q` and `k`'s components are independent random variables with mean
0 and variance 1 (a reasonable assumption for well-initialized/normalized
network activations). Then their dot product:

```
q·k = Σᵢ₌₁^{d_k} qᵢkᵢ

E[q·k] = 0                       (since qᵢ,kᵢ independent, mean 0)
Var(q·k) = Σᵢ Var(qᵢkᵢ) = d_k · Var(qᵢ)Var(kᵢ) = d_k · 1 · 1 = d_k
```

**The variance of the raw dot product GROWS LINEARLY with the dimension
`d_k`.** For large `d_k` (e.g., 64 or 128, typical in Transformers), this
means dot products can have LARGE magnitude, pushing softmax inputs into
its SATURATING regime (where the largest input dominates almost
completely, and gradients for the other inputs vanish — directly
analogous to the softmax saturation issue from Phase 1 Topic 2).

### The Fix

Dividing by `√d_k` exactly counteracts this:

```
Var(q·k / √d_k) = Var(q·k) / d_k = d_k / d_k = 1

The scaled dot product has variance 1, REGARDLESS of d_k — keeping
softmax's input in a well-behaved range where gradients flow properly,
regardless of how large the head dimension is.
```

---

## 5. Self-Attention vs Cross-Attention

```
Self-Attention:   Q, K, V all derived from the SAME sequence
                  (e.g., every word in a sentence attends to every other
                  word in that SAME sentence)

Cross-Attention:  Q derived from one sequence, K/V derived from a
                  DIFFERENT sequence
                  (e.g., in a translation Transformer's decoder: Q comes
                  from the target/decoder sequence, K/V come from the
                  source/encoder sequence — directly generalizing Phase 3
                  Topic 3's Bahdanau attention into this Q/K/V framework)
```

Both use the EXACT SAME scaled dot-product attention formula (§3) — the
only difference is WHERE the Q, K, V vectors come from.

---

## 6. Multi-Head Attention

### Motivation: One Attention "View" Isn't Enough

A single attention computation produces ONE weighted average per query —
but a sequence position might need to attend to DIFFERENT other positions
for DIFFERENT reasons simultaneously (e.g., a word might need to track both
its grammatical subject AND a coreferent pronoun, via different
relationships). A single softmax distribution can only express ONE
"attention pattern" per position.

### The Solution: Parallel Attention "Heads"

```
1. Linearly project Q,K,V into h DIFFERENT, LOWER-DIMENSIONAL subspaces:
     Qᵢ = QWᵢ^Q,  Kᵢ = KWᵢ^K,  Vᵢ = VWᵢ^V     for i=1,...,h  (h = num heads)
     each Wᵢ^Q ∈ ℝ^(d_model × d_k),  d_k = d_model/h

2. Compute attention INDEPENDENTLY in each of the h subspaces:
     headᵢ = Attention(Qᵢ, Kᵢ, Vᵢ)             ∈ ℝ^(L × d_k)

3. Concatenate all heads' outputs, then project back to d_model:
     MultiHead(Q,K,V) = Concat(head₁,...,head_h) W^O
     W^O ∈ ℝ^(h·d_k × d_model)
```

### Why This Is (Almost) Free Computationally

Splitting `d_model` into `h` heads of dimension `d_k = d_model/h` means the
TOTAL computation across all heads is comparable to a SINGLE full-dimension
attention computation (`h` heads × `O(L²·d_k)` each ≈ `O(L²·d_model)` total,
same order as one full-width attention) — multi-head attention gets
multiple independent "representation subspaces" essentially for free,
compared to single-head attention at the same total dimensionality.

### Implementation Detail: Contiguous Head Splitting

In practice (and confirmed by inspecting PyTorch's `nn.MultiheadAttention`
implementation), rather than maintaining `h` separate weight matrices, a
SINGLE combined projection matrix computes all heads' Q (or K, or V)
simultaneously, and the output is then SLICED into `h` contiguous chunks
along the feature dimension — mathematically equivalent to `h` separate
projections, but more efficient as one larger matrix multiply.

---

## 7. Causal (Masked) Attention

For AUTOREGRESSIVE tasks (predicting the next token given only previous
tokens — e.g., language modeling, Topic 4), a position must NOT be allowed
to attend to FUTURE positions, or the model could "cheat" by looking ahead
at the very token it's supposed to predict.

### The Masking Mechanism

```
Before softmax, set all "forbidden" (future) positions' scores to -∞:

  S'ᵢⱼ = { QKᵀ/√d_k  if j ≤ i   (key position j is at or before query position i)
         { -∞         if j > i   (key position j is in the future — forbidden)

After softmax: exp(-∞) = 0 exactly, so forbidden positions receive EXACTLY
zero attention weight, regardless of the other (allowed) positions' scores.
```

This produces a LOWER-TRIANGULAR attention weight matrix — position `i`'s
attention weights are non-zero only for keys `j ≤ i`.

**Why this matters for training efficiency:** causal masking allows an
autoregressive Transformer to be trained on an ENTIRE sequence in ONE
forward pass (computing all positions' predictions simultaneously, each
correctly restricted to only see its own past) — unlike an RNN, which
must process positions strictly sequentially even during training. This
parallelism is a major reason Transformers train faster than RNNs on the
same hardware (ties back to Phase 3 Topic 4 §1's RNN/Transformer/SSM
comparison table).

---

## 8. Positional Encoding

### The Problem: Attention Is Permutation-Invariant

Self-attention, as defined in §3, treats its input as an UNORDERED SET —
if you permute the input tokens' order, the SET of attention computations
performed is identical (only WHICH position ends up paired with which
weight changes, not the underlying computation) — the mechanism itself has
NO inherent notion of sequence ORDER. This is a fundamental problem for
sequence modeling: "the dog bit the man" and "the man bit the dog" would
be processed identically by pure self-attention without additional
position information.

### Sinusoidal Positional Encoding (Original Transformer)

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

pos: position in the sequence (0, 1, 2, ...)
i:   dimension index pair (0, 1, ..., d_model/2 - 1)
```

This produces a UNIQUE, FIXED vector for each position, which is ADDED
(not concatenated) to the token embedding before the first Transformer
layer — injecting position information directly into the representation
that self-attention then operates on.

### Why Sinusoids Specifically?

```
1. BOUNDED: sin/cos are always in [-1,1], regardless of position — unlike
   e.g. using the raw position INDEX directly, which would grow unboundedly
   for long sequences and dominate the embedding's scale.

2. UNIQUE per position (for reasonable sequence lengths): the combination
   of many different frequencies (varying with i) makes each position's
   full d_model-dimensional encoding vector distinguishable.

3. RELATIVE POSITION IS A LINEAR FUNCTION: for any fixed offset k, PE(pos+k)
   can be expressed as a LINEAR FUNCTION of PE(pos) (via trigonometric
   angle-addition identities) — this property makes it easier for the
   model to learn to attend to RELATIVE positions (e.g., "the token 3
   positions back"), since that relationship has a consistent, learnable
   linear structure regardless of the ABSOLUTE position involved.
```

### Alternative: Learned Positional Embeddings

Many modern models (BERT, GPT — Topics 3-4) instead use a LEARNED embedding
table indexed by position (exactly like a token embedding table, but
indexed by position instead of token identity) — simpler to implement,
empirically comparable performance to sinusoidal encoding for the
sequence lengths seen during training, but does NOT generalize to
sequence lengths longer than those seen in training (sinusoidal encoding,
being a fixed mathematical function, can in principle extrapolate to
arbitrary lengths).

---

## 9. Computational Complexity

```
Self-attention:  O(L² · d_model)   time and memory — QUADRATIC in sequence length L
                 (every position attends to every other position)

RNN/LSTM/GRU:     O(L · d_model²)   time, O(d_model) memory — LINEAR in L
                 (Phase 3 Topics 1-2)

SSM (S4/Mamba):   O(L · d_model)    time, O(d_model) memory — LINEAR in L,
                 with the same L·d cost structure as RNNs but with
                 PARALLELIZABLE training (Phase 3 Topic 4)
```

**The fundamental trade-off:** self-attention's O(L²) cost is expensive for
very long sequences, but buys FULL, DIRECT, UNATTENUATED access from any
position to any other position in a SINGLE layer (no vanishing-gradient-style
decay across distance, unlike RNNs) — plus full parallelism across the
sequence dimension during training (unlike RNNs' inherently sequential
computation). This is precisely the trade-off table introduced in Phase 3
Topic 4 §10, now grounded in the actual mechanism producing each row of
that table.

---

## Key Equations Summary

| Concept | Formula |
|---|---|
| Scaled dot-product attention | Attention(Q,K,V) = softmax(QKᵀ/√d_k)V |
| Variance of raw dot product | Var(q·k) = d_k |
| Multi-head attention | Concat(head₁,...,head_h)W^O |
| Per-head attention | headᵢ = Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V) |
| Causal mask | S'ᵢⱼ = -∞ if j>i else QKᵀ/√d_k |
| Sinusoidal PE (even dims) | sin(pos/10000^(2i/d_model)) |
| Sinusoidal PE (odd dims) | cos(pos/10000^(2i/d_model)) |
| Self-attention complexity | O(L²·d_model) |

