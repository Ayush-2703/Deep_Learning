# LoRA, RAG, and RLHF

Three widely-used techniques for adapting and controlling pretrained
language models, each solving a different problem. This topic covers all
three, each with its own working, verified implementation.

---

## PART A: LoRA (Low-Rank Adaptation)

### 1. The problem: full fine-tuning is expensive

Fine-tuning every parameter of a large pretrained model for each new task
requires storing a full copy of the model per task and updating (and
computing gradients for) potentially billions of parameters — expensive in
both compute and storage, especially when you need many task-specific
variants of the same base model.

### 2. The LoRA idea: freeze the base, learn a low-rank update

Hu et al. (2021) observed that the *change* in weights needed to adapt a
pretrained model to a new task tends to have low "intrinsic rank" — it can
be well-approximated by a low-rank decomposition. Instead of updating a
weight matrix `W in R^{d x k}` directly, LoRA freezes `W` entirely and adds
a learned low-rank update:

```
W' = W + (alpha / r) * B @ A
```

- `A in R^{r x k}`, `B in R^{d x r}`, with rank `r << min(d, k)` (e.g. r=4
  or r=8, vs. `d,k` potentially in the thousands)
- `A` is initialized randomly (small values), `B` is initialized to
  **zero** — so at the start of training, `B @ A = 0` and the adapted
  model is numerically identical to the frozen base model. Training then
  gradually moves `B @ A` away from zero.
- `alpha` is a scaling constant (often set so `alpha/r` is a fixed
  hyperparameter independent of `r`, making it easier to compare different
  rank choices).

### 3. Why this is dramatically cheaper

The number of trainable parameters for one LoRA-adapted layer is
`r * (d + k)` instead of `d * k` for full fine-tuning. For `d=k=256`,
`r=4`: `4 * 512 = 2048` trainable parameters vs. `256 * 256 = 65536` for
full fine-tuning — a **32x reduction** for this layer alone. The frozen
base weights `W` are shared across every task-specific LoRA adapter, so
you can store one full base model plus many small `(A, B)` pairs (often a
few MB each) rather than many full model copies.

### 4. What this implementation verifies

A small Transformer (GPT-style, reusing the architecture pattern from
Topic 3) is pretrained on the synthetic grammar's "Domain A" distribution,
then adapted to a shifted "Domain B" distribution two ways: (1) full
fine-tuning of every parameter, and (2) LoRA adaptation of only the
attention projection matrices. Both are compared on: trainable parameter
count, Domain B validation performance, and — critically — whether
frozen-base LoRA training actually preserves Domain A performance better
than full fine-tuning (a real, checkable claim, not just a parameter-count
argument).

---

## PART B: RAG (Retrieval-Augmented Generation)

### 1. The problem: parametric knowledge is frozen and limited

A trained language model's "knowledge" is baked into its weights at
training time. It can't be updated without retraining, can't cite sources,
and will confidently generate plausible-sounding but wrong answers about
facts it never saw enough times to memorize (or that changed after
training).

### 2. The RAG idea: look it up instead of memorizing it

Retrieval-Augmented Generation splits the problem into two components:

```
query -> [RETRIEVER: find relevant documents from a knowledge base]
       -> [GENERATOR: produce an answer conditioned on query + retrieved documents]
```

The retriever is typically a similarity search over document embeddings
(dense, via a trained encoder) or sparse term-based similarity (e.g.
TF-IDF/BM25). The generator then has the actual relevant text available in
its context window, rather than needing to have memorized it during
training — this also makes answers traceable to a specific source document.

### 3. What this implementation covers

A synthetic knowledge base of short factual sentences (template-generated,
e.g. "The [entity] is located in [place]"), a TF-IDF-based sparse
retriever (implemented from scratch with numpy — no external search
library), and a template-based extractive answerer that pulls the
requested fact from the top-retrieved document. This is compared against
a **no-retrieval baseline** that must answer purely from a small
parametric classifier trained to memorize entity->fact mappings — directly
demonstrating the case for retrieval: the parametric baseline's accuracy
degrades on facts it saw rarely during training, while the retrieval-based
approach's accuracy is essentially independent of training frequency,
since it looks the fact up at inference time rather than recalling it from
weights.

---

## PART C: RLHF (Reinforcement Learning from Human Feedback), simplified

### 1. The problem: likelihood training doesn't directly optimize what we want

A language model trained purely to predict the next token (Topic 3) learns
to model the training distribution, not to produce outputs that are
"helpful," "safe," or otherwise preferred by humans along some axis not
directly captured by next-token likelihood.

### 2. The three-stage RLHF pipeline

1. **Supervised pretraining** (already covered — this is the GPT-style
   model from Topic 3).
2. **Reward model training**: collect pairs of model outputs with human
   (or, here, synthetic rule-based) preference labels — "output A is
   preferred over output B" — and train a reward model `R_phi(x)` to
   predict a scalar score consistent with these pairwise preferences,
   typically via the Bradley-Terry loss:
   ```
   L(phi) = -E[ log( sigmoid( R_phi(preferred) - R_phi(rejected) ) ) ]
   ```
3. **Policy optimization**: fine-tune the language model (the "policy")
   using reinforcement learning to maximize the reward model's score on
   its own generated outputs, typically with a KL-divergence penalty
   against the original pretrained policy to prevent it from drifting too
   far and "reward hacking" (finding degenerate outputs that fool the
   reward model without being genuinely better):
   ```
   objective = E[ R_phi(x) ] - beta * KL( pi_theta || pi_theta_original )
   ```

### 3. Policy gradient estimation (REINFORCE, simplified from full PPO)

Real RLHF systems typically use PPO (Proximal Policy Optimization) for
stability. This implementation uses plain REINFORCE with a moving-average
baseline (a simpler, older policy-gradient method) to keep the
implementation compact and inspectable, while still exercising the same
core mechanism: sample an action (token sequence) from the current policy,
score it with the reward model, and push up the log-probability of
higher-than-baseline-reward sequences:

```
grad = E[ (R_phi(x) - baseline) * grad(log pi_theta(x)) ]
```

### 4. What this implementation verifies

A synthetic, ground-truth reward function is defined (count of a specific
"preferred" token in the generated sequence — simple and directly
checkable). A reward *model* is trained on pairwise preferences derived
from this ground truth (not given the ground truth function directly,
mimicking the real RLHF setup where the true "human preference function"
is never directly accessible to the policy-optimization stage). The policy
is then optimized against the *learned* reward model, and success is
measured against the **ground-truth** reward function on freshly-generated
sequences — checking whether RL fine-tuning actually shifted the policy in
the intended direction, not just whether the learned reward model's score
went up (which could indicate reward-model exploitation rather than
genuine improvement).

### 5. Known failure modes to watch for

- **Reward hacking**: the policy finds a degenerate way to maximize the
  reward model's score without actually improving on the true objective
  (e.g. repeating the preferred token exploiting a reward model blind
  spot rather than distributing it naturally within grammatical
  sequences).
- **Mode collapse**: the policy converges to a small number of
  high-reward outputs, losing diversity — worth checking output diversity
  before and after RL fine-tuning, not just average reward.
- **KL penalty too weak**: policy drifts far from the original
  distribution, potentially degrading fluency/grammaticality even as the
  reward score increases.

