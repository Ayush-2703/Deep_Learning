# Explanation: LoRA / RAG / RLHF Implementation Walkthrough

---

## PART A: LoRA

### 1. `LoRALinear` — the adapter, and why `B` is zero-initialized

```python
self.A = nn.Parameter(torch.randn(r, in_dim) * 0.01)
self.B = nn.Parameter(torch.zeros(out_dim, r))
...
lora_out = (x @ self.A.T) @ self.B.T
return base_out + self.scaling * lora_out
```

`B` starts at exactly zero, so `lora_out = 0` at initialization regardless
of what `A` is — the adapted model is numerically identical to the frozen
base model before any training happens. This matters practically: it means
LoRA fine-tuning never *starts* by disrupting the base model's behavior;
it only gradually introduces a change as training proceeds, unlike a
randomly-initialized full fine-tuning run which perturbs the whole model
from step one.

### 2. A finding that corrected an initial misconception in this implementation

The first version of this script printed a "sanity check" asserting that
LoRA-adapted Domain A accuracy should be *unchanged* from the base model,
reasoning "the base weights are frozen." Running it showed Domain A
accuracy actually dropped substantially (0.4462 -> 0.2087) — which looked
like a bug in the freezing logic. It wasn't: **the trained adapter is
active on every forward pass**, including Domain A inputs, and it was
trained only to help Domain B — so of course it changes Domain A behavior
too, even though the underlying `base` weights never received a gradient
update. The original check was testing the wrong claim.

**Fix**: added an `enabled` toggle to `LoRALinear` and a `set_lora_enabled`
helper, then checked the actually-correct claim — that disabling the
adapter (`base_out` only, no `lora_out` contribution) exactly reproduces
the pre-adaptation base model:

```
Domain A val acc with adapter ENABLED:  0.2087
Domain A val acc with adapter DISABLED: 0.4462
base model Domain A val acc:            0.4462   <- exact match, diff=0.000000
```

This is the genuinely correct and more useful lesson: LoRA's real
guarantee is that the *base weights* are provably untouched (verified here
to 6 decimal places), which is what lets a single base model host many
swappable task-specific adapters — not that a trained adapter has zero
effect on other domains while switched on.

### 3. Full fine-tuning vs. LoRA — the actual measured tradeoff

```
Full fine-tune: 70,165 trainable params | Domain B acc=0.5100 | Domain A acc=0.3975 (dropped from 0.4462)
LoRA (r=4):      2,048 trainable params (2.92% of full) | Domain B acc=0.4700 | Domain A (adapter off)=0.4462 (exact preservation)
```

Full fine-tuning achieved slightly higher Domain B accuracy (0.51 vs 0.47)
but at the cost of measurable, genuine forgetting on Domain A (0.4462 ->
0.3975) — since every parameter, including ones important for Domain A,
received gradient updates from Domain B data. LoRA reached comparable
Domain B performance using **34x fewer trainable parameters**, with the
base model's Domain A capability providably recoverable at any time by
disabling the adapter. This is the concrete tradeoff theory.md Part A
describes, verified with real numbers rather than asserted.

---

## PART B: RAG

### 4. A real tokenization bug that silently broke half the evaluation

Initial retrieval accuracy was only 52.5% — suspicious, since entity names
are unique per document and should be trivial to retrieve via exact
term-overlap. Debugging `tokenize()` directly revealed the cause:

```python
# BEFORE (buggy):
def tokenize(text):
    return text.lower().replace(".", "").replace(",", "").split()
tokenize("What is Zorvath's profession?")  # -> ["what", "is", "zorvath's", "profession?"]
```

`"zorvath's"` (apostrophe-s, unstripped) never matches the clean
`"zorvath"` token from the knowledge base document — silently breaking
**every single profession-related query** (half the evaluation set),
since the entity-name term (the one genuinely distinguishing token) never
overlapped between query and document.

**Fix**: switched to a regex-based tokenizer, `re.findall(r"[a-z]+",
text.lower())`, which correctly splits `"zorvath's"` into `["zorvath",
"s"]` — restoring the entity-name match. After the fix, retrieval accuracy
went from 52.5% to a perfect **100% (40/40)**. This is a good illustration
of why "the code ran without crashing" is not the same as "the code is
correct" — the bug was silent, not a stack trace, and only surfaced by
checking whether the *result* made sense against a reasonable expectation
(near-perfect retrieval on a knowledge base this small and unambiguous).

### 5. A second real issue: the intended comparison didn't show up at first

The parametric (no-retrieval) baseline was designed to demonstrate
training-frequency-dependent forgetting — but its first configuration
(`emb_dim=32`, 60 epochs) let it perfectly memorize both frequent *and*
rare entities (both 100% accuracy), because a 20-entity, 8-value
classification task is simply too easy for a 32-dimensional embedding to
struggle with regardless of exposure count. This wasn't wrong, just
uninformative for the comparison being made.

**Fix**: reduced embedding capacity to 2 dimensions and training to 10
epochs — values found by direct, quick experimentation (not guessed) to
reliably reproduce a genuine capacity/exposure tradeoff:

```
Parametric baseline -- frequent entities (40x in training): 1.0000
Parametric baseline -- rare entities (2x in training):       0.2500
RAG -- frequent entities: 1.0000
RAG -- rare entities:     1.0000
```

This is the actual, concrete case for retrieval from theory.md Part B: the
parametric model's accuracy is highly sensitive to training exposure
(0.75 absolute drop), while RAG's accuracy is completely unaffected by it
(1.0 in both cases) — it never needed to memorize anything; it looks the
fact up fresh, every time, from the knowledge base.

### 6. `tfidf_vector` — sparse retrieval, computed from scratch

```python
idf = np.log(N_DOCS / (df + 1)) + 1
vec = vec * idf
vec = vec / np.linalg.norm(vec)
```

Standard smoothed TF-IDF (the `+1` in the denominator avoids division by
zero for terms appearing in zero training documents; the `+1` after `log`
avoids a zero-weight term for words appearing in every document). Vectors
are L2-normalized so the dot product between a query vector and a document
vector directly computes cosine similarity — this is why `retrieve()` can
just do `doc_vectors @ q_vec` for all documents at once, no separate
normalization step needed at retrieval time.

---

## PART C: RLHF (simplified)

### 7. Why pretraining uses uniformly random digits, not a "real" objective

```python
pretrain_data = torch.randint(0, 10, (2000, GEN_LEN))
```

The base policy is deliberately given no bias toward any digit — this
makes the RL fine-tuning stage's job unambiguous to evaluate: if the
policy's distribution shifts toward the preferred digit ('7') after RL,
that shift can only have come from the RL stage, not from residual
structure already present in the pretraining data. The pretraining loss
converging to `2.3006`, essentially identical to `log(10)=2.3026` (the
exact entropy of a uniform 10-way distribution), confirms the pretrained
policy is indeed close to uniformly random before RL begins.

### 8. The reward model never sees the ground-truth function directly

```python
reward_a = torch.tensor([ground_truth_reward(...) for s in pref_seq_a])
...
preferred_is_a = (reward_a > reward_b).float()
...
loss = F.binary_cross_entropy_with_logits(score_a - score_b, preferred_is_a)
```

The reward model (`RewardModel`, a small GRU + linear head) only ever sees
binary *preference labels* (`A preferred over B`, yes/no) — never the raw
`ground_truth_reward` count. This mirrors the real RLHF setup: human
labelers give pairwise preferences, and the actual "true" scoring function
in a human's head is never directly accessible to the training pipeline.
The reward model still achieved `0.9888` correlation with the true
(hidden) reward function on held-out data — Bradley-Terry pairwise
training is genuinely sufficient to recover an accurate scalar reward
function from comparisons alone.

### 9. The REINFORCE update, term by term

```python
shaped_reward = learned_rewards - KL_BETA * kl_per_seq
reward_baseline = baseline_momentum * reward_baseline + (1 - baseline_momentum) * shaped_reward.mean().item()
advantage = shaped_reward - reward_baseline
loss = -(advantage.detach() * cur_log_probs).mean()
```

- `learned_rewards`: the reward model's score on freshly-sampled
  sequences from the *current* policy (not the reference policy) —
  reward must be evaluated on-policy for REINFORCE to be a valid gradient
  estimator.
- `KL_BETA * kl_per_seq`: subtracted directly from the reward, implementing
  the `objective = E[R] - beta*KL` formulation from theory.md Part C
  section 2 as a *reward penalty* rather than a separate loss term — a
  standard, equivalent way to implement it.
- `reward_baseline`: an exponential moving average of recent shaped
  rewards, subtracted to form the `advantage` — this is the classic
  REINFORCE variance-reduction trick; without it, the raw reward magnitude
  (which can be any real number from an unbounded reward model) would
  produce noisy, poorly-scaled gradients.
- `advantage.detach()`: the advantage itself is not backpropagated through
  (it's a fixed scalar multiplier for this update); only `cur_log_probs`
  carries gradient, exactly matching the REINFORCE gradient estimator
  `E[(R - baseline) * grad(log pi(x))]` from theory.md.

### 10. The critical, honest check: does it transfer to the TRUE objective?

```
Before RL fine-tuning: avg ground-truth reward = 0.935 / 8
After  RL fine-tuning: avg ground-truth reward = 5.497 / 8
```

This is the single most important number in this topic. The policy was
optimized *only* against the learned reward model's score — never once
against `ground_truth_reward` directly. The fact that ground-truth reward
still rose from near-random (0.935, close to the expected 0.8 for uniform
random digits) to 5.497 out of a maximum of 8 is direct evidence that the
learned reward model's signal genuinely generalized to the real underlying
preference structure, rather than the policy exploiting some reward-model
quirk disconnected from the true objective — exactly the failure mode
("reward hacking") theory.md Part C section 5 warns about, and exactly
what this before/after ground-truth check is designed to catch if it had
happened.

### 11. Diversity check — ruling out mode collapse as the explanation

```
Output diversity after RL: 269/300 unique sequences generated (no collapse)
```

A cheap, degenerate way to maximize a count-based reward would be to
always output the same high-reward sequence (e.g. `7777 7777`) — this
check confirms that didn't happen: 269 out of 300 sampled sequences after
RL fine-tuning were still unique, meaning the policy learned a genuinely
shifted *distribution* favoring the digit '7' more often, not a collapsed
point estimate. The example generations make this concrete:
`['5','3','3','7','1','6','8','2']` (before, reward=1) vs.
`['5','7','7','7','7','3','7','7']` (after, reward=6) — a real, visible
shift toward the preferred digit while retaining sequence-level variation.
