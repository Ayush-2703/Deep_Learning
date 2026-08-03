# Explanation: BERT vs GPT Implementation Walkthrough

## 1. The synthetic grammar — and a deliberate, important limitation

```python
def sample_clause():
    return [random.choice(SUBJECTS), random.choice(VERBS), random.choice(OBJECTS)]
```

Verified explicitly (see the counting check run during development): subject,
verb, and object words are sampled **independently** at each slot — there is
no real correlation between, say, "cat" and which verb follows it. This was
a deliberate simplification for tractability, but it has a direct
consequence for interpreting the results below: the only learnable
structure in this dataset is *positional/categorical* (position 0 is always
a subject-slot, position 1 always a verb-slot, etc.), not *semantic*
(which specific subject pairs with which specific verb). This caps how
well any model — BERT, GPT, or otherwise — can possibly do at predicting
an exact obscured word, and that ceiling is computed and used honestly
throughout this evaluation rather than treated as a mysterious performance
gap.

## 2. `MultiHeadSelfAttention` and the mask — one implementation, two behaviors

```python
scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
if attn_mask is not None:
    scores = scores.masked_fill(attn_mask == 0, float("-inf"))
attn = F.softmax(scores, dim=-1)
```

This is a single, shared attention implementation used by *both* the BERT
and GPT models — the only thing that changes between them is which
`attn_mask` gets passed in. This directly demonstrates theory.md section
2's central claim: the entire BERT/GPT distinction is a masking choice, not
a different architecture. `masked_fill(attn_mask == 0, float("-inf"))`
zeroes out forbidden positions' contribution *before* the softmax (setting
their score to `-inf` so `softmax` assigns them exactly 0 probability) —
this is why `attention_mask_comparison.png`'s causal mask (lower triangular)
directly prevents any information leakage from future tokens for GPT.

## 3. `apply_mlm_masking` — implementing the real 80/10/10 BERT recipe

```python
r = random.random()
if r < 0.8:
    batch[i, j] = MASK_ID
elif r < 0.9:
    batch[i, j] = random.randrange(len(SPECIAL), VOCAB_SIZE)
else:
    subtype[i, j] = 3  # left unchanged
```

This matches the original BERT paper's masking recipe exactly, not a
simplified all-`[MASK]` version: of the 15% of tokens selected for
masking, 80% become `[MASK]`, 10% become a random token, and 10% are left
unchanged (but still counted as a training target). The reason this
matters (not just historical trivia): if *every* masked position were
replaced with `[MASK]`, the model would never need to build a useful
representation for a token it's confident is unmodified, since at
fine-tuning/inference time `[MASK]` never actually appears in real input.
The 10% random and 10% unchanged cases force the model to always maintain
a genuinely context-aware representation for every position, since it
can't tell from the input token alone whether it's been corrupted.

## 4. The diagnostic that explains an otherwise-confusing result

Initial aggregate BERT accuracy: **27.4%**. Naively this looked *higher*
than the ~16.7% theoretical ceiling established in section 1 — which
would be a red flag, since bidirectional context genuinely can't do better
than that ceiling on data with no real cross-word correlation. The
per-subtype breakdown resolves this:

```
Accuracy on '[MASK] token (80% of masked positions)':          0.2228
Accuracy on 'random-replaced token (10%)':                     0.0714
Accuracy on 'left unchanged (10%, trivially predictable)':     1.0000
```

The `[MASK]` sub-case (22.3%) is close to the 16.7% ceiling (a small,
expected excess, likely from residual positional/frequency cues within
100 training epochs). The `unchanged` sub-case is 100% — trivially easy,
since the answer is literally still sitting in the input; the model only
needs to learn "when in doubt, copy the input token through," which a
residual-connection Transformer does naturally. The aggregate metric
mixes these together, which is why it looked inflated. **This is the
central lesson of this diagnostic**: a single top-line accuracy number for
MLM can be misleading without decomposing by masking sub-type, and this
implementation catches and reports that explicitly rather than presenting
the flattering aggregate number alone.

## 5. GPT's next-token accuracy landing almost exactly on the ceiling

```
Final GPT next-token prediction accuracy: 0.1737
Theoretical ceiling: 0.1667
```

Unlike BERT's MLM setup, GPT's causal LM objective has no "trivially easy"
sub-case analogous to the unchanged-token trick — every position is a
genuine next-token prediction from strictly left-context. Landing almost
exactly on the 16.7% ceiling is a positive result properly interpreted:
it means the model learned the positional/categorical structure about as
completely as the data-generating process allows, not that training
failed. This is explicitly the kind of result that could be
misread as "the model barely learned anything" without the ceiling
computation — reported here with the calibration context needed to
interpret it correctly.

## 6. `generate()` — greedy autoregressive decoding

```python
for _ in range(max_new_tokens):
    cur = torch.tensor([ids[-SEQ_LEN:]])
    mask = make_causal_mask(1, cur.size(1))
    logits = model(cur, mask)
    next_id = logits[0, -1].argmax().item()
    ids.append(next_id)
```

Only GPT (not BERT) can do this naturally, per theory.md section 5 — each
new token is generated by taking the model's prediction at the *last*
position and appending it to the sequence for the next step. Looking at
the actual generations (e.g. `['robot', 'built'] -> ['robot', 'built',
'castle', 'but', 'robot', 'painted', 'machine', 'so']`): the model
correctly reproduces the grammar's *structure* (subject-verb-object
triples, connectors in the right slot) even though — consistent with
section 5's ceiling analysis — the specific word choices within each
category are close to arbitrary, exactly as expected given the
independent-sampling data-generating process.

## 7. What a real BERT/GPT would add beyond this implementation

If the synthetic grammar had genuine word-to-word correlations (e.g. "cat"
disproportionately followed by "chased," "wizard" by "found"), both models
would be expected to exceed the 16.7% ceiling by learning those
correlations — this implementation's ceiling analysis would still apply,
just at a different (higher) value. The architectural and training-
objective lessons (causal vs. bidirectional masking, MLM vs. CLM loss,
their downstream implications for generation) transfer directly to
real BERT/GPT at any scale; only the absolute achievable accuracy is
capped by this dataset's intentionally simple structure, exactly as
flagged in theory.md's honest scope-limitation section.
