# Large Language Models: BERT (encoder, MLM) vs. GPT (decoder, causal LM)

## 1. Scope note relative to Phase 4

Phase 4 (Attention & Transformers) already covers the Transformer
architecture itself in depth (encoder/decoder blocks, multi-head
attention, positional encoding). This topic assumes that architectural
foundation and focuses specifically on what makes BERT and GPT different
**as pretraining objectives and usage patterns**, not on re-deriving
attention from scratch.

## 2. The core architectural difference: attention masking direction

Both BERT and GPT are stacks of Transformer blocks. The entire behavioral
difference between them comes down to **what each token is allowed to
attend to**:

```
BERT (encoder, bidirectional):
  token_i can attend to ALL tokens in the sequence (both before and after)

GPT (decoder, causal/autoregressive):
  token_i can only attend to tokens at position <= i (itself and earlier)
```

This is implemented via an attention mask: GPT applies a triangular
("causal") mask that sets attention scores to `-inf` for any `j > i`
before the softmax, so future tokens contribute exactly zero probability
mass to token `i`'s representation. BERT applies no such mask — full
bidirectional context.

## 3. BERT's pretraining objective: Masked Language Modeling (MLM)

Since BERT can see the whole sequence at once, "predict the next token"
would be a trivial task (the answer is already visible). Instead, BERT is
trained by randomly masking a percentage of input tokens (15% in the
original paper) and predicting only the masked tokens from their
bidirectional context:

```
Input:  "the [MASK] sat on the [MASK]"
Target: predict "cat" at position 2, "mat" at position 6
```

This forces the model to build representations that use context from
*both directions* to fill in missing information — well suited to
understanding/classification tasks (the original BERT paper also uses a
Next Sentence Prediction auxiliary objective; this implementation focuses
on MLM, the primary and more impactful of the two per later ablation
studies such as RoBERTa).

## 4. GPT's pretraining objective: Causal Language Modeling (CLM)

GPT is trained with the much simpler objective of predicting the next
token given everything before it:

```
Input:     "the cat sat on the"
Target:    "cat sat on the mat"     (shifted by one position)
```

Every position's target is just the next token in the sequence — the
causal mask (section 2) is what prevents this from being trivially
solvable by looking ahead. This objective directly matches how GPT is
*used* at inference time (autoregressive generation, one token at a time),
which is why GPT-family models are the natural choice for open-ended text
generation, while BERT-family models are typically used for classification/
extraction tasks requiring a fine-tuning head rather than free-form
generation.

## 5. Why this distinction matters practically

| | BERT (encoder) | GPT (decoder) |
|---|---|---|
| Attention | Bidirectional | Causal (left-to-right only) |
| Pretraining objective | Masked token prediction | Next-token prediction |
| Natural use case | Classification, extraction, embeddings | Free-form generation |
| Can generate text left-to-right? | Not directly (would need separate decoding scheme) | Yes, natively |
| Sees "future" tokens during training? | Yes (that's the point) | Never |

## 6. What this implementation covers

Two small Transformer models, sharing the same architecture family
(embedding + positional encoding + N transformer blocks + output head),
differing only in the attention mask and the training objective, trained
on the same synthetic token sequences — this isolates exactly the
BERT-vs-GPT distinction described above, rather than comparing two models
that also differ in size, data, or tokenization.

**Synthetic data**: token sequences generated from a simple synthetic
grammar (a small vocabulary with fixed positional patterns, e.g.
`subject verb object` triples with some structure), so both objectives
have genuine, learnable structure to recover — a real signal for the
"did MLM/CLM training actually work" checks in explanation.md.

## 7. Honest scope limitation

Real BERT/GPT are trained on billions of tokens of natural language with
vocabularies of 30,000+ subword tokens. This implementation uses a tiny
synthetic vocabulary (~40 tokens) and a few thousand training sequences,
specifically to keep CPU training feasible within this repository's
constraints while still exercising the real mechanism (causal vs.
bidirectional masking, MLM vs. CLM loss) — the architectural and
objective-level lessons transfer; the absolute scale does not.
