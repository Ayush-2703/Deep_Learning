<div align="center">

![Phase 5: Generative AI](https://capsule-render.vercel.app/api?type=waving&color=0:0B0C0E,50:363B45,100:586174&height=200&section=header&text=Phase%205:%20Generative%20AI&fontSize=30&fontColor=ffffff&fontAlignY=25&animation=fadeIn&desc=Deep%20Learning&descSize=25&descAlignY=58)

**Made with ❤️ by [Ayush Kumar Singh](https://github.com/Ayush-2703)**

</div>

---

Six topics spanning the major families of generative modeling —
autoencoding, adversarial training, autoregressive language modeling,
diffusion, normalizing flows — plus a practical closing topic on how
large models are actually adapted and steered once trained: LoRA, RAG, and
RLHF. Almost everything runs CPU-only on synthetic data (two-moons point
clouds, procedurally-drawn shape images, synthetic grammars and knowledge
bases); the one deliberate exception is Topic 01, which trains on real,
downloaded MNIST via `torchvision` rather than synthetic data — flagged
explicitly below rather than left unremarked, since it breaks this
repository's usual convention. Every runnable topic's `explanation.md`
reports the real numbers that came out — including two genuinely silent
bugs (a miscalibrated diffusion noise schedule, a tokenizer bug that broke
half a retrieval evaluation) that produced no crash or exception and were
only caught by checking whether the result actually made sense.

Every topic follows the repository's 3-file structure:

```
0X-topic-name/
├── theory.md            ← Full derivations, ASCII diagrams, historical context
├── implementation.py    ← Runnable PyTorch/NumPy code
└── explanation.md       ← Line-by-line walkthrough + live results
```

## 📌 Table of Contents

- [Topics](#topics)
- [Generative Model Family Comparison](#generative-model-family-comparison)
- [Two Real Bugs, Caught the Same Way](#two-real-bugs-caught-the-same-way)
- [Notable Engineering Detours and Honest Findings](#notable-engineering-detours-and-honest-findings-see-each-topics-explanationmd-for-full-detail)
- [Running the Code](#running-the-code)

---

## Topics

| # | Topic | Data | Core result |
|---|-------|------|-------------|
| 01 | Autoencoders & VAEs | **MNIST** — real, downloaded via `torchvision` (the one exception to this phase's synthetic-data convention) | The reparameterization trick, ELBO, and the AE→VAE probabilistic generalization implemented and verified structurally (tensor shape traces, the closed-form KL-divergence derivation checked term-by-term against the code); the VAE's key advantage over a vanilla AE — coherent generation by sampling `z ~ N(0,I)` directly from the prior — follows directly from the KL term regularizing the latent space, since a plain AE's latent space has no probabilistic structure to safely sample from |
| 02 | GANs — DCGAN & CycleGAN | 1,500 synthetic shape images (DCGAN); 300 synthetic images/domain (CycleGAN) | DCGAN reaches an intermediate discriminator/generator balance after 30 epochs (`D(real)=0.843`, `D(fake)=0.160`) — reported as genuinely moderate rather than cropped to look better; CycleGAN's cycle-consistency L1 loss drops `1.75→0.099` over a deliberately abbreviated, explicitly-labeled 15-epoch "skeleton" run that still exercises the full two-generator/two-discriminator/cycle-loss algorithm |
| 03 | LLMs — BERT vs. GPT (mini) | Synthetic subject-verb-object grammar, independently-sampled words | A single shared attention implementation drives *both* models — only the attention mask passed in differs — directly demonstrating that the BERT/GPT distinction is a masking choice, not a different architecture; GPT's next-token accuracy (17.4%) lands almost exactly on a rigorously-derived 16.7% chance ceiling; a confusing 27.4% aggregate BERT accuracy was decomposed by masking sub-type and traced to the "left unchanged" 10% sub-case being trivially 100% solvable |
| 04 | Diffusion Models (DDPM) | Synthetic two-moons point cloud | A real, documented bug — `T=200` steps at the paper's default beta range left `alpha_bar[T-1]=0.132` (37% of the original signal still present at the supposed "pure noise" endpoint) — silently miscalibrated reverse sampling despite training loss decreasing normally; raising `beta_max` from `0.02` to `0.05` restored the correct near-zero endpoint (`0.006`) and the generated distribution recovered the two-moons crescent shape |
| 05 | Normalizing Flows (RealNVP) | Synthetic two-moons point cloud | The only architecture in this entire repository with an **exact**, tractable likelihood rather than a bound or no likelihood at all — confirmed by a stable, smoothly-converging NLL (train=1.397, val=1.377) optimized with a single loss term and no adversarial or KL-balancing act; the learned density visibly concentrates along the two crescent arcs on a full grid evaluation, and inverting real data through the flow recovers an approximately `N(0,I)` latent distribution as a falsifiable structural check |
| 06 | LoRA, RAG & RLHF | Synthetic two-domain classification (LoRA); synthetic knowledge-base Q&A (RAG); synthetic digit-preference task (RLHF) | LoRA reaches comparable Domain B performance to full fine-tuning using **34× fewer** trainable parameters (2,048 vs. 70,165), with Domain A performance exactly recoverable by disabling the adapter (`diff=0.000000`); a tokenizer bug silently broke half of a RAG evaluation (retrieval accuracy stuck at 52.5%) before a regex fix restored a perfect 100%; RLHF's policy shifted its *ground-truth* reward from 0.935→5.497 (out of 8) while optimizing only a learned reward model that never saw the true reward function — confirmed as genuine generalization, not reward hacking, via a 269/300 output-diversity check ruling out mode collapse |

---

## Generative Model Family Comparison

Four fundamentally different approaches to "learn `p(x)` and sample from
it" sit inside this phase — this repository's own diffusion theory
already lays out the VAE/GAN/Diffusion trade-offs; extended here with the
Normalizing Flow row this phase also covers:

| Property | VAE (01) | GAN (02) | Diffusion (04) | Normalizing Flow (05) |
|---|---|---|---|---|
| **Training stability** | High | Low — adversarial, two competing networks | High | High — single MLE loss, no min-max game |
| **Sample quality** | Blurry | Sharp | Sharp | Sharp |
| **Sampling speed** | 1 pass | 1 pass | `T` sequential passes | 1 pass |
| **Likelihood** | Approximate — ELBO, a lower bound | Not available at all | Approximate — ELBO-derived | **Exact** |
| **Mode coverage** | Good | Prone to collapse | Good | Good |

Normalizing Flows trade away GANs' sample sharpness-per-compute-cost and
diffusion's sampling flexibility for the one thing neither can offer: an
exact density you can evaluate at any point, not just sample from — which
is exactly what Topic 05's grid-evaluated density heatmap demonstrates
directly, something no other model in this phase can produce.

---

## Two Real Bugs, Caught the Same Way

The two most instructive findings in this phase weren't architecture
choices — they were silent bugs, and both were caught by the same
discipline: not trusting that "the code ran without crashing" means "the
code is correct," and instead checking whether the *result* matched a
reasonable expectation.

- **Topic 04 (Diffusion)**: training loss decreased normally, no exception
  was ever raised, yet the noise schedule was miscalibrated — `x_T` still
  carried 37% of the original signal instead of being approximately pure
  noise. The only way this surfaced was generated samples visibly
  collapsing to a blob instead of the two-moons shape, prompting a check
  of `alpha_bar[T-1]`'s actual numerical value against the "should be ~0"
  assumption the whole reverse-sampling algorithm depends on.
- **Topic 06 (RAG)**: retrieval accuracy sitting at a suspicious 52.5% —
  suspicious specifically because entity names are unique per document and
  *should* be trivial to retrieve via exact term overlap — led to
  inspecting `tokenize()` directly, where an unstripped apostrophe
  (`"zorvath's"` never matching `"zorvath"`) was silently breaking every
  profession-related query.

Neither bug produced a stack trace. Both were caught only because a
result that looked "fine but a bit underwhelming" was treated as worth
investigating rather than accepted as an inherent limitation.

---

## Notable Engineering Detours and Honest Findings (see each topic's `explanation.md` for full detail)

1. **Topic 01**: `log_var` (not `var` or `std`) is the encoder's output
   parametrization specifically because it's unconstrained — no
   positivity-enforcing activation is needed, since `std =
   exp(0.5·log_var)` is guaranteed positive for any real input.
2. **Topic 01** trains on real MNIST rather than synthetic data — the one
   exception to this phase's usual convention — and its `explanation.md`
   is written as a design-rationale walkthrough rather than reporting
   specific live training numbers, unlike every other topic in this
   phase. Flagged explicitly here rather than left unremarked.
3. **Topic 02**: DCGAN's `D(real)=0.84` / `D(fake)=0.16` balance after 30
   epochs is reported as genuinely moderate, not retrained or cropped
   until it looked more polished.
4. **Topic 02**: CycleGAN is explicitly labeled a "skeleton" — 15 epochs
   on 300 synthetic images per domain instead of the paper's 100–200
   epochs on thousands of real photos — specifically to keep every real
   algorithmic component CPU-feasible while stating plainly that image
   quality won't be polished at this scale.
5. **Topic 03**: a single attention implementation feeds both the BERT
   and GPT models in this topic — only the mask passed in differs — a
   concrete architectural demonstration rather than two independent
   reimplementations of the same underlying mechanism.
6. **Topic 04**: RealNVP's `tanh` clamp on the coupling layer's scale term
   was added specifically after `NaN` losses appeared during early
   development epochs, and the reasoning is kept in the code comments
   rather than silently included with no explanation of why it's there.
7. **Topic 06**: an initial "sanity check" wrongly assumed a trained
   LoRA adapter should leave other domains' accuracy unchanged (since the
   base weights are frozen) — Domain A accuracy actually dropped
   substantially with the adapter *enabled*. The corrected check verifies
   the claim that's actually true: disabling the adapter exactly
   reproduces the frozen base model's behavior (`diff=0.000000`).
8. **Topic 06**: RLHF's policy was optimized only against a learned
   reward model — never once against the true `ground_truth_reward`
   function directly — yet ground-truth reward still rose from near-random
   (0.935) to 5.497 out of a maximum of 8, with a 269/300 output-diversity
   check specifically run to rule out mode collapse as an alternative,
   less interesting explanation for the gain.

---

## Running the Code

```bash
cd 0X-topic-name/
python3 implementation.py
```

Requires: `torch`, `numpy`, `matplotlib` for every topic. Topic 01
additionally requires `torchvision` and `scikit-learn`, and — unlike
every other topic in this repository — downloads real MNIST data on
first run rather than generating synthetic data, so it needs internet
access the first time it's run. Every other topic is CPU-only with no
internet dependency at all.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0B0C0E,50:363B45,100:586174&height=70&section=footer" width="100%"/>

</div>
