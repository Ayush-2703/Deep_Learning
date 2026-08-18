<div align="center">

![Phase 6: Advanced Deployment](https://capsule-render.vercel.app/api?type=waving&color=0:0B0C0E,50:363B45,100:586174&height=200&section=header&text=Phase%206:%20Advanced%20Deployment&fontSize=30&fontColor=ffffff&fontAlignY=25&animation=fadeIn&desc=Deep%20Learning&descSize=25&descAlignY=58)

**Made with ❤️ by [Ayush Kumar Singh](https://github.com/Ayush-2703)**

</div>

---

Three topics that each step outside the "train a model, report a metric"
pattern of earlier phases: Graph Neural Networks (a genuinely different
input structure — graphs, not grids or sequences), Deep Reinforcement
Learning (a genuinely different training signal — reward from
interaction, not a labeled dataset), and MLOps (what happens to a model
*after* training — quantization, cross-framework export, serving). All
three are CPU-only and independently runnable, and every
`implementation.py` was actually executed end-to-end during this build —
see each folder's `explanation.md` for the honest results, including two
real, documented bugs from Topic 03 (an outright crash and a genuinely
misleading measurement), reported below rather than smoothed over.

Every topic follows the repository's 3-file structure:

```
0X-topic-name/
├── readme.md            ← Full derivations, ASCII diagrams, historical context
├── implementation.py    ← Runnable PyTorch/NumPy code
└── explanation.md       ← Line-by-line walkthrough + live results
```

## 📌 Table of Contents

- [Topics](#topics)
- [The Same Experimental Discipline, Three Times](#the-same-experimental-discipline-three-times)
- [Real Bugs Found and Fixed](#real-bugs-found-and-fixed-see-topic-03s-explanationmd-for-full-detail)
- [Notable Engineering Detours and Honest Findings](#notable-engineering-detours-and-honest-findings-see-each-topics-explanationmd-for-full-detail)
- [Running the Code](#running-the-code)

---

## Topics

| # | Topic | Data | Core result |
|---|-------|------|-------------|
| 01 | Graph Neural Networks (GCN) | Synthetic stochastic-block-model graph, 4 communities (avg. degree 12.63, 0 isolated nodes), with deliberately weak per-node features | GCN reaches **98.4%** test accuracy vs. **41.9%** for an architecturally-identical MLP baseline trained on the same features with the graph hidden from it — a controlled, direct demonstration that message passing recovers signal (community structure) that isn't present in any single node's own features; a 4-snapshot embedding-evolution visualization (epochs 1/50/100/200, projected to 2D) shows the four communities visibly collapsing from a jumbled cloud into separated clusters over training |
| 02 | Deep Reinforcement Learning (DQN) | Custom synthetic GridWorld built from scratch (no `gym` dependency), with obstacles at (2,2)/(2,3)/(3,2) | Both DQN stabilization tricks (target network, experience replay) and epsilon-greedy exploration implemented and exercised in full; average reward improved from **-0.70** (first 50 episodes) to **+0.80** (last 50) while average episode length dropped from **23.9 to 11.7 steps** — success and efficiency improving together, not just one; a separate epsilon=0 greedy-policy evaluation over 100 fresh episodes confirms **100% success** at an average of **10.0 steps** — within one step of the true obstacle-avoiding shortest-path optimum |
| 03 | MLOps — Quantization, ONNX Export & Serving | Synthetic 20-feature tabular binary classification (5 informative, 15 pure-noise nuisance features) | INT8 dynamic quantization shrinks the model **68.7%** (78.93→24.70 KB) but is measured **slower**, not faster, than FP32 on this small model (0.243 ms vs. 0.090 ms) — reported honestly rather than assuming quantization always helps; ONNX Runtime is fastest overall (0.033 ms), for an entirely unrelated reason (a statically-optimized execution graph, not quantization); all three backends reach **100% prediction agreement**, with ONNX's logits matching FP32's to `0.000000` — bit-exact, not just argmax-equal |

---

## The Same Experimental Discipline, Three Times

These three topics don't share an architecture family the way, say, the
Generative AI phase's VAE/GAN/Diffusion/Flow topics do — but they do share
something more fundamental: every headline result in this phase comes from
an experiment designed so that exactly **one variable changes** and
everything else is held fixed, so the result can only mean what it claims
to mean.

- **Topic 01**: the MLP baseline has the same hidden width, same dropout,
  same optimizer settings, and the same number of training epochs as the
  GCN. The *only* difference is whether the forward pass mixes in
  neighbor information (`A_norm @ H`) or processes each node independently
  (`fc1(X)`). The 98.4% vs. 41.9% gap is therefore attributable to the
  graph structure specifically — not to a confound in model capacity or
  training budget.
- **Topic 02**: the "first 50 episodes" and "last 50 episodes" comparison
  uses the identical network, environment, and reward function throughout
  a single training run — the only thing that changed is how much
  training the agent had received. The separate greedy-policy evaluation
  (epsilon forced to exactly 0) then isolates *what the network learned*
  from *how it behaved while still exploring* — two genuinely different
  questions that a single training-reward curve would conflate.
- **Topic 03**: FP32, INT8-quantized, and ONNX-exported are three
  different *runtimes* for the exact same trained weights — not three
  separately trained models. Any difference in prediction or latency is
  therefore attributable to the numeric/serialization format alone, which
  is what makes the `max_logit_diff = 0.000000` result a meaningful
  correctness proof rather than a coincidence.

---

## Real Bugs Found and Fixed (see Topic 03's `explanation.md` for full detail)

1. **Topic 03**: `torch.onnx.export` crashed outright
   (`ModuleNotFoundError: No module named 'onnxscript'`) on the first run.
   This PyTorch version's exporter defaults to a newer dynamo-based path
   that depends on the separate `onnxscript` package, not installed
   alongside `torch`/`onnx`/`onnxruntime` by default — fixed with
   `pip install onnxscript`. A real, environment-specific issue that only
   surfaces by actually running the code, not by writing it from memory
   of how the API "usually" works.
2. **Topic 03**: the exported ONNX model's first reported size (6.36 KB)
   was genuinely misleading — dramatically smaller than the 78.93 KB FP32
   PyTorch file, which should have been suspicious on its own. The cause:
   this exporter writes large weight tensors to a separate external-data
   file (`model.onnx.data`, 76 KB) rather than embedding them in the
   `.onnx` file itself. Fixed by summing both files for the true on-disk
   footprint — 82.36 KB, now honestly larger than the FP32 file, not
   smaller.
3. *(Cross-reference, Phase 5 Topic 04 — Diffusion/DDPM)*: a noise-schedule
   bug left `x_T` retaining ~37% of the original signal instead of being
   near-pure noise, silently degrading generation quality until the
   schedule was corrected. Noted here since it's the same class of bug as
   #2 above — a plausible-looking number that turned out to be measuring
   the wrong thing — and it's part of the same overall repository.

---

## Notable Engineering Detours and Honest Findings (see each topic's `explanation.md` for full detail)

1. **Topic 01**: node features are given only a small, deliberately weak
   class-correlated bump on top of random noise (`X_feat[i, labels[i] %
   feature_dim] += 0.6`) — specifically so an MLP baseline *can't* solve
   the task from features alone, which is what makes the GCN-vs-MLP
   comparison a meaningful test of what the graph structure contributes,
   rather than a foregone conclusion.
2. **Topic 01**: the symmetric normalization (`D_hat^(-1/2) · A_hat ·
   D_hat^(-1/2)`) is computed once, outside the training loop, since the
   graph structure itself never changes — only node features and weights
   update per step. Recognizing which quantities are static vs. trainable
   ahead of time turns each layer's message-passing step into a single
   dense matrix multiply rather than a per-epoch recomputation.
3. **Topic 02**: the environment includes goal position in every
   observation (normalized to `[0,1]`) even though this particular run
   never moves the goal — a deliberate design choice so the network
   genuinely learns a goal-conditioned policy rather than one implicitly
   hard-coded to a single fixed goal location.
4. **Topic 02**: the `-0.01` per-step reward exists specifically to
   incentivize *shorter* successful paths — with a reward of exactly 0
   per step, any path reaching the goal would look equally good to the
   learned Q-values, removing the agent's only incentive to be efficient.
5. **Topic 02**: training only begins once the replay buffer holds at
   least 500 transitions (`MIN_BUFFER_BEFORE_TRAIN`), so the earliest
   gradient updates aren't computed from a handful of highly-correlated,
   early-episode transitions before the buffer has any real diversity to
   sample from.
6. **Topic 02**: the learned policy is also audited visually — an arrow
   grid where every non-terminal cell's arrow is literally `argmax_a
   Q(s,a)` for that cell — described as a stronger check than the reward
   curve alone, since a reward curve could look good from a lucky
   evaluation seed, while a full grid of sensible-looking arrows is much
   harder to get by chance.
7. **Topic 03**: only 5 of 20 input features are actually
   class-informative; the remaining 15 are pure noise — ensuring the base
   FP32 model has to do genuine learning rather than memorize a
   near-trivial threshold, which matters for the later
   quantized/ONNX-agreement check to be a meaningful test.
8. **Topic 03**: a `DeprecationWarning` noting that the quantization API
   used here (`torch.ao.quantization`) is slated for removal in favor of
   `torchao`'s newer API is reported as-is in the write-up rather than
   silently suppressed — a genuine signal for anyone maintaining this
   code that the exact call may need to migrate in a future PyTorch
   version.
9. **Topic 03**: the observed 68.7% size reduction from quantization is
   explicitly compared against, and doesn't quite reach, the theoretical
   4× (75%) maximum from FP32→INT8 — attributed honestly to biases and
   any non-`Linear` parameters remaining FP32, plus fixed serialization
   overhead, rather than left unexplained as a rounding curiosity.

---

## Running the Code

```bash
cd 0X-topic-name/
python3 implementation.py
```

Requires: `torch`, `numpy`, `matplotlib`, `networkx` (Topic 01 only),
`onnx`, `onnxruntime`, `onnxscript` (Topic 03 only). CPU-only, no GPU/CUDA
needed, no internet dataset downloads — every task in this phase is
procedurally generated.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0B0C0E,50:363B45,100:586174&height=70&section=footer" width="100%"/>

</div>
