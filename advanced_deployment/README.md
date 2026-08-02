# Phase 6: Advanced Deployment

Three topics, each independently runnable, CPU-only, synthetic data,
PyTorch (+ networkx for Topic 1, + onnx/onnxruntime/onnxscript for Topic 3).
Every `implementation.py` was actually executed end-to-end during this
build — see each folder's `explanation.md` for honest results, including
three real bugs found and fixed during development (documented below, not
smoothed over).

| # | Topic | Data | Core result |
|---|-------|------|-------------|
| 01 | Graph Neural Networks (GCN) | Synthetic stochastic-block-model graph (4 communities) | GCN reached 98.4% test accuracy vs. 41.9% for a graph-blind MLP baseline with identical features -- a controlled, direct demonstration that graph structure carries exploitable signal |
| 02 | Deep RL (DQN) | Custom synthetic GridWorld (built from scratch, no gym dependency) | Learned a near-optimal policy: reward improved from -0.70 (first 50 episodes) to +0.80 (last 50), reaching 100% greedy-policy success rate over 100 held-out evaluation episodes |
| 03 | MLOps (Quantization, ONNX, Serving) | Synthetic tabular binary classification | INT8 dynamic quantization shrank the model 68.7% but was measured *slower* than FP32 on this small model (0.24ms vs 0.09ms) -- reported honestly rather than assuming quantization always helps; ONNX Runtime was fastest overall (0.033ms) |

## Real bugs found and fixed during this build (see each topic's explanation.md for full detail)

1. **Topic 3**: `torch.onnx.export` crashed outright (`ModuleNotFoundError: onnxscript`)
   until the `onnxscript` dependency was installed -- this torch version's exporter
   defaults to a dynamo-based path requiring it.
2. **Topic 3**: the exported ONNX model's reported size (6.36 KB) was misleading --
   the exporter split weights into a separate `model.onnx.data` file (76 KB) not
   counted in the first measurement. Fixed to sum both files (82.36 KB true total).
3. (Phase 5, Topic 3 -- Diffusion/DDPM) a noise-schedule bug left `x_T` retaining
   36% of the original signal instead of being ~pure noise, silently degrading
   generation quality until the schedule was corrected. Included here for
   cross-reference since it's part of the same overall repository.

Run any topic standalone:
```
cd 0X-topic-name/
python3 implementation.py
```

Requires: `torch`, `numpy`, `matplotlib`, `networkx` (Topic 1),
`onnx`, `onnxruntime`, `onnxscript` (Topic 3). CPU-only, no GPU/CUDA needed,
no internet dataset downloads.
