<div align="center">

![Phase 1: Foundations](https://capsule-render.vercel.app/api?type=waving&color=0:0B0C0E,50:363B45,100:586174&height=200&section=header&text=Phase%201:%20Foundations&Fields&fontSize=23&fontColor=ffffff&fontAlignY=25&animation=fadeIn&desc=Deep%20Learning&descSize=20&descAlignY=58)
<br/>
**Made with ❤️ by [Ayush Kumar Singh](https://github.com/Ayush-2703)**
</div>

---
Seven topics, each independently runnable, CPU-only, built on synthetic data
or small scikit-learn datasets (`make_moons`, `make_circles`) — no external
downloads needed. Every `implementation.py` was actually executed end-to-end
during this build; each topic's `explanation.md` reports the real numbers
that came out, including a couple of counter-intuitive results that were
kept rather than smoothed over (see below).

Every topic follows the repository's 3-file structure:

```
0X-topic-name/
├── theory.md           ← Full derivations, ASCII diagrams, historical context
├── implementation.py   ← Runnable PyTorch/NumPy code
└── explanation.md      ← Line-by-line walkthrough + live results
```

## Topics

| # | Topic | Data | Core result |
|---|-------|------|-------------|
| 01 | Perceptron & MLP | NumPy AND/XOR gates, `make_moons` | Perceptron converges on AND in 4 epochs (100% acc) but plateaus at 50% on XOR — a direct proof of the linear-separability limit; a from-scratch MLP then solves XOR at 100%, and a production `nn.Module` MLP hits 99% val accuracy on `make_moons` with 6,465 parameters |
| 02 | Activation Functions | Synthetic 15-layer MLP (gradient flow), `make_circles` | Sigmoid's gradient at layer 1 is ~10⁹× smaller than at the output layer (severe vanishing) vs. ReLU's near-flat 1.2× ratio; training accuracy follows suit — Sigmoid 75% vs. Tanh/ReLU/SiLU/GELU at 99–99.5% |
| 03 | Gradient Descent & Backprop | 2D quadratic bowl, manual NumPy MLP | Hand-derived backprop gradients verified against central-difference numerical gradients to well under 1e-5 relative error; also reproduces PyTorch's gradient-accumulation behavior (`.grad` growing 6 → 12 → 18 over 3 steps) when `zero_grad()` is skipped |
| 04 | Loss Functions & Overfitting | `make_moons`, 5-architecture complexity sweep | Empirical bias-variance trade-off: the smallest model (4 hidden units) shows Bias²=0.0982 vs. the largest (128×128) at Bias²=0.0483 — but variance roughly doubles (0.0036 → 0.0059) going the other way |
| 05 | Regularization, Optimizers & BatchNorm | Synthetic tabular data | L1 (λ=1e-4) drives 36.3% of weights near-zero vs. L2 (λ=1e-3) at 10.1% — 3.6× sparser despite a *smaller* λ; early stopping (patience=15) matches a 300-epoch run's best validation loss in just 40 epochs via best-weight restoration (7.5× less compute) |
| 06 | Hyperparameter Tuning & Augmentation | `make_moons` (240 samples) | Random search matched grid search's best accuracy (96.9%) while covering 6 unique hidden-size values vs. grid's 3, for the same 9-run budget; reported honestly: none of Gaussian noise, feature dropout, or Mixup improved validation accuracy here, because the baseline wasn't actually overfitting in the first place |
| 07 | Linear Algebra & PyTorch Tensors | None — pure math/tensor identities | Every matrix-calculus identity (linear/quadratic form gradients, Jacobians, broadcasting) is checked against PyTorch autograd with `torch.allclose` assertions, so the script fails loudly rather than silently printing a wrong value |

## Notable pitfalls and honest findings (see each topic's explanation.md for full detail)

1. **Topic 01**: skipping `.unsqueeze(1)` on labels before `nn.BCELoss` causes a
   silent broadcasting bug — `(N,)` against `(N,1)` broadcasts to `(N,N)`, so
   training still produces a finite (but wrong) loss instead of erroring out.
2. **Topic 02**: dead-neuron counts were *higher* under the "proper" low
   learning rate (9/256) than under a deliberately high one (3/256) — with
   plain SGD, low LR just means neurons that dipped negative haven't had
   enough signal yet to recover, not that they're permanently dead. Kept in
   as a nuance rather than forced to match the textbook expectation.
3. **Topic 03**: PyTorch accumulates (not replaces) gradients on
   `.backward()` by design — useful for micro-batch accumulation, but a
   forgotten `optimizer.zero_grad()` silently grows gradients every step.
4. **Topic 06**: augmentation was tested and reported as *not helping* on
   this problem — the validation gap was already negative before any
   augmentation was applied, so there was no overfitting left to fix.

Run any topic standalone:
```
cd 0X-topic-name/
python3 implementation.py
```

Requires: `torch`, `numpy`, `matplotlib`, `scikit-learn`. CPU-only, no
GPU/CUDA needed, no internet dataset downloads.
