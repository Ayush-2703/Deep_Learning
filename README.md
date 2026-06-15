<div align="center">

<!-- HERO BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=Deep%20Learning%20Mastery&fontSize=52&fontColor=ffffff&fontAlignY=38&desc=From%20Perceptrons%20to%20Diffusion%20Models%20—%20Theory%20%2B%20Code%20%2B%20Explanation&descAlignY=60&descColor=a78bfa" width="100%"/>

<!-- BADGES -->
<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Topics-30%2B-8B5CF6?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Structure-Theory%20%7C%20Code%20%7C%20Explanation-10B981?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-F59E0B?style=for-the-badge"/>
</p>

<p>
  <img src="https://img.shields.io/github/stars/Ayush-2703/deep-learning-mastery?style=social"/>
  <img src="https://img.shields.io/github/forks/Ayush-2703/deep-learning-mastery?style=social"/>
  <img src="https://img.shields.io/github/watchers/Ayush-2703/deep-learning-mastery?style=social"/>
</p>

<br/>

> **The most structured deep learning curriculum on GitHub.**  
> Every topic ships with rigorous mathematical theory, production-grade PyTorch code, and a line-by-line explanation — nothing is left unexplained.

<br/>

[![Open in Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/Ayush-2703/deep-learning-mastery)
[![View on GitHub](https://img.shields.io/badge/View%20on-GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Ayush-2703/deep-learning-mastery)

</div>

---

## 📌 Table of Contents

- [Why This Repository](#-why-this-repository)
- [The 3-Component System](#-the-3-component-system)
- [Curriculum Map](#-curriculum-map)
- [Phase Breakdown](#-phase-breakdown)
- [Live Results Snapshot](#-live-results-snapshot)
- [Getting Started](#-getting-started)
- [Repository Structure](#-repository-structure)
- [Prerequisites](#-prerequisites)
- [Progress Tracker](#-progress-tracker)
- [Contributing](#-contributing)
- [Author](#-author)

---

## 🎯 Why This Repository

Most deep learning resources give you one of three things: theory without code, code without intuition, or tutorials that don't scale. This repository gives you all three — for every single topic, without exception.

| What you'll find here | What you won't find here |
|---|---|
| ✅ Full mathematical derivations with LaTeX | ❌ "Just run this Colab" tutorials |
| ✅ Self-contained, runnable PyTorch code | ❌ Code that requires hidden setup |
| ✅ Line-by-line explanations of *why* | ❌ Copy-paste snippets without context |
| ✅ Inline tensor shape annotations | ❌ Undocumented tensor ops |
| ✅ Empirical proof of theoretical claims | ❌ Theory disconnected from results |
| ✅ CPU + GPU compatible, zero config | ❌ Environment hell |

---

## 🔩 The 3-Component System

Every topic in this repository follows an **identical, non-negotiable 3-file structure**:

```
topic-name/
├── theory.md           ← Mathematics, architecture, intuition
├── implementation.py   ← Production-ready PyTorch code
└── explanation.md      ← Line-by-line code breakdown
```

### What each file contains

<table>
<tr>
<td width="33%">

**📐 `theory.md`**
- Full mathematical derivations
- LaTeX-rendered equations
- ASCII/Mermaid architecture diagrams
- Geometric and visual intuition
- Historical context and paper references
- Common pitfalls and misconceptions

</td>
<td width="33%">

**⚙️ `implementation.py`**
- Fully executable PyTorch code
- Dataset loading + preprocessing
- Model definition + training loop
- Evaluation + metrics
- Matplotlib visualizations
- Best-weight saving & reproducibility

</td>
<td width="34%">

**🔍 `explanation.md`**
- Every non-obvious line explained
- Tensor shape traces `[B, T, D]`
- Design decisions justified
- Common bug table per topic
- `no_grad()` vs `eval()` distinctions
- Why each function call exists

</td>
</tr>
</table>

---

## 🗺 Curriculum Map

```
deep-learning-mastery/
│
├── 📦 phase-1-foundations/              ← The bedrock of everything
│   ├── 01-perceptron-and-mlp/
│   ├── 02-activation-functions/
│   ├── 03-gradient-descent-and-backprop/
│   ├── 04-loss-functions-and-overfitting/
│   ├── 05-regularization-optimizers-batchnorm/
│   ├── 06-hyperparameter-tuning-augmentation/
│   └── 07-extra-linear-algebra-pytorch-tensors/
│
├── 🖼  phase-2-cnns/                    ← Vision intelligence
│   ├── 01-convolution-basics/
│   ├── 02-architectures-lenet-to-densenet/
│   ├── 03-object-detection-rcnn-yolo/
│   ├── 04-segmentation-unet-maskrcnn/
│   └── 05-transfer-learning-finetuning/
│
├── 🔄 phase-3-sequential/              ← Memory and sequences
│   ├── 01-rnns/
│   ├── 02-lstm-and-gru/
│   ├── 03-seq2seq-nlp/
│   └── 04-extra-state-space-models-mamba/
│
├── 🧠 phase-4-attention-transformers/  ← The modern backbone
│   ├── 01-attention-mechanisms/
│   ├── 02-transformer-architecture/
│   └── 03-vision-transformers-swin/
│
├── 🎨 phase-5-generative-ai/           ← Create, not just classify
│   ├── 01-autoencoders-and-vaes/
│   ├── 02-gans-dcgan-cyclegan/
│   ├── 03-llms-bert-gpt/
│   ├── 04-diffusion-models-ddpm/
│   └── 05-extra-lora-rag-rlhf/
│
└── 🚀 phase-6-advanced-deployment/     ← From model to production
    ├── 01-graph-neural-networks/
    ├── 02-deep-reinforcement-learning/
    └── 03-mlops-quantization-onnx-serving/
```

---

## 📚 Phase Breakdown

<details>
<summary><b>Phase 1 — Deep Learning Foundations</b> (7 topics)</summary>

| # | Topic | Key Concepts |
|---|---|---|
| 01 | Perceptron & MLP | Biological neuron, convergence theorem, Universal Approximation |
| 02 | Activation Functions | Sigmoid, ReLU, GELU, SELU, SiLU — vanishing gradient proof |
| 03 | Gradient Descent & Backprop | SGD, mini-batch, autodiff chain rule, computational graphs |
| 04 | Loss Functions & Overfitting | BCE, CE, MSE, bias-variance trade-off |
| 05 | Regularization & Optimizers | L1/L2, Dropout, BatchNorm, Adam, RMSprop, Early Stopping |
| 06 | Hyperparameter Tuning | LR schedules, batch size effects, data augmentation strategies |
| 07 ★ | Linear Algebra & PyTorch | Tensor ops, broadcasting, einsum, autograd mechanics |

</details>

<details>
<summary><b>Phase 2 — Convolutional Neural Networks</b> (5 topics)</summary>

| # | Topic | Key Concepts |
|---|---|---|
| 01 | Convolution Basics | 1D/2D/3D kernels, parameter sharing, receptive field, pooling |
| 02 | Architectures | LeNet → AlexNet → VGG → ResNet → DenseNet → GoogLeNet |
| 03 | Object Detection | Faster R-CNN, anchor boxes, RPN, YOLO v1–v8 |
| 04 | Segmentation | U-Net skip connections, Mask R-CNN instance segmentation |
| 05 | Transfer Learning | Feature extraction vs fine-tuning, domain adaptation |

</details>

<details>
<summary><b>Phase 3 — Sequential Modeling</b> (4 topics)</summary>

| # | Topic | Key Concepts |
|---|---|---|
| 01 | RNNs | Unrolled computation, BPTT, exploding/vanishing gradients |
| 02 | LSTM & GRU | Cell state, forget/input/output gates, GRU simplification |
| 03 | Seq2Seq | Encoder-decoder, teacher forcing, attention bridge |
| 04 ★ | State Space Models | Mamba, Jamba, selective scan, linear recurrence |

</details>

<details>
<summary><b>Phase 4 — Attention & Transformers</b> (3 topics)</summary>

| # | Topic | Key Concepts |
|---|---|---|
| 01 | Attention Mechanisms | Scaled dot-product, multi-head, causal masking |
| 02 | Transformer Architecture | Encoder-decoder, positional encoding, LayerNorm placement |
| 03 | Vision Transformers | ViT patch embedding, Swin shifted windows, hierarchical features |

</details>

<details>
<summary><b>Phase 5 — Generative AI & LLMs</b> (5 topics)</summary>

| # | Topic | Key Concepts |
|---|---|---|
| 01 | Autoencoders & VAEs | Bottleneck, ELBO, reparameterization trick |
| 02 | GANs | DCGAN, CycleGAN, mode collapse, Wasserstein loss |
| 03 | LLMs — BERT & GPT | Masked LM, causal LM, fine-tuning, summarization, translation |
| 04 | Diffusion Models | DDPM, forward/reverse diffusion, noise schedules, U-Net backbone |
| 05 ★ | LoRA / RAG / RLHF | Parameter-efficient fine-tuning, retrieval augmentation, DPO alignment |

</details>

<details>
<summary><b>Phase 6 — Advanced Topics & Deployment</b> (3 topics)</summary>

| # | Topic | Key Concepts |
|---|---|---|
| 01 | Graph Neural Networks | Message passing, GCN, GAT, node/edge/graph classification |
| 02 | Deep Reinforcement Learning | DQN, PPO, policy gradients, replay buffer |
| 03 | MLOps | INT8 quantization, ONNX export, FastAPI serving, Dockerization |

</details>

---

## 📊 Live Results Snapshot

These are **actual empirical outputs** from the implementations — not theoretical claims:

### Phase 1 — Topic 1: Perceptron & MLP

| Experiment | Result |
|---|---|
| Perceptron — AND gate | Converged in **4 epochs**, 100% accuracy |
| Perceptron — XOR gate | Hit max 1000 epochs, stuck at **50%** *(proves single-layer limitation)* |
| MLP Scratch — XOR | **100% accuracy** from epoch ~2000 onward |
| Production MLP — make_moons | **99% val accuracy** in 150 epochs, 6,465 parameters |

### Phase 1 — Topic 2: Activation Functions

| Activation | Input-Layer Gradient Norm | Output-Layer Gradient Norm | Ratio |
|---|---|---|---|
| Sigmoid | 2.82e-10 | 2.62e-01 | **1.1e-09 ← catastrophic vanishing** |
| Tanh | 1.78e-01 | 9.63e-02 | 1.8× ← manageable |
| ReLU | 3.78e-03 | 3.16e-03 | 1.2× ← stable |
| SiLU/Swish | — | — | **Lowest val loss (0.0155)** |

*Every topic will include its own results table as it is completed.*

---

## ⚡ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Ayush-2703/deep-learning-mastery.git
cd deep-learning-mastery
```

### 2. Set up the environment

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Run any topic independently

Every implementation is **fully self-contained** — no shared state, no hidden dependencies:

```bash
# Example: Run the MLP implementation
python phase-1-foundations/01-perceptron-and-mlp/implementation.py

# Example: Run the Activation Functions comparison
python phase-1-foundations/02-activation-functions/implementation.py
```

### 4. Open in Google Colab (zero setup)

Click the Colab badge at the top of any `implementation.py` or browse directly:

```
https://colab.research.google.com/github/Ayush-2703/deep-learning-mastery
```

---

## 📁 Repository Structure

```
deep-learning-mastery/
│
├── README.md
├── requirements.txt
├── .github/
│   └── workflows/
│       └── ci.yml                  ← Auto-test all implementations
│
├── phase-1-foundations/
│   └── 01-perceptron-and-mlp/
│       ├── theory.md               ← 9 sections, UAT proof, convergence bound
│       ├── implementation.py       ← 430+ lines, 7 self-contained sections
│       └── explanation.md          ← 10 sections, 10-bug pitfall table
│
│   └── 02-activation-functions/
│       ├── theory.md               ← 9 sections, vanishing gradient bound (0.25)^L
│       ├── implementation.py       ← 580+ lines, hook-based dead neuron counter
│       └── explanation.md          ← 8 sections, closure-based hook deep dive
│
│   └── ...                         ← All 7 topics, same structure
│
├── phase-2-cnns/                   ← Coming next
├── phase-3-sequential/
├── phase-4-attention-transformers/
├── phase-5-generative-ai/
└── phase-6-advanced-deployment/
```

---

## 🛠 Prerequisites

**Python knowledge assumed:**
- Comfortable with Python classes, decorators, and list comprehensions
- Familiarity with NumPy arrays

**Math assumed:**
- High-school calculus (derivatives, chain rule)
- Basic linear algebra (matrices, dot products)

> Everything beyond this is taught from scratch inside `theory.md`.

**Software:**

```
Python     ≥ 3.10
PyTorch    ≥ 2.0
NumPy      ≥ 1.24
Matplotlib ≥ 3.7
scikit-learn ≥ 1.3
```

---

## ✅ Progress Tracker

| Phase | Topics | Status |
|---|---|---|
| Phase 1 — Foundations | 7 topics | 🟩🟩⬜⬜⬜⬜⬜ In Progress |
| Phase 2 — CNNs | 5 topics | ⬜ Not Started |
| Phase 3 — Sequential | 4 topics | ⬜ Not Started |
| Phase 4 — Attention & Transformers | 3 topics | ⬜ Not Started |
| Phase 5 — Generative AI & LLMs | 5 topics | ⬜ Not Started |
| Phase 6 — Advanced & Deployment | 3 topics | ⬜ Not Started |

**Overall: 2 / 27 topics complete**

> ⭐ Star the repo to get notified when new topics drop.

---

## 🤝 Contributing

Contributions are welcome, but this repository maintains a **strict quality bar**. Before submitting a PR, please read the guidelines:

1. Every PR must follow the 3-component structure (`theory.md`, `implementation.py`, `explanation.md`)
2. Code must run end-to-end without modification
3. Tensor shapes must be annotated inline: `# [batch, seq_len, d_model]`
4. No black-box functions — every non-obvious call gets a comment
5. Results must be empirically verified and included in the explanation

```bash
# Fork, clone, and create a branch
git checkout -b feature/phase-2-topic-1-convolutions

# After your changes
git commit -m "feat: add phase-2/01-convolution-basics (theory + impl + explanation)"
git push origin feature/phase-2-topic-1-convolutions
```

Then open a Pull Request with a brief description of what your implementation demonstrates empirically.

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.  
You're free to use, fork, and build on this for personal and commercial projects.

---

## 👤 Author

<div align="center">

### Ayush Kumar Singh

**B.Tech — AI**

*Researcher in Adversarial ML, Geospatial AI, and LLM Systems*

[![GitHub](https://img.shields.io/badge/GitHub-Ayush--2703-181717?style=for-the-badge&logo=github)](https://github.com/Ayush-2703)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-ayushsingh2703-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/ayushsingh2703)

</div>

---

<div align="center">

**If this repository helped you, please consider giving it a ⭐**  
*It takes 2 seconds and helps others discover it.*

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=100&section=footer" width="100%"/>

</div>
