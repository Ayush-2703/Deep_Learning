<div align="center">

![Theory: Autoencoders & Variational Autoencoders](https://capsule-render.vercel.app/api?type=waving&color=0:0B0C0E,50:363B45,100:586174&height=200&section=header&text=Theory:%20Autoencoder%20and%20Variational%20Autoencoders&Fields&fontSize=30&fontColor=ffffff&fontAlignY=25&animation=fadeIn&desc=Deep%20Learning&descSize=25&descAlignY=58)
<br/>
**Made with ❤️ by [Ayush Kumar Singh](https://github.com/Ayush-2703)**
</div>

---

## Table of Contents
1. [The Information Bottleneck Principle](#1-the-information-bottleneck-principle)
2. [Vanilla Autoencoder Architecture](#2-vanilla-autoencoder-architecture)
3. [Loss Function: Reconstruction Loss](#3-loss-function-reconstruction-loss)
4. [Latent Space Geometry](#4-latent-space-geometry)
5. [From AE to VAE: The Probabilistic Leap](#5-from-ae-to-vae-the-probabilistic-leap)
6. [Variational Inference & ELBO](#6-variational-inference--elbo)
7. [The Reparameterization Trick](#7-the-reparameterization-trick)
8. [VAE Architecture Deep Dive](#8-vae-architecture-deep-dive)
9. [KL Divergence Analytical Form](#9-kl-divergence-analytical-form)
10. [Beta-VAE and Disentanglement](#10-beta-vae-and-disentanglement)
11. [Applications](#11-applications)

---

## 1. The Information Bottleneck Principle

An autoencoder learns to compress data through a **bottleneck** (latent space) and reconstruct it. The fundamental trade-off:

$$\mathcal{L}_{\text{IB}} = I(X; Z) - \beta I(Z; Y)$$

Where:
- $I(X; Z)$ = mutual information between input $X$ and representation $Z$ (compression)
- $I(Z; Y)$ = mutual information between representation $Z$ and target $Y$ (prediction)
- $\beta$ = Lagrange multiplier controlling the trade-off

For unsupervised autoencoders, we maximize reconstruction fidelity while minimizing the "description length" of $Z$.

---

## 2. Vanilla Autoencoder Architecture

### Encoder
Maps input to latent representation:
$$\mathbf{z} = f_{\text{enc}}(\mathbf{x}; \theta) = \sigma(\mathbf{W}_e \mathbf{x} + \mathbf{b}_e)$$

### Decoder
Maps latent back to input space:
$$\hat{\mathbf{x}} = f_{\text{dec}}(\mathbf{z}; \phi) = \sigma(\mathbf{W}_d \mathbf{z} + \mathbf{b}_d)$$

### Dimensionality Constraint
$$\dim(\mathbf{z}) \ll \dim(\mathbf{x})$$

This forces the network to learn **salient features** rather than identity mapping.

### Architectural Diagram

```
Input x ∈ R^d          Latent z ∈ R^k           Reconstruction x̂ ∈ R^d
    │                        │                           │
    ▼                        ▼                           ▼
┌─────────┐             ┌──────────┐                 ┌───────────┐
│  Linear │             │Bottleneck│                 │  Linear   │
│ d → 512 │───────────▶│  k=32     │───────────────▶│ 512 → d   │
│  + ReLU │             │          │                 │  + Sigmoid│
└─────────┘             └──────────┘                 └───────────┘
    │                        │                           │
    └───── Encoder ──────────┘                           │
                             └────── Decoder ────────────┘
```

---

## 3. Loss Function: Reconstruction Loss

### Binary Cross-Entropy (for [0,1]-normalized data):
$$\mathcal{L}_{\text{BCE}}(\mathbf{x}, \hat{\mathbf{x}}) = -\sum_{i=1}^{d} \left[ x_i \log(\hat{x}_i) + (1 - x_i) \log(1 - \hat{x}_i) \right]$$

### Mean Squared Error (for continuous data):
$$\mathcal{L}_{\text{MSE}}(\mathbf{x}, \hat{\mathbf{x}}) = \frac{1}{d} \sum_{i=1}^{d} (x_i - \hat{x}_i)^2$$

**Key insight**: BCE assumes Bernoulli-distributed pixels; MSE assumes Gaussian. For MNIST (binary-ish), BCE is theoretically preferred.

---

## 4. Latent Space Geometry

### The Interpolation Problem
In a vanilla AE, the latent space is **not structured**. Two issues arise:

1. **Non-continuous**: Gaps in latent space decode to nonsensical outputs
2. **Non-smooth**: Nearby points may decode to wildly different outputs

### Visual Intuition

```
Vanilla AE Latent Space          Ideal Latent Space
┌─────────────────┐             ┌─────────────────┐
│  ●        ●     │             │    ●───●───●    │
│                 │             │   ╱         ╲   │
│        ●        │             │  ●     ●     ●  │
│                 │             │   ╲         ╱   │
│  ●        ●     │             │    ●───●───●    │
└─────────────────┘             └─────────────────┘
  Disconnected clusters          Smooth, continuous manifold
```

---

## 5. From AE to VAE: The Probabilistic Leap

Instead of learning a deterministic mapping $\mathbf{z} = f(\mathbf{x})$, a VAE learns a **probability distribution** over latent variables:

$$q_{\phi}(\mathbf{z} | \mathbf{x}) \approx p(\mathbf{z} | \mathbf{x})$$

### Generative Story
1. Sample $\mathbf{z} \sim p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ (prior)
2. Generate $\mathbf{x} \sim p_{\theta}(\mathbf{x} | \mathbf{z})$ (decoder)

### The Challenge
The true posterior is intractable:
$$p(\mathbf{z} | \mathbf{x}) = \frac{p_{\theta}(\mathbf{x} | \mathbf{z}) p(\mathbf{z})}{p(\mathbf{x})}$$

The marginal likelihood $p(\mathbf{x}) = \int p_{\theta}(\mathbf{x} | \mathbf{z}) p(\mathbf{z}) d\mathbf{z}$ requires integrating over all possible $\mathbf{z}$ — computationally impossible for high-dimensional $Z$.

---

## 6. Variational Inference & ELBO

We introduce a **variational posterior** $q_{\phi}(\mathbf{z} | \mathbf{x})$ (the encoder) to approximate the true posterior.

### Deriving the ELBO

Starting from log-likelihood:
$$\log p_{\theta}(\mathbf{x}) = \log \int p_{\theta}(\mathbf{x} | \mathbf{z}) p(\mathbf{z}) d\mathbf{z}$$

Multiply by $\frac{q_{\phi}(\mathbf{z}|\mathbf{x})}{q_{\phi}(\mathbf{z}|\mathbf{x})} = 1$:
$$= \log \mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})} \left[ \frac{p_{\theta}(\mathbf{x} | \mathbf{z}) p(\mathbf{z})}{q_{\phi}(\mathbf{z} | \mathbf{x})} \right]$$

By Jensen's inequality ($\log$ is concave, so $\log \mathbb{E}[\cdot] \geq \mathbb{E}[\log(\cdot)]$):
$$\geq \mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})} \left[ \log p_{\theta}(\mathbf{x} | \mathbf{z}) \right] - \mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})} \left[ \log \frac{q_{\phi}(\mathbf{z} | \mathbf{x})}{p(\mathbf{z})} \right]$$

### The Evidence Lower Bound (ELBO):
$$\boxed{\mathcal{L}_{\text{ELBO}} = \underbrace{\mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})} \left[ \log p_{\theta}(\mathbf{x} | \mathbf{z}) \right]}_{\text{Reconstruction}} - \underbrace{D_{\text{KL}}(q_{\phi}(\mathbf{z} | \mathbf{x}) \| p(\mathbf{z}))}_{\text{Regularization}}}$$

### Why ELBO Works
- **Reconstruction term**: How well does the decoder reconstruct $\mathbf{x}$ from samples of $q_{\phi}(\mathbf{z}|\mathbf{x})$?
- **KL term**: How close is the learned posterior to the prior $\mathcal{N}(\mathbf{0}, \mathbf{I})$?
- The gap between $\log p(\mathbf{x})$ and ELBO equals $D_{\text{KL}}(q_{\phi}(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}|\mathbf{x}))$

---

## 7. The Reparameterization Trick

### The Problem
To compute $\nabla_{\phi} \mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})}[\cdot]$, we need gradients through the sampling operation:
$$\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2 \mathbf{I})$$

Direct sampling is non-differentiable (stochastic node blocks gradients).

### The Solution
Rewrite sampling as a deterministic function of a noise variable:
$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \text{where } \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

Now gradients flow through $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$ (learnable parameters) while $\boldsymbol{\epsilon}$ is treated as a constant during backprop.

```
Without Reparameterization:        With Reparameterization:
┌──────────┐                       ┌──────────┐
│    μ     │                       │    μ     │
│    σ     │───► Sample ──► z      │    σ     │───► z = μ + σ⊙ε
└──────────┘      ↑                └──────────┘      ↑
                  │                                  │
            ┌─────┘                             ┌────┘
            │ ε ~ N(0,I)                        │ ε ~ N(0,I)
            │ (stops gradient)                  │ (detached constant)
```

---

## 8. VAE Architecture Deep Dive

### Encoder (Inference Network)
Outputs parameters of the posterior:
$$q_{\phi}(\mathbf{z} | \mathbf{x}) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}_{\phi}(\mathbf{x}), \text{diag}(\boldsymbol{\sigma}^2_{\phi}(\mathbf{x})))$$

Network outputs **two vectors**:
- $\boldsymbol{\mu} \in \mathbb{R}^k$ — mean of latent distribution
- $\log \boldsymbol{\sigma}^2 \in \mathbb{R}^k$ — log-variance (numerical stability)

### Decoder (Generative Network)
$$p_{\theta}(\mathbf{x} | \mathbf{z}) = \text{Bernoulli}(\mathbf{x}; \boldsymbol{\pi}_{\theta}(\mathbf{z})) \text{ or } \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_{\theta}(\mathbf{z}), \mathbf{I})$$

### Full Forward Pass
```
x ∈ R^784 (MNIST flattened)
    │
    ▼
┌─────────────────────────┐
│  Encoder Network        │
│  Linear(784 → 400)      │
│  ReLU                   │
│  Linear(400 → 400)      │
│  ReLU                   │
│  Linear(400 → k*2)      │  ← outputs [μ, log(σ²)]
└─────────────────────────┘
    │
    ├──► μ ∈ R^k
    └──► log(σ²) ∈ R^k
         │
         ▼
    z = μ + σ ⊙ ε,  ε ~ N(0, I)
         │
         ▼
┌─────────────────────────┐
│  Decoder Network        │
│  Linear(k → 400)        │
│  ReLU                   │
│  Linear(400 → 400)      │
│  ReLU                   │
│  Linear(400 → 784)      │
│  Sigmoid                │  ← outputs p(x|z)
└─────────────────────────┘
    │
    ▼
x̂ ∈ R^784
```

---

## 9. KL Divergence Analytical Form

For two diagonal Gaussians, KL divergence has a closed form:

$$D_{\text{KL}}(q_{\phi} \| p) = \frac{1}{2} \sum_{j=1}^{k} \left( \sigma_j^2 + \mu_j^2 - 1 - \log(\sigma_j^2) \right)$$

Where:
- $q_{\phi}(\mathbf{z}) = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$
- $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$

**Interpretation of each term**:
- $\sigma_j^2$: Penalizes variance deviation from 1
- $\mu_j^2$: Penalizes mean deviation from 0
- $-1$: Normalization constant
- $-\log(\sigma_j^2)$: Encourages non-degenerate variance (prevents collapse to point mass)

---

## 10. Beta-VAE and Disentanglement

Modify the ELBO with a hyperparameter $\beta$:

$$\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}_{q_{\phi}}[\log p_{\theta}(\mathbf{x} | \mathbf{z})] - \beta \cdot D_{\text{KL}}(q_{\phi}(\mathbf{z} | \mathbf{x}) \| p(\mathbf{z}))$$

- $\beta = 1$: Standard VAE
- $\beta > 1$: Stronger pressure toward factorized latent space → **disentangled representations**
- $\beta < 1$: Better reconstruction, less structured latent space

**Trade-off**: Higher $\beta$ improves interpretability but may hurt reconstruction quality.

---

## 11. Applications

| Application | How VAE Helps |
|-------------|---------------|
| **Image Generation** | Sample $z \sim \mathcal{N}(0, I)$ and decode |
| **Anomaly Detection** | High reconstruction error = anomaly |
| **Denoising** | Encode noisy input, decode clean output |
| **Data Imputation** | Encode partial data, decode complete reconstruction |
| **Latent Space Arithmetic** | "smile vector" = $z_{\text{smiling}} - z_{\text{neutral}}$ |
| **Semi-supervised Learning** | Use latent representations as features for classifiers |

---

## Key Equations Summary

| Concept | Equation |
|---------|----------|
| AE Reconstruction | $\hat{\mathbf{x}} = f_{\text{dec}}(f_{\text{enc}}(\mathbf{x}))$ |
| ELBO | $\mathcal{L} = \mathbb{E}[\log p(\mathbf{x}\|\mathbf{z})] - D_{\text{KL}}(q\|p)$ |
| Reparameterization | $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$ |
| KL (closed form) | $\frac{1}{2}\sum(\sigma^2 + \mu^2 - 1 - \log\sigma^2)$ |
| $\beta$-VAE | $\mathcal{L} = \text{Recon} - \beta \cdot D_{\text{KL}}$ |

---

## References
1. Kingma & Welling (2014). "Auto-Encoding Variational Bayes." ICLR.
2. Higgins et al. (2017). "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework." ICLR.
3. Doersch (2016). "Tutorial on Variational Autoencoders." arXiv:1606.05908.
"""

with open('/mnt/agents/output/phase5_01_vae_theory.md', 'w') as f:
    f.write(theory_md)
print("Saved theory.md")

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0B0C0E,50:363B45,100:586174&height=70&section=footer" width="100%"/>

</div>
