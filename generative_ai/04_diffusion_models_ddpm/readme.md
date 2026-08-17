<div align="center">

![Theory: Denoising Diffusion Probabilistic Models](https://capsule-render.vercel.app/api?type=waving&color=0:0B0C0E,50:363B45,100:586174&height=200&section=header&text=Theory:%20Denoising%20Diffusion%20Probabilistic%20Models&fontSize=30&fontColor=ffffff&fontAlignY=25&animation=fadeIn&desc=Deep%20Learning&descSize=25&descAlignY=58)

</div>

---

## 1. The core idea

Diffusion models learn to generate data by reversing a gradual noising
process. Two processes:

- **Forward process** `q(x_t | x_{t-1})`: fixed, no learning — gradually
  adds Gaussian noise to data `x_0` over `T` timesteps until `x_T` is
  approximately pure noise `N(0, I)`.
- **Reverse process** `p_theta(x_{t-1} | x_t)`: learned — a neural network
  predicts how to remove a small amount of noise at each step, eventually
  turning pure noise back into a data sample.

## 2. Forward process, in closed form

Define a noise schedule `beta_1, ..., beta_T` (small, increasing values,
e.g. linearly from 1e-4 to 0.02). At each step:

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1}, beta_t * I)
```

Critically, this process has a closed-form shortcut — you don't need to
simulate all `t` steps to get `x_t` from `x_0`:

```
alpha_t = 1 - beta_t
alpha_bar_t = product(alpha_1 * alpha_2 * ... * alpha_t)

x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon,   epsilon ~ N(0, I)
```

This is the same reparameterization idea as in the VAE (Topic 1) — sampling
`x_t` directly is a deterministic function of `x_0` and a noise draw
`epsilon`, so it's cheap to compute for any random `t` during training.

## 3. What the network actually predicts

Rather than predicting `x_{t-1}` directly, DDPM (Ho et al., 2020) found it
works much better to have the network predict the **noise** `epsilon` that
was added:

```
epsilon_theta(x_t, t) ≈ epsilon
```

Given a predicted `epsilon_theta`, you can recover an estimate of `x_0`:

```
x_0_hat = ( x_t - sqrt(1 - alpha_bar_t) * epsilon_theta ) / sqrt(alpha_bar_t)
```

## 4. Training objective (simplified)

The full ELBO-derived loss simplifies (Ho et al. show the simplified
version works as well or better in practice) to plain noise-prediction MSE:

```
L_simple = E_{t, x_0, epsilon} [ || epsilon - epsilon_theta(x_t, t) ||^2 ]
```

Training loop:
1. Sample real data `x_0`
2. Sample random timestep `t ~ Uniform(1, T)`
3. Sample noise `epsilon ~ N(0, I)`
4. Compute `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon`
5. Predict `epsilon_theta(x_t, t)`, minimize MSE against the true `epsilon`

Note this is a **single-step** training objective per sample — the network
never has to unroll the full chain during training, which is what makes
diffusion models tractable to train despite having hundreds/thousands of
generation steps at sampling time.

## 5. Sampling (reverse process), step by step

Starting from `x_T ~ N(0, I)`, iteratively for `t = T, T-1, ..., 1`:

```
z ~ N(0, I) if t > 1 else z = 0

x_{t-1} = 1/sqrt(alpha_t) * ( x_t - (beta_t / sqrt(1 - alpha_bar_t)) * epsilon_theta(x_t, t) )
          + sqrt(beta_t) * z
```

This is the expensive part of diffusion models: generation requires `T`
sequential network evaluations (vs. one forward pass for a GAN generator or
VAE decoder). This implementation uses a small `T` (200 steps) on 2D toy
data specifically to keep full sampling CPU-feasible while still
demonstrating the real iterative denoising mechanism.

## 6. Why 2D toy data instead of images here

A full image-based DDPM normally uses a U-Net with self-attention across
hundreds of training epochs — well beyond CPU-feasible runtime for this
repository's constraints. Toy 2D distributions (two interleaving "moons")
are the standard pedagogical setup for diffusion models precisely because
they let you *see* the entire forward noising trajectory and the entire
reverse denoising trajectory as simple scatter plots, while the underlying
math (noise schedule, closed-form forward sampling, epsilon-prediction
network, iterative reverse sampling) is identical to the image case. The
denoising network here is a small MLP conditioned on `(x, t)` rather than a
convolutional U-Net.

## 7. Diffusion vs. GAN vs. VAE — practical tradeoffs

| Property              | VAE          | GAN              | Diffusion       |
|-----------------------|--------------|------------------|-----------------|
| Training stability    | High         | Low (adversarial)| High            |
| Sample quality        | Blurry       | Sharp            | Sharp           |
| Sampling speed        | 1 pass       | 1 pass           | T sequential passes |
| Likelihood-based      | Yes (ELBO)   | No               | Yes (ELBO-derived) |
| Mode coverage         | Good         | Prone to collapse| Good            |

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0B0C0E,50:363B45,100:586174&height=70&section=footer" width="100%"/>

</div>
