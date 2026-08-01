# Explanation: DDPM Implementation Walkthrough

## 1. Synthetic two-moons data (`make_two_moons`)

Generated manually via parametric equations for two half-circle arcs
(`cos`/`sin` of a uniform angle for moon 1, a shifted/flipped version for
moon 2) plus Gaussian jitter — no `sklearn.datasets.make_moons` dependency.
Normalized to zero mean / unit variance per dimension so the diffusion
process's fixed noise schedule (calibrated for roughly unit-scale data)
behaves as expected.

## 2. `q_sample` — the forward-process shortcut

```python
sqrt_ab = alpha_bar[t].sqrt().unsqueeze(-1)
sqrt_1m_ab = (1 - alpha_bar[t]).sqrt().unsqueeze(-1)
return sqrt_ab * x0 + sqrt_1m_ab * noise, noise
```

This is the closed-form equation from theory.md section 2 — it lets us jump
directly from `x_0` to `x_t` for any `t`, without simulating steps 1 through
`t-1`. `t` here is a batch of *different* random timesteps (one per sample),
which is what makes DDPM training efficient: every training step sees a
random point along the whole noise trajectory, not a fixed one.

## 3. A real bug that was caught and fixed: insufficient noise schedule

This is the part to read carefully, per the repository's "no smoothing over
bugs" rule. The first version of this script used `T=200` steps with
`beta` linear from `1e-4` to `0.02` — values copied from the standard DDPM
paper, which uses `T=1000`. Running the numbers:

```
alpha_bar[T-1] = 0.132   (should be ~0 for x_T to be "pure noise")
```

With only 200 steps at that beta range, `x_T` still retained about 36% of
the original signal (`sqrt(alpha_bar[T-1]) ≈ 0.36`) instead of being
approximately `N(0, I)`. This is a real, silent bug: nothing crashes,
training loss still goes down, but the reverse-sampling process starts from
`torch.randn(...)` — genuine pure noise — while the model was *trained* to
denoise a distribution where `x_T` still had leftover structure. That
mismatch between the assumed and actual `x_T` distribution is exactly what
produced the first run's bad result: generated samples collapsed to a
central blob instead of the two-moons shape (visible in the very first
`final_comparison.png` before the fix — since overwritten by the corrected
run, but the loss numbers before/after are preserved in this file for the
record: MSE went from 0.94→0.50 before the fix to 0.92→0.33 after it).

**Fix:** raise `beta_max` from `0.02` to `0.05`, which drives
`alpha_bar[T-1]` down to `0.006` — restoring the "x_T is approximately pure
noise" assumption the whole reverse-sampling algorithm depends on. After
the fix, the generated distribution visibly recovers the crescent shape in
`final_comparison.png` and `reverse_sampling_trajectory.png`, and the
training loss drop nearly doubled (2.5-3x vs. the earlier ~1.9x). This is
the single most important lesson in this topic: a diffusion model can look
like it's training fine (loss decreasing, no errors) while still being
fundamentally miscalibrated, because the noise schedule and `T` are coupled
parameters — shortening `T` requires increasing `beta` to compensate, and
skipping that check produces confidently-wrong generation.

## 4. `SinusoidalTimeEmbed` — how the network knows *which* timestep

```python
freqs = torch.exp(-math.log(10000) * torch.arange(half).float() / half)
args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
```

Same construction as Transformer positional embeddings (Phase 4) — `t` is a
scalar integer, but the network needs a rich, smoothly-varying
representation of it so nearby timesteps produce similar embeddings and the
MLP can generalize across `t`. Directly feeding raw integer `t` as a single
scalar input would give the network almost no useful signal to condition on.

## 5. `DenoiseMLP.forward` — conditioning on both `x` and `t`

```python
te = self.time_embed(t)
return self.net(torch.cat([x, te], dim=-1))
```

The 2D point `x` and the 32-dim time embedding are concatenated before the
first linear layer — this is the standard (simple) way to condition an MLP
on an auxiliary input, versus the FiLM-style conditioning or cross-attention
used in image U-Nets. For 2D toy data this simple concatenation is
sufficient; it would likely be a bottleneck for high-resolution images.

## 6. Training loop — one line does most of the conceptual work

```python
t_batch = torch.randint(0, T, (x0.size(0),))
x_t, noise = q_sample(x0, t_batch)
noise_pred = model(x_t, t_batch)
loss = ((noise - noise_pred) ** 2).mean()
```

Four lines implement the entirety of DDPM training: sample a random
timestep per example, noise the data to that timestep, ask the network to
predict the noise that was added, and minimize the MSE. No unrolling, no
recurrence — this is what makes diffusion models tractable to train despite
requiring many sequential steps at sampling time.

## 7. `sample()` — the expensive, sequential part

```python
for t in reversed(range(T)):
    ...
    mean = (1 / torch.sqrt(alpha_t)) * (x - coef * eps_pred)
    if t > 0:
        x = mean + torch.sqrt(beta_t) * z
    else:
        x = mean
```

Notice `z` (fresh noise) is added at every step *except* the last
(`t == 0`) — adding noise at the final step would just re-corrupt the
finished sample for no reason. This loop runs `T=200` full forward passes
through the network per batch of samples generated — the direct, concrete
cost of diffusion's iterative sampling that theory.md's tradeoff table
refers to.

## 8. Distributional comparison — why mean/std alone isn't enough

The script prints a mean/std comparison as a coarse sanity check, but the
real evidence of success is the visual shape match in
`final_comparison.png`, not the summary statistics — two very different
distributions can share the same mean and standard deviation (a blob and a
crescent can both be zero-mean, unit-variance). This is flagged explicitly
here rather than treating the numeric check as sufficient on its own.
