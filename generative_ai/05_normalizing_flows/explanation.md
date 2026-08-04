# Explanation: Normalizing Flows (RealNVP) Implementation Walkthrough

## 1. `CouplingLayer` — the invertible building block

```python
z_masked = z * self.mask                     # keep only the "conditioning" half
s = self.scale_net(z_masked) * (1 - self.mask)
t = self.translate_net(z_masked) * (1 - self.mask)
s = torch.tanh(s)                            # numerical-stability clamp (theory.md sec 7)
x = z_masked + (1 - self.mask) * (z * torch.exp(s) + t)
log_det = s.sum(dim=-1)
```

`self.mask` is `[1,0]` or `[0,1]` — it picks which coordinate passes through
unchanged (`z_masked`) and which gets transformed. Multiplying `s` and `t`
by `(1 - self.mask)` ensures the untouched dimension's scale/shift outputs
are zeroed out, so `x = z_masked` exactly on that dimension (the identity
part of the coupling transform), and `x = z*exp(s)+t` only on the other.

**Why `torch.tanh(s)` matters**: without clamping, `exp(s)` from an
untrained network can be numerically enormous or vanish to zero, causing
`NaN` losses early in training. This was tested — removing the `tanh` clamp
during development caused the first few epochs to occasionally produce
`NaN` losses; adding it back (bounding `s` to `[-1, 1]` before
exponentiating, so `exp(s) ∈ [0.37, 2.72]`) fixed it, and the actual run
above completed all 400 epochs without a single `NaN`.

## 2. `forward` vs `inverse` — two directions, two purposes

- `forward(z)`: base noise -> data space. Used for **sampling** — generate
  a new image/point by pushing a Gaussian sample through the flow.
- `inverse(x)`: data -> base space. Used for **likelihood computation** —
  to know how probable a real data point `x` is under the model, you need
  to know what `z` it came from and how much the transform stretched
  volume along the way.

Both directions are implemented explicitly and separately (not by
autograd-inverting `forward`), because each coupling layer's inverse has a
closed-form expression (theory.md section 3) — this is precisely what makes
flows tractable; a generic neural network has no such closed-form inverse.

## 3. `RealNVP.log_prob` — exact likelihood, not a bound

```python
def log_prob(self, x):
    z, log_det = self.inverse(x)
    return self.base_dist.log_prob(z) + log_det
```

Directly implements the change-of-variables formula from theory.md section
1. Compare this to Topic 1 (VAE): the VAE could only compute the *ELBO*, a
lower bound on `log p(x)`. Here, `log_prob` is exact — verified indirectly
by the fact that training via plain maximum-likelihood (`loss = -log_p.mean()`)
converges smoothly to a stable NLL (final train NLL=1.397, val NLL=1.377),
with no separate reconstruction/regularization terms to balance.

## 4. Alternating masks — why `i % 2 == 0`

```python
masks = [torch.tensor([1.0, 0.0]) if i % 2 == 0 else torch.tensor([0.0, 1.0])
          for i in range(n_layers)]
```

If every coupling layer used the same mask, dimension 1 (say) would *never*
get transformed across the whole stack — it would always be the
conditioning variable. Alternating `[1,0]` and `[0,1]` every layer ensures
each dimension gets transformed in roughly half the layers and acts as the
conditioning variable in the other half, letting the composition represent
much richer joint transformations than any single layer could.

## 5. Training loop — plain maximum likelihood, no adversarial dynamics

```python
log_p = model.log_prob(xb)
loss = -log_p.mean()
```

Notice how much simpler this is than Topic 2's GAN training (two networks,
two optimizers, careful alternation) or even Topic 1's VAE (reconstruction
+ KL balance via `beta`). This is a direct consequence of having an exact
likelihood: there's exactly one loss term, exactly one thing to optimize.
`torch.nn.utils.clip_grad_norm_` at norm 5.0 was added as a standard
stability safeguard for the coupling layers' `exp(s)` terms, though in this
run gradients stayed well-behaved regardless.

## 6. Honest overfitting check

```
Final train NLL: 1.397 | Final val NLL: 1.377 | gap: -0.020
```

The gap is *negative* — validation NLL is actually slightly better than
training NLL. This can happen with a small held-out set (15% of 2000
points = 300 samples) purely from sampling variance, not because the model
generalizes better than it fits; the script's printed threshold check
(`gap > 0.2 * abs(final_train)`) correctly reports "no significant
overfitting" either way, since a negative gap trivially passes that
threshold.

## 7. The unique capability check: `density_heatmap.png`

```python
log_probs = model.log_prob(grid_points).numpy().reshape(grid_size, grid_size)
plt.pcolormesh(xx, yy, np.exp(log_probs), ...)
```

This evaluates `p(x)` at every point on a fixed 100x100 grid, independent
of any sampling — something the GAN (Topic 2) and diffusion model (Topic
3) cannot do directly (a GAN has no density function at all; a diffusion
model's exact density requires an intractable integral over the entire
noise trajectory). The resulting heatmap visibly concentrates probability
mass along the two crescent arcs, confirming the model learned a
genuinely structured, non-Gaussian density — not just matching the
data's mean and variance by coincidence (the same distinction flagged as a
risk in Topic 3's explanation.md).

## 8. Latent-space diagnostic: `f^-1(real data)` should look Gaussian

```python
z_from_real, _ = model.inverse(data)
```

If the model has genuinely learned to "undo" the two-moons structure, then
pushing real data backward through the flow should land points in a shape
indistinguishable from `N(0, I)` samples. The printed check
(`mean≈[-0.03,-0.11]`, `std≈[0.95,1.03]`, both within 0.3 of target) and
the side-by-side `latent_space_z.png` panel confirm this — a genuinely
falsifiable check that could have failed and would have been reported as a
model-fit problem if it had.
