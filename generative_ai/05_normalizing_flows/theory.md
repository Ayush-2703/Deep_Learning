<div align="center">

![Theory: Normalizing Flows](https://capsule-render.vercel.app/api?type=waving&color=0:0B0C0E,50:363B45,100:586174&height=200&section=header&text=Theory:%20Normalizing%20Flows&fontSize=30&fontColor=ffffff&fontAlignY=25&animation=fadeIn&desc=Deep%20Learning&descSize=25&descAlignY=58)

</div>

---

## 1. The core idea: exact likelihood via invertible transforms

VAEs give a *lower bound* on likelihood (the ELBO). GANs give no likelihood
at all. Normalizing flows are unusual among generative models: they give an
**exact** likelihood, by construction, using the change-of-variables formula
for probability densities.

The idea: start from a simple distribution `z ~ p_Z(z)` (e.g. standard
Gaussian), and apply an invertible, differentiable transformation
`x = f(z)`. Because `f` is invertible, the density of `x` can be computed
exactly:

```
p_X(x) = p_Z(f^{-1}(x)) * | det( d f^{-1}(x) / dx ) |
```

Or equivalently, in the direction we sample from (`z -> x`):

```
log p_X(x) = log p_Z(z) - log | det( df(z)/dz ) |,   where z = f^{-1}(x)
```

The `log|det(Jacobian)|` term accounts for how the transformation locally
stretches or compresses volume — density has to be corrected by exactly
this factor to remain a valid probability distribution.

## 2. Why you need special architectures

For an arbitrary neural network `f`, computing `det(Jacobian)` costs
`O(d^3)` for `d`-dimensional data — completely intractable for anything
beyond tiny dimensions. Normalizing flows are built from specific
transformation families chosen so the Jacobian is triangular (or otherwise
structured), making the determinant trivial: **the product of the diagonal
entries**, computed in `O(d)`.

## 3. Coupling layers (RealNVP-style) — the building block used here

Split the input `z = [z_1, z_2]` (e.g. first half / second half of
dimensions). Transform only `z_2`, conditioned on `z_1`, leaving `z_1`
unchanged:

```
x_1 = z_1
x_2 = z_2 * exp(s(z_1)) + t(z_1)
```

where `s(.)` and `t(.)` ("scale" and "translate") are arbitrary neural
networks — they can be as complex as you like, because they never need to
be inverted themselves; only the overall coupling transform needs to be
invertible, and it is, by construction:

```
Inverse:  z_1 = x_1
          z_2 = (x_2 - t(x_1)) / exp(s(x_1))
```

The Jacobian of this transform is lower-triangular (`x_1` doesn't depend on
`z_2`, and `dx_1/dz_1 = I`), so:

```
det(J) = exp( sum(s(z_1)) )
log det(J) = sum(s(z_1))
```

— a simple sum, no actual determinant computation needed.

## 4. Why stack multiple coupling layers, and why alternate the split

A single coupling layer leaves half the dimensions completely untouched
(`x_1 = z_1`), which alone can't represent a rich transformation. Stacking
several layers, **alternating which half is passed through unchanged**
each time, lets every dimension eventually get transformed conditioned on
every other dimension across the stack:

```
z -> [coupling 1: transform dims 2 | condition on dims 1]
  -> [coupling 2: transform dims 1 | condition on dims 2]
  -> [coupling 3: transform dims 2 | condition on dims 1]
  -> ... -> x
```

Total log-det-Jacobian for the stack is just the sum of each layer's
log-det (log-det of a composition of functions is the sum of log-dets,
directly from the chain rule / properties of determinants of products).

## 5. Training objective: exact maximum likelihood

```
L = -E_{x~data} [ log p_Z(f^{-1}(x)) + log|det(df^{-1}(x)/dx)| ]
```

Unlike the VAE's ELBO (a bound) or the GAN's adversarial loss (no
likelihood at all), this is directly and exactly the negative
log-likelihood of the data under the model — minimizing it is exact maximum
likelihood estimation, no approximation involved.

## 6. What this implementation covers

A stack of RealNVP-style affine coupling layers trained on the same
synthetic 2D two-moons distribution used in Topic 3 (Diffusion), which
lets the two topics be directly compared: both are exact/near-exact
likelihood-based generative approaches, but flows compute likelihood
exactly and sample in a single forward pass (no iterative denoising),
at the cost of architectural constraints (every layer must be invertible
with a cheap Jacobian).

## 7. Known limitations of coupling-layer flows

- Expressiveness is bounded by how many coupling layers you stack — too few
  layers and complex multi-modal distributions (like two-moons) may only be
  partially captured.
- The affine coupling transform is still fairly restrictive (scale +
  shift); more expressive flows (spline flows, autoregressive flows) trade
  more computation for better fit.
- Numerical stability of `exp(s(z_1))` requires care — unclamped scale
  networks can produce exploding or vanishing volume changes early in
  training.
