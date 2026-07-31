# Generative Adversarial Networks (GANs) — DCGAN and CycleGAN

## 1. The adversarial game

A GAN trains two networks against each other:

- **Generator** `G(z)`: maps noise `z ~ p(z)` (usually `N(0, I)`) to a fake
  sample `x_fake = G(z)`
- **Discriminator** `D(x)`: outputs a probability that `x` is real (from the
  true data distribution) vs fake (produced by `G`)

The minimax objective:

```
min_G max_D  V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p(z)}[log(1 - D(G(z)))]
```

`D` is trained to maximize this (correctly classify real vs fake).
`G` is trained to minimize it — i.e., to fool `D` into outputting high
probability for fake samples.

## 2. Non-saturating generator loss (used in practice)

Early in training, `D` is much stronger than `G`, so `log(1 - D(G(z)))`
saturates (gradient near zero) when `D(G(z))` is close to 0. The standard
fix, used here, is to instead have `G` maximize `log D(G(z))` directly:

```
L_D = -[ log D(x_real) + log(1 - D(G(z))) ]     (minimize)
L_G = -log D(G(z))                               (minimize)
```

This gives `G` strong gradients even when it's currently losing badly.

## 3. DCGAN architecture principles

DCGAN (Radford et al., 2015) established the conv-architecture conventions
still used today:

- Replace pooling with **strided convolutions** (discriminator) and
  **transposed convolutions** (generator) for learned up/downsampling
- Use **BatchNorm** in both G and D (except G's output layer and D's input
  layer) to stabilize training
- **ReLU** activations in G (except output: `Tanh`, matching data
  normalized to `[-1, 1]`)
- **LeakyReLU** in D (avoids dead gradients when discriminating)
- No fully-connected hidden layers — all-convolutional

```
G:  z [B,100,1,1] -> ConvTranspose -> [B,128,7,7]
                   -> ConvTranspose -> [B,64,14,14]
                   -> ConvTranspose -> [B,1,28,28] -> Tanh

D:  x [B,1,28,28] -> Conv(stride2) -> [B,64,14,14] -> LeakyReLU
                   -> Conv(stride2) -> [B,128,7,7] -> LeakyReLU
                   -> Conv -> [B,1,1,1] -> Sigmoid
```

## 4. CycleGAN — unpaired image-to-image translation

CycleGAN extends the GAN idea to translate between two *unpaired* image
domains X and Y (e.g. horses <-> zebras) using two generators
(`G: X->Y`, `F: Y->X`) and two discriminators (`D_X`, `D_Y`), plus a
**cycle-consistency loss**:

```
L_cyc = E_x[ || F(G(x)) - x ||_1 ] + E_y[ || G(F(y)) - y ||_1 ]
```

The intuition: if you translate an image to the other domain and back, you
should get (approximately) the original image. This is what allows training
without paired examples — there's no direct `(x, y)` supervision, only the
constraint that the round-trip is consistent.

```
Full CycleGAN loss:
L = L_GAN(G, D_Y, X, Y) + L_GAN(F, D_X, Y, X) + lambda * L_cyc(G, F)
```

**Honest scope note for this implementation:** CycleGAN requires two full
image domains (e.g. two distinct datasets to translate between) and
typically trains for hundreds of epochs on real photographic data to show
convincing translation. Given the CPU-only, synthetic-data, single-topic-file
constraint of this repository, `implementation.py` below implements a
**working DCGAN** end-to-end (trained and verified) on synthetic image data,
and includes a **minimal, honestly-labeled CycleGAN skeleton** (correct loss
math, correct architecture shape, generator/discriminator pair, cycle-loss
computation) trained briefly on two synthetic shape domains (circles vs.
squares) to demonstrate the mechanism — not to claim publication-quality
translation results. Where the CycleGAN section underperforms (round-trip
error still non-trivial after limited epochs), this is reported honestly in
`explanation.md`, not hidden.

## 5. Known GAN failure modes (relevant to what you'll see in the run)

- **Mode collapse**: `G` finds one or a few outputs that reliably fool `D`
  and stops exploring — output diversity drops sharply.
- **Non-convergence / oscillation**: `D` and `G` losses can oscillate rather
  than converge to a fixed point, since this is a minimax game, not a single
  loss being minimized. A GAN "converging" doesn't mean the loss curves flatten
  the way a supervised loss does — oscillation around an equilibrium is
  normal and expected, not necessarily a bug.
- **Discriminator overpowering generator**: if `D` reaches near-perfect
  accuracy early, `G`'s gradient signal vanishes. Watch `D(real)` and
  `D(fake)` probabilities directly, not just the loss values.
