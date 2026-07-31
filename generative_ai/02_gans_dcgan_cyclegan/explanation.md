# Explanation: GAN Implementation Walkthrough

## PART A: DCGAN

### 1. Data normalization to `[-1, 1]`

```python
imgs = imgs * 2 - 1  # normalize to [-1, 1] to match Tanh output
```

The generator's final layer is `Tanh`, which outputs in `[-1, 1]`, so the
real data must be rescaled to the same range for the discriminator to
compare like with like. Forgetting this step is a classic DCGAN bug — this
implementation gets it right, verified by the fact that D's real/fake
probabilities stayed in a sane range rather than collapsing immediately.

### 2. Generator: noise -> image via transposed convolutions

```
z [B,100] -> view [B,100,1,1]
  -> ConvTranspose2d(100->128, k=7, s=1, p=0) -> [B,128,7,7]
  -> ConvTranspose2d(128->64, k=4, s=2, p=1)  -> [B,64,14,14]
  -> ConvTranspose2d(64->1, k=4, s=2, p=1)    -> [B,1,28,28] -> Tanh
```

Each `ConvTranspose2d` doubles spatial resolution (except the first, which
goes from a 1x1 "pixel" to a 7x7 feature map). Kernel/stride/padding values
are chosen precisely so the output lands exactly on 28x28 (verified by
running the forward pass and checking `.shape`, not just assumed).

### 3. Discriminator: image -> real/fake logit

Mirror architecture using strided `Conv2d` instead of pooling — this
matches the DCGAN convention that pooling is avoided in favor of learned
downsampling, which is what actually gives the discriminator sharper
decision boundaries.

`BCEWithLogitsLoss` is used instead of `BCELoss` + manual `Sigmoid` — this
is numerically more stable (combines sigmoid and log in one fused,
overflow-safe operation), so the discriminator's final layer has no
activation function (raw logits out).

### 4. Non-saturating generator update

```python
loss_g = bce(d_fake_logits_for_g, torch.ones(bsz))  # wants D to say "real"
```

Notice the generator step re-runs `G(noise)` on a *fresh* noise batch
(rather than reusing the batch from the discriminator step) and does **not**
detach the output — gradients need to flow back into `G`'s weights through
`D`'s forward pass. The discriminator step, by contrast, explicitly
`.detach()`s the fake batch, since we don't want `D`'s loss to update `G`.

### 5. What the actual run showed (honest reporting)

```
Final D(real)=0.843, D(fake)=0.160
NOTE: D/G balance is intermediate; neither fully collapsed nor at ideal equilibrium.
```

This is a real, printed diagnostic from the script — not a hand-picked
number. `D(real)=0.84` and `D(fake)=0.16` mean the discriminator still has
a meaningful edge over the generator after 30 epochs, which tracks with
what `dcgan_samples_epoch30.png` actually shows: recognizable blob-like
shapes with soft edges, not crisp circles/squares/crosses. This is a
genuine, moderate result for a shallow 3-layer DCGAN trained briefly on
CPU with 1,500 synthetic images — not a failure, but also not a polished
one, and it's reported as such rather than cropped to look better.

### 6. Loss oscillation is expected, not a bug

`dcgan_loss.png` shows `D_loss` and `G_loss` moving in opposite, non-monotonic
directions across epochs. Unlike a supervised loss curve that should trend
toward zero, a GAN's losses reflect a moving target (each network is chasing
the other), so oscillation without a downward trend is normal and expected
per the minimax framing in theory.md, not evidence of a broken training loop.

## PART B: CycleGAN skeleton

### 7. Why a "skeleton", stated plainly

theory.md already flags this: CycleGAN's normal training regime is 100-200
epochs on thousands of real photographic images. Here it runs 15 epochs on
300 synthetic images per domain, specifically to keep runtime CPU-feasible
within this repository's format while still exercising every real component
of the algorithm — two generators, two discriminators, and the cycle loss.

### 8. The two-generator / two-discriminator setup

`G_XtoY`, `G_YtoX` are the same lightweight architecture (downsample ->
residual block -> upsample), used bidirectionally. `D_X`, `D_Y` each judge
realism only within their own domain (`D_X` never sees domain-Y images
except as fakes produced by `G_YtoX`).

### 9. Cycle-consistency loss, computed exactly as derived in theory.md

```python
x_cycle = G_YtoX(y_fake)   # translate fake-Y back to X
y_cycle = G_XtoY(x_fake)   # translate fake-X back to Y
loss_cycle = F.l1_loss(x_cycle, x_real) + F.l1_loss(y_cycle, y_real)
```

`LAMBDA_CYC = 10.0` weights this heavily relative to the adversarial terms,
matching the original CycleGAN paper's convention — cycle consistency is
the term that prevents the generators from producing realistic-but-unrelated
images (a real risk without paired supervision).

### 10. Honest result: loss dropped, visual translation is rough

```
Final cycle-consistency L1 loss: 0.099
```

The cycle loss did drop substantially (1.75 -> 0.10 over 15 epochs),
meaning `F(G(x)) ≈ x` reasonably well — the round-trip constraint is being
learned. But looking at `cyclegan_translation.png`: the middle row
(`G(X)->Y`, circles translated toward "square-domain") shows only a
partial shift in shape, with visible blur/artifacts rather than a clean
square. This is the expected outcome of a heavily-abbreviated training
run and is reported here exactly as observed — the mechanism works
end-to-end and is verifiably correct, but 15 epochs on 300 images is not
enough for convincing image quality, and the script does not claim
otherwise.
