**Explanation: Autoencoders & VAEs**

## Table of Contents
1. [Imports & Setup](#1-imports--setup)
2. [Vanilla Autoencoder Architecture](#2-vanilla-autoencoder-architecture)
3. [VAE Architecture](#3-vae-architecture)
4. [Loss Functions](#4-loss-functions)
5. [Training Loop](#5-training-loop)
6. [Visualization Utilities](#6-visualization-utilities)
7. [Main Execution Flow](#7-main-execution-flow)
8. [Common Pitfalls & Debugging](#8-common-pitfalls--debugging)

---

## 1. Imports & Setup

```python
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Why**: Automatically selects GPU if available. The `.to(DEVICE)` call moves tensors to the correct hardware. Always use this pattern — never hardcode `'cuda'` or `'cpu'`.

```python
transform = transforms.Compose([
    transforms.ToTensor(),  # [0, 255] -> [0.0, 1.0]
])
```

**Why `ToTensor()`**: MNIST images are PIL Images with pixel values 0-255. `ToTensor()` scales to [0.0, 1.0] and converts to `torch.FloatTensor`. This matches our decoder's `Sigmoid` output range. Without normalization, BCE loss would be numerically unstable.

```python
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
```

**Why `num_workers=2`**: Loads data in background processes, preventing the GPU from idling while the CPU reads from disk. Set to 0 only for debugging (prevents multiprocessing errors).

**Why `shuffle=True` for train**: Prevents the model from learning spurious order-based correlations. Never shuffle the test set — evaluation must be deterministic.

---

## 2. Vanilla Autoencoder Architecture

```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
```

**Why `latent_dim=20`**: MNIST is relatively simple (10 digit classes). 20 dimensions provide enough capacity for reconstruction while forcing compression. For more complex datasets (CIFAR-10, CelebA), typical values are 64-512.

```python
self.encoder = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, latent_dim),
)
```

**Why two hidden layers**: A single hidden layer is often sufficient for MNIST, but two layers with 400 units each provide a better capacity/parameter trade-off. The "hourglass" shape (784 → 400 → 400 → 20) is standard.

**Why no activation on the final encoder layer**: The latent space in a vanilla AE is unbounded (can be any real number). ReLU would clip negative values, losing information. The decoder can handle any real input.

```python
self.decoder = nn.Sequential(
    ...
    nn.Sigmoid(),  # Output in [0, 1]
)
```

**Why `Sigmoid` on decoder output**: Our input pixels are normalized to [0, 1]. The sigmoid squashes outputs to the same range, making BCE loss mathematically valid. Without it, the decoder could output values outside [0, 1], causing BCE to produce `NaN` (log of negative numbers).

```python
def forward(self, x):
    x_flat = x.view(x.size(0), -1)        # [B, 784]
```

**Why `.view(x.size(0), -1)`**: MNIST images are `[B, 1, 28, 28]`. We flatten to `[B, 784]` for the linear layers. `x.size(0)` preserves batch size; `-1` infers the remaining dimension (1×28×28 = 784).

**Tensor shape trace through AE**:
```
Input:        [B, 1, 28, 28]
Flatten:      [B, 784]
Encoder L1:   [B, 400]   (ReLU)
Encoder L2:   [B, 400]   (ReLU)
Latent z:     [B, 20]
Decoder L1:   [B, 400]   (ReLU)
Decoder L2:   [B, 400]   (ReLU)
Decoder L3:   [B, 784]   (Sigmoid)
Reshape:      [B, 1, 28, 28]
```

---

## 3. VAE Architecture

### Encoder Split into μ and log(σ²)

```python
self.fc_mu = nn.Linear(hidden_dim, latent_dim)
self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
```

**Why two separate heads**: The encoder must output the parameters of a Gaussian distribution: mean (μ) and variance (σ²). Two linear layers branch from the shared backbone.

**Why `log_var` instead of `var` or `std`**:
- `var` must be positive → requires softplus/ReLU constraint, harder to optimize
- `std` must be positive → same issue
- `log_var` is unbounded → no constraint needed, optimizer can freely explore negative values (small variance) and positive values (large variance)
- `std = exp(0.5 * log_var)` is always positive

```python
def encode(self, x):
    h = F.relu(self.fc1(x))
    h = F.relu(self.fc2(h))
    mu = self.fc_mu(h)
    log_var = self.fc_logvar(h)
    return mu, log_var
```

**Why `F.relu` instead of `nn.ReLU()`**: `F.relu` is the functional form. Since these are inside custom methods (not `nn.Sequential`), functional activations are cleaner. Equivalent to `nn.ReLU()` in forward pass.

### The Reparameterization Trick

```python
def reparameterize(self, mu, log_var):
    std = torch.exp(0.5 * log_var)     # [B, 20]
    eps = torch.randn_like(std)      # [B, 20]
    z = mu + std * eps               # [B, 20]
    return z
```

**Line-by-line breakdown**:

1. `std = torch.exp(0.5 * log_var)`
   - `log_var` is the log of variance: `log_var = log(σ²)`
   - `0.5 * log_var = log(σ)`
   - `exp(0.5 * log_var) = σ` (standard deviation)
   - We use std (not var) because the Gaussian sampling formula uses std: `z = μ + σ·ε`

2. `eps = torch.randn_like(std)`
   - Samples from standard normal: `ε ~ N(0, I)`
   - `randn_like` creates a tensor with same shape and device as `std`
   - **Critical**: This tensor is treated as a constant during backprop. The stochasticity is "external" to the computation graph.

3. `z = mu + std * eps`
   - Algebraically: `z ~ N(μ, σ²I)` because `std * ε ~ N(0, σ²I)` and `μ + N(0, σ²I) = N(μ, σ²I)`
   - Gradients flow through `mu` and `std` (learnable parameters)
   - `eps` has `requires_grad=False`, so gradients don't flow into the sampling operation

**Why this works for backprop**:
```
Without reparameterization:  z ← Sample(N(μ, σ²))  [stochastic node, gradient blocked]
With reparameterization:     z = μ + σ·ε           [deterministic w.r.t. μ, σ]
```

### Decoder

```python
def decode(self, z):
    h = F.relu(self.fc3(z))
    h = F.relu(self.fc4(h))
    x_recon = torch.sigmoid(self.fc5(h))
    return x_recon
```

**Same as AE decoder**: The decoder doesn't care whether `z` came from deterministic encoding or probabilistic sampling. It just maps latent vectors to pixel probabilities.

---

## 4. Loss Functions

### AE Loss

```python
def ae_loss(recon_x, x):
    recon_x = recon_x.view(recon_x.size(0), -1)   # [B, 784]
    x = x.view(x.size(0), -1)                      # [B, 784]
    loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    return loss / x.size(0)
```

**Why `reduction='sum'` then divide by batch size**:
- `sum` aggregates loss over all pixels in the batch
- Dividing by `x.size(0)` (batch size) gives **per-sample average**
- Alternative: `reduction='mean'` would average over all pixels AND all samples, making loss scale inversely with batch size. Our approach keeps loss scale independent of batch size.

**Why BCE and not MSE for MNIST**:
- MNIST pixels are essentially binary (ink or no ink)
- BCE is the negative log-likelihood of a Bernoulli distribution
- MSE assumes Gaussian noise, which is less appropriate for binary-ish data
- In practice, both work; BCE is theoretically preferred

### VAE Loss (ELBO)

```python
def vae_loss(recon_x, x, mu, log_var, beta=1.0):
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)
    
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.size(0)
    
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    kl_loss = kl_loss.mean()
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss
```

#### Reconstruction Term
Same as AE: measures how well the decoder reconstructs the input from a sampled `z`.

#### KL Divergence Term
```python
kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
```

**Mathematical derivation**:
For diagonal Gaussian posterior `q(z|x) = N(μ, diag(σ²))` and prior `p(z) = N(0, I)`:

$$D_{KL}(q \| p) = \frac{1}{2} \sum_{j=1}^{k} \left( \sigma_j^2 + \mu_j^2 - 1 - \log(\sigma_j^2) \right)$$

In code:
- `log_var.exp()` = `exp(log(σ²))` = `σ²`
- `mu.pow(2)` = `μ²`
- `1` = the constant term
- `log_var` = `log(σ²)`

So: `1 + log_var - mu.pow(2) - log_var.exp()` = `1 + log(σ²) - μ² - σ²`

Multiply by `-0.5`: `-0.5 * (1 + log(σ²) - μ² - σ²)` = `0.5 * (σ² + μ² - 1 - log(σ²))` ✓

**Why `dim=1` then `.mean()`**:
- `dim=1` sums over the latent dimension (k=20), giving one KL value per sample: `[B]`
- `.mean()` averages over the batch
- This is equivalent to `reduction='sum'` on recon_loss then dividing by batch — consistent scaling

**Why `beta=1.0`**:
- Standard VAE uses β=1 (balanced reconstruction and regularization)
- β>1 pushes latent space closer to standard normal → better disentanglement but worse reconstruction
- β<1 prioritizes reconstruction → less structured latent space

---

## 5. Training Loop

```python
def train_model(model, train_loader, test_loader, loss_fn, epochs, model_name,
                optimizer=None, is_vae=False):
```

**Why generic function**: Both AE and VAE share the same training structure. The `is_vae` flag handles the different forward pass signatures.

```python
model.to(DEVICE)
```

**Why before the loop**: Moves model parameters to GPU once. Subsequent operations automatically use the same device. Must be done before creating the optimizer (otherwise optimizer tracks CPU parameters).

```python
for batch_idx, (data, _) in enumerate(train_loader):
    data = data.to(DEVICE)
    optimizer.zero_grad()
```

**Why `zero_grad()` before forward pass**: PyTorch accumulates gradients by default (`.backward()` adds to existing `.grad`). If you don't zero, gradients from the previous batch contaminate the current batch. This is useful for gradient accumulation but must be explicitly managed.

```python
if is_vae:
    recon, mu, log_var = model(data)
    loss, recon_l, kl_l = loss_fn(recon, data, mu, log_var)
else:
    recon, z = model(data)
    loss = loss_fn(recon, data)

loss.backward()
optimizer.step()
```

**Shape trace for VAE forward**:
```
data:         [B, 1, 28, 28]
view:         [B, 784]
encode:       mu=[B, 20], log_var=[B, 20]
reparameterize: z=[B, 20]
decode:       [B, 784]
view:         [B, 1, 28, 28]
```

**Why `.backward()` then `.step()`**:
1. `loss.backward()`: Computes gradients via autograd (reverse-mode automatic differentiation)
2. `optimizer.step()`: Updates parameters using computed gradients

### Validation Block

```python
model.eval()
with torch.no_grad():
```

**Why `model.eval()`**: Sets dropout and batch normalization layers to evaluation mode. For our simple networks (no dropout/BN), this is technically a no-op, but it's mandatory good practice.

**Why `torch.no_grad()`**: Disables gradient computation for the validation pass. Saves memory and speeds up inference by not building the computation graph. Always wrap evaluation in this context manager.

---

## 6. Visualization Utilities

### Reconstruction Visualization

```python
def visualize_reconstructions(model, test_loader, title, is_vae=False, n=8):
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data[:n].to(DEVICE)
```

**Why `next(iter(test_loader))`**: Gets the first batch. Since test_loader has `shuffle=False`, this is deterministic (first 128 images of the test set).

**Why `model.eval()` + `torch.no_grad()`**: Even for visualization, we don't need gradients. `model.eval()` is especially important for VAE — in eval mode, some implementations use the mean (μ) directly instead of sampling. Our VAE always samples during forward, but `eval()` is still good practice.

### Latent Space Visualization (PCA)

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
latents_2d = pca.fit_transform(latents)
```

**Why PCA and not t-SNE/UMAP**: PCA is deterministic, fast, and preserves global structure. t-SNE is better for local clusters but non-deterministic and slower. For a 20D → 2D projection, PCA is sufficient to show whether digits cluster.

**Expected result**: VAE latent space should show more overlap between classes (because KL pushes toward N(0,I)), while AE may show tighter clusters but with gaps.

### Generation from Prior

```python
z = torch.randn(n, LATENT_DIM).to(DEVICE)
samples = vae.decode(z)
```

**Why this works**: After training, the encoder maps inputs to posteriors close to N(0,I). The KL term ensures this. Therefore, sampling `z ~ N(0,I)` and decoding should produce valid digit-like images. This is the **generative capability** of VAEs that AEs lack.

**Why AE can't do this**: An AE's latent space has no probabilistic structure. Sampling random points might decode to garbage because the latent space has "holes" (regions the encoder never maps to).

### Latent Interpolation

```python
z = (1 - alpha) * z1_mu + alpha * z2_mu
```

**Why linear interpolation**: In a well-trained VAE, the latent space is smooth and continuous. Linearly interpolating between two latent codes should produce a smooth visual transition between the two digits. This is a key test of latent space quality.

**Why use `mu` (not sampled `z`) for interpolation**: The mean is deterministic. Using sampled `z` would add noise to the interpolation path, making it less smooth.

---

## 7. Main Execution Flow

```python
if __name__ == "__main__":
```

**Why this guard**: Prevents code from running when the module is imported. Essential for reusable scripts.

**Execution order**:
1. Train vanilla AE (10 epochs) — establishes baseline
2. Train VAE (15 epochs) — needs more epochs due to KL term complexity
3. Generate all visualizations
4. Compute final MSE statistics

**Why train AE first**: Provides a baseline for comparison. The VAE should have slightly worse reconstruction (due to information bottleneck of sampling) but better generative capabilities.

---

## 8. Common Pitfalls & Debugging

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Forgetting `zero_grad()` | Loss increases, training unstable | Call before every forward pass |
| Missing `.to(DEVICE)` | `RuntimeError: expected device cuda but got cpu` | Move model AND data to same device |
| Sigmoid on encoder output | Latent values clamped to [0,1], poor reconstruction | Remove final activation in encoder |
| Using `reduction='mean'` on BCE | Loss scales with batch size, LR needs tuning | Use `sum` then divide by batch size |
| Forgetting `torch.no_grad()` in eval | Memory leak, slower inference | Always wrap evaluation |
| `log_var` initialized to 0 | Initial std = 1, may be too large | Xavier init handles this automatically |
| β too high (>10) | Reconstruction becomes blurry | Reduce β or increase model capacity |
| β too low (<0.1) | Latent space doesn't match prior | Increase β |
| Not flattening before Linear | `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | Always `view(B, -1)` before Linear layers |
| Using MSE on [0,1] data | Works but theoretically suboptimal | Use BCE for binary/continuous-in-[0,1] data |

### Debugging Checklist
1. Print tensor shapes at each layer: `print(x.shape)`
2. Verify loss decreases in first few epochs
3. Check KL loss is not NaN (can happen if `log_var` explodes)
4. Visualize reconstructions after 1 epoch — should already be vaguely digit-shaped
5. If reconstructions are all gray: check that input is in [0,1] and sigmoid is applied
