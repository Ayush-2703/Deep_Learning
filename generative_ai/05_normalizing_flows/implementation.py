"""
Phase 5 - Topic 4: Normalizing Flows (RealNVP-style affine coupling layers)
CPU-only, synthetic 2D toy data (two-moons), PyTorch.

Run: python3 implementation.py
Produces: outputs/training_loss.png, outputs/samples_comparison.png,
          outputs/latent_space_z.png, outputs/density_heatmap.png
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(2)
np.random.seed(2)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# 1. Synthetic two-moons data (same generator as Topic 3, for direct comparison)
# ---------------------------------------------------------------------------
def make_two_moons(n_samples=2000, noise=0.06):
    n1 = n_samples // 2
    n2 = n_samples - n1
    theta1 = np.random.uniform(0, np.pi, n1)
    moon1_x = np.cos(theta1); moon1_y = np.sin(theta1)
    theta2 = np.random.uniform(0, np.pi, n2)
    moon2_x = 1 - np.cos(theta2); moon2_y = 1 - np.sin(theta2) - 0.5
    X = np.concatenate([np.stack([moon1_x, moon1_y], axis=1),
                         np.stack([moon2_x, moon2_y], axis=1)], axis=0)
    X += np.random.normal(0, noise, X.shape)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    np.random.shuffle(X)
    return X.astype(np.float32)

data_np = make_two_moons(2000)
data = torch.tensor(data_np)
print(f"Two-moons synthetic dataset: {data.shape}")

n_train = int(0.85 * len(data))
train_data, val_data = data[:n_train], data[n_train:]

# ---------------------------------------------------------------------------
# 2. RealNVP-style affine coupling layer
# ---------------------------------------------------------------------------
class CouplingLayer(nn.Module):
    """Splits 2D input via a binary mask; transforms the masked-out half
    conditioned on the passed-through half. mask=[1,0] or [0,1] alternates
    which dimension is transformed."""
    def __init__(self, mask, hidden=64):
        super().__init__()
        self.register_buffer("mask", mask)
        self.scale_net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),
        )
        self.translate_net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, z):
        """z -> x (used for sampling). Returns x, log_det_jacobian."""
        z_masked = z * self.mask
        s = self.scale_net(z_masked) * (1 - self.mask)
        t = self.translate_net(z_masked) * (1 - self.mask)
        s = torch.tanh(s)  # clamp scale for numerical stability (theory.md section 7)
        x = z_masked + (1 - self.mask) * (z * torch.exp(s) + t)
        log_det = s.sum(dim=-1)
        return x, log_det

    def inverse(self, x):
        """x -> z (used for computing exact likelihood of real data)."""
        x_masked = x * self.mask
        s = self.scale_net(x_masked) * (1 - self.mask)
        t = self.translate_net(x_masked) * (1 - self.mask)
        s = torch.tanh(s)
        z = x_masked + (1 - self.mask) * ((x - t) * torch.exp(-s))
        log_det = -s.sum(dim=-1)
        return z, log_det


class RealNVP(nn.Module):
    def __init__(self, n_layers=6, hidden=64):
        super().__init__()
        masks = [torch.tensor([1.0, 0.0]) if i % 2 == 0 else torch.tensor([0.0, 1.0])
                  for i in range(n_layers)]
        self.layers = nn.ModuleList([CouplingLayer(m, hidden) for m in masks])
        self.base_dist = D.MultivariateNormal(torch.zeros(2), torch.eye(2))

    def forward(self, z):
        """Sampling direction: z (base noise) -> x (data space)."""
        log_det_total = torch.zeros(z.size(0))
        x = z
        for layer in self.layers:
            x, log_det = layer.forward(x)
            log_det_total += log_det
        return x, log_det_total

    def inverse(self, x):
        """Density-evaluation direction: x (data) -> z (base space)."""
        log_det_total = torch.zeros(x.size(0))
        z = x
        for layer in reversed(self.layers):
            z, log_det = layer.inverse(z)
            log_det_total += log_det
        return z, log_det_total

    def log_prob(self, x):
        z, log_det = self.inverse(x)
        return self.base_dist.log_prob(z) + log_det

    def sample(self, n):
        z = self.base_dist.sample((n,))
        x, _ = self.forward(z)
        return x, z


model = RealNVP(n_layers=6, hidden=64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------------------------------------------------------
# 3. Training: exact maximum likelihood (minimize negative log-likelihood)
# ---------------------------------------------------------------------------
EPOCHS = 400
BATCH = 256
n = train_data.size(0)
history = {"train_nll": [], "val_nll": []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(n)
    epoch_losses = []
    for i in range(0, n - BATCH, BATCH):
        xb = train_data[perm[i:i + BATCH]]
        log_p = model.log_prob(xb)
        loss = -log_p.mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        epoch_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_nll = -model.log_prob(val_data).mean().item()
    history["train_nll"].append(np.mean(epoch_losses))
    history["val_nll"].append(val_nll)

    if epoch % 40 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | train NLL={history['train_nll'][-1]:.3f} | val NLL={val_nll:.3f}")

final_train = history["train_nll"][-1]
final_val = history["val_nll"][-1]
gap = final_val - final_train
print(f"\nFinal train NLL: {final_train:.3f} | Final val NLL: {final_val:.3f} | gap: {gap:.3f}")
if gap > 0.2 * abs(final_train):
    print("NOTE: val NLL noticeably worse than train NLL -> mild overfitting given small dataset/model.")
else:
    print("NOTE: train/val NLL gap is small -> no significant overfitting detected.")

plt.figure(figsize=(7, 4))
plt.plot(history["train_nll"], label="train NLL")
plt.plot(history["val_nll"], label="val NLL")
plt.title("RealNVP Training: Exact Negative Log-Likelihood")
plt.xlabel("epoch"); plt.ylabel("NLL (nats)"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_loss.png"), dpi=110)
plt.close()

# ---------------------------------------------------------------------------
# 4. Sampling: single forward pass, z ~ N(0,I) -> x (no iterative denoising)
# ---------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    gen_samples, z_used = model.sample(1000)

gen_np = gen_samples.numpy()
real_np = data_np

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].scatter(real_np[:1000, 0], real_np[:1000, 1], s=5, alpha=0.5)
axes[0].set_title("Real data (two-moons)")
axes[0].set_xlim(-4, 4); axes[0].set_ylim(-4, 4)
axes[1].scatter(gen_np[:, 0], gen_np[:, 1], s=5, alpha=0.5, color="green")
axes[1].set_title("RealNVP-generated samples (single forward pass)")
axes[1].set_xlim(-4, 4); axes[1].set_ylim(-4, 4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "samples_comparison.png"), dpi=110)
plt.close()

real_mean, real_std = real_np.mean(axis=0), real_np.std(axis=0)
gen_mean, gen_std = gen_np.mean(axis=0), gen_np.std(axis=0)
print(f"\nReal data      mean={real_mean.round(3)}, std={real_std.round(3)}")
print(f"Generated data mean={gen_mean.round(3)}, std={gen_std.round(3)}")

# ---------------------------------------------------------------------------
# 5. Latent space check: does inverse(real_data) actually look Gaussian?
# ---------------------------------------------------------------------------
with torch.no_grad():
    z_from_real, _ = model.inverse(data)
z_np = z_from_real.numpy()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].scatter(z_np[:, 0], z_np[:, 1], s=5, alpha=0.5, color="purple")
axes[0].set_title("f^-1(real data) -- should look ~N(0,I) if model fit well")
axes[0].set_xlim(-4, 4); axes[0].set_ylim(-4, 4)
ref_z = torch.randn(1000, 2).numpy()
axes[1].scatter(ref_z[:, 0], ref_z[:, 1], s=5, alpha=0.5, color="gray")
axes[1].set_title("Reference: true N(0,I) samples")
axes[1].set_xlim(-4, 4); axes[1].set_ylim(-4, 4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "latent_space_z.png"), dpi=110)
plt.close()

z_mean, z_std = z_np.mean(axis=0), z_np.std(axis=0)
print(f"f^-1(real data) mean={z_mean.round(3)}, std={z_std.round(3)} (target: mean=0, std=1)")
if np.abs(z_mean).max() < 0.3 and np.abs(z_std - 1).max() < 0.3:
    print("NOTE: inverse-mapped real data closely matches the N(0,I) base distribution -> good fit.")
else:
    print("NOTE: inverse-mapped real data deviates from N(0,I) -> imperfect fit, reported honestly.")

# ---------------------------------------------------------------------------
# 6. Density heatmap: exact log p(x) evaluated on a grid (unique to flows --
#    GANs (Topic 2) and this DDPM (Topic 3) cannot produce this directly)
# ---------------------------------------------------------------------------
grid_size = 100
xx, yy = np.meshgrid(np.linspace(-3, 3, grid_size), np.linspace(-3, 3, grid_size))
grid_points = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32)
with torch.no_grad():
    log_probs = model.log_prob(grid_points).numpy().reshape(grid_size, grid_size)

plt.figure(figsize=(6, 5))
plt.pcolormesh(xx, yy, np.exp(log_probs), shading="auto", cmap="viridis")
plt.colorbar(label="p(x) (exact density)")
plt.scatter(real_np[:300, 0], real_np[:300, 1], s=3, color="white", alpha=0.4)
plt.title("Exact learned density p(x) -- a capability unique to normalizing flows")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "density_heatmap.png"), dpi=110)
plt.close()

print("\nSaved: training_loss.png, samples_comparison.png, latent_space_z.png, density_heatmap.png")
print("Topic 4 (Normalizing Flows) run complete.")
