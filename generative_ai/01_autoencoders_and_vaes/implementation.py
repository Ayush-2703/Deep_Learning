"""
Topic: Autoencoders & Variational Autoencoders (VAEs)
============================================================
Repository : deep-learning/foundations/01_autoencoders_and_vaes/
File       : implementation.py
Framework  : PyTorch 2.x | NumPy | scikit-learn | matplotlib
Python     : 3.10+

Implementation: Vanilla Autoencoder + VAE on MNIST
Import Library & Configuration
SECTION 1: DATA LOADING — MNIST with normalization to [0, 1]
SECTION 2: VANILLA AUTOENCODER
SECTION 3: VARIATIONAL AUTOENCODER
SECTION 4: LOSS FUNCTIONS
SECTION 5: TRAINING LOOP
SECTION 6: VISUALIZATION UTILITIES
SECTION 7: MAIN EXECUTION
SECTION 8: VISUALIZATION
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
EPOCHS_AE = 10
EPOCHS_VAE = 15
LATENT_DIM = 20          # Dimensionality of latent space
HIDDEN_DIM = 400         # Hidden layer size
LEARNING_RATE = 1e-3

print(f"Device: {DEVICE}")
print(f"Latent dim: {LATENT_DIM}, Hidden dim: {HIDDEN_DIM}")

# =============================================================================
# SECTION 1: DATA LOADING — MNIST with normalization to [0, 1]
# =============================================================================
transform = transforms.Compose([
    transforms.ToTensor(),  # [0, 255] -> [0.0, 1.0]
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# =============================================================================
# SECTION 2: VANILLA AUTOENCODER
# =============================================================================
class Autoencoder(nn.Module):
    """
    Deterministic Autoencoder.
    Encoder: 784 -> 400 -> 400 -> 20 (latent)
    Decoder: 20 -> 400 -> 400 -> 784
    """
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),   # [B, 784] -> [B, 400]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # [B, 400] -> [B, 400]
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),  # [B, 400] -> [B, 20]
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),  # [B, 20] -> [B, 400]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # [B, 400] -> [B, 400]
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),   # [B, 400] -> [B, 784]
            nn.Sigmoid(),  # Output in [0, 1] to match normalized input
        )
    
    def forward(self, x):
        # x: [B, 1, 28, 28]
        x_flat = x.view(x.size(0), -1)        # [B, 784]
        z = self.encoder(x_flat)               # [B, 20]
        x_hat = self.decoder(z)                # [B, 784]
        x_hat = x_hat.view(x.size(0), 1, 28, 28)  # [B, 1, 28, 28]
        return x_hat, z

# =============================================================================
# SECTION 3: VARIATIONAL AUTOENCODER
# =============================================================================
class VAE(nn.Module):
    """
    Variational Autoencoder with diagonal Gaussian posterior.
    Encoder outputs mu and log_var.
    Decoder reconstructs from sampled z.
    """
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder: shared backbone, then split into mu and log_var
        self.fc1 = nn.Linear(input_dim, hidden_dim)      # [B, 784] -> [B, 400]
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)     # [B, 400] -> [B, 400]
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)      # [B, 400] -> [B, 20]
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # [B, 400] -> [B, 20]
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)     # [B, 20] -> [B, 400]
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)       # [B, 400] -> [B, 400]
        self.fc5 = nn.Linear(hidden_dim, input_dim)        # [B, 400] -> [B, 784]
    
    def encode(self, x):
        """Returns mu and log_var for the posterior q(z|x)."""
        h = F.relu(self.fc1(x))        # [B, 400]
        h = F.relu(self.fc2(h))        # [B, 400]
        mu = self.fc_mu(h)             # [B, 20]
        log_var = self.fc_logvar(h)    # [B, 20]
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick:
        z = mu + sigma * epsilon, where epsilon ~ N(0, I)
        """
        std = torch.exp(0.5 * log_var)           # [B, 20]
        eps = torch.randn_like(std)              # [B, 20], sampled from N(0,1)
        z = mu + std * eps                       # [B, 20]
        return z
    
    def decode(self, z):
        """Decode latent z back to image space."""
        h = F.relu(self.fc3(z))        # [B, 400]
        h = F.relu(self.fc4(h))        # [B, 400]
        x_recon = torch.sigmoid(self.fc5(h))  # [B, 784]
        return x_recon
    
    def forward(self, x):
        # x: [B, 1, 28, 28]
        x_flat = x.view(x.size(0), -1)           # [B, 784]
        mu, log_var = self.encode(x_flat)        # [B, 20], [B, 20]
        z = self.reparameterize(mu, log_var)     # [B, 20]
        x_recon = self.decode(z)                 # [B, 784]
        x_recon = x_recon.view(x.size(0), 1, 28, 28)  # [B, 1, 28, 28]
        return x_recon, mu, log_var

# =============================================================================
# SECTION 4: LOSS FUNCTIONS
# =============================================================================
def ae_loss(recon_x, x):
    """Binary Cross-Entropy loss for Autoencoder."""
    # recon_x, x: [B, 1, 28, 28]
    recon_x = recon_x.view(recon_x.size(0), -1)   # [B, 784]
    x = x.view(x.size(0), -1)                    # [B, 784]
    loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    return loss / x.size(0)  # Average per sample

def vae_loss(recon_x, x, mu, log_var, beta=1.0):
    """
    VAE ELBO loss = Reconstruction - beta * KL(q(z|x) || p(z))
    
    Reconstruction: BCE between input and reconstruction
    KL: analytical form for diagonal Gaussian posterior and standard Gaussian prior
    """
    # recon_x, x: [B, 1, 28, 28]
    recon_x = recon_x.view(recon_x.size(0), -1)   # [B, 784]
    x = x.view(x.size(0), -1)                      # [B, 784]
    
    # Reconstruction term (negative log-likelihood)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.size(0)
    
    # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    # This is the analytical form for KL(N(mu, sigma^2) || N(0, 1))
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    kl_loss = kl_loss.mean()  # Average over batch
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

# =============================================================================
# SECTION 5: TRAINING LOOP
# =============================================================================
def train_model(model, train_loader, test_loader, loss_fn, epochs, model_name,
                optimizer=None, is_vae=False):
    """Generic training loop for AE or VAE."""
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.to(DEVICE)
    train_losses = []
    test_losses = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(DEVICE)               # [B, 1, 28, 28]
            optimizer.zero_grad()
            
            if is_vae:
                recon, mu, log_var = model(data)
                loss, recon_l, kl_l = loss_fn(recon, data, mu, log_var)
            else:
                recon, z = model(data)
                loss = loss_fn(recon, data)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(DEVICE)
                if is_vae:
                    recon, mu, log_var = model(data)
                    loss, _, _ = loss_fn(recon, data, mu, log_var)
                else:
                    recon, z = model(data)
                    loss = loss_fn(recon, data)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        if epoch % 2 == 0 or epoch == 1:
            print(f"[{model_name}] Epoch {epoch}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
    
    return train_losses, test_losses

# =============================================================================
# SECTION 6: VISUALIZATION UTILITIES
# =============================================================================
def visualize_reconstructions(model, test_loader, title, is_vae=False, n=8):
    """Display original vs reconstructed images side by side."""
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data[:n].to(DEVICE)
        if is_vae:
            recon, _, _ = model(data)
        else:
            recon, _ = model(data)
    
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    for i in range(n):
        axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)
        
        axes[1, i].imshow(recon[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)
    
    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'/mnt/agents/output/phase5_01_{title.lower().replace(" ", "_")}.png', dpi=150)
    plt.show()
    print(f"Saved reconstruction plot: {title}")

def visualize_latent_space(model, test_loader, title):
    """Visualize 2D PCA of latent space colored by digit class."""
    from sklearn.decomposition import PCA
    
    model.eval()
    latents = []
    labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(DEVICE)
            _, z = model(data)  # [B, latent_dim]
            latents.append(z.cpu())
            labels.append(target)
    
    latents = torch.cat(latents, dim=0).numpy()      # [N, latent_dim]
    labels = torch.cat(labels, dim=0).numpy()         # [N]
    
    # PCA to 2D
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], 
                          c=labels, cmap='tab10', alpha=0.5, s=5)
    plt.colorbar(scatter, label='Digit Class')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title(f'{title} — Latent Space (PCA)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'/mnt/agents/output/phase5_01_{title.lower().replace(" ", "_")}_latent.png', dpi=150)
    plt.show()
    print(f"Saved latent space plot: {title}")

def generate_from_prior(vae, n=16):
    """Generate new images by sampling z ~ N(0, I) and decoding."""
    vae.eval()
    with torch.no_grad():
        z = torch.randn(n, LATENT_DIM).to(DEVICE)   # [16, 20]
        samples = vae.decode(z)                        # [16, 784]
        samples = samples.view(n, 1, 28, 28)           # [16, 1, 28, 28]
    
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].cpu().squeeze(), cmap='gray')
        ax.axis('off')
    plt.suptitle('VAE: Samples from Prior N(0, I)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/mnt/agents/output/phase5_01_vae_generated_samples.png', dpi=150)
    plt.show()
    print("Saved generated samples plot")

def interpolate_latent(vae, test_loader, n_steps=10):
    """Linear interpolation between two random test images in latent space."""
    vae.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        x1, x2 = data[0:1].to(DEVICE), data[1:2].to(DEVICE)  # [1, 1, 28, 28]
        
        # Encode to latent
        z1_mu, _ = vae.encode(x1.view(1, -1))   # [1, 20]
        z2_mu, _ = vae.encode(x2.view(1, -1))   # [1, 20]
        
        # Interpolate
        alphas = torch.linspace(0, 1, n_steps).to(DEVICE)
        interpolations = []
        for alpha in alphas:
            z = (1 - alpha) * z1_mu + alpha * z2_mu   # [1, 20]
            x_interp = vae.decode(z)                     # [1, 784]
            interpolations.append(x_interp.view(1, 28, 28).cpu())
        
        interpolations = torch.cat(interpolations, dim=0)  # [n_steps, 28, 28]
    
    fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 1.2, 1.5))
    for i, ax in enumerate(axes):
        ax.imshow(interpolations[i].squeeze(), cmap='gray')
        ax.axis('off')
    plt.suptitle('Latent Space Interpolation', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/mnt/agents/output/phase5_01_vae_interpolation.png', dpi=150)
    plt.show()
    print("Saved interpolation plot")

# =============================================================================
# SECTION 7: MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TRAINING VANILLA AUTOENCODER")
    print("="*70)
    
    ae_model = Autoencoder(input_dim=784, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM)
    ae_optimizer = optim.Adam(ae_model.parameters(), lr=LEARNING_RATE)
    ae_train_losses, ae_test_losses = train_model(
        ae_model, train_loader, test_loader, ae_loss, EPOCHS_AE,
        "Autoencoder", optimizer=ae_optimizer, is_vae=False
    )
    
    print("\n" + "="*70)
    print("TRAINING VARIATIONAL AUTOENCODER")
    print("="*70)
    
    vae_model = VAE(input_dim=784, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM)
    vae_optimizer = optim.Adam(vae_model.parameters(), lr=LEARNING_RATE)
    vae_train_losses, vae_test_losses = train_model(
        vae_model, train_loader, test_loader, vae_loss, EPOCHS_VAE,
        "VAE", optimizer=vae_optimizer, is_vae=True
    )
    
    # =============================================================================
    # SECTION 8: VISUALIZATION
    # =============================================================================
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Reconstruction comparison
    visualize_reconstructions(ae_model, test_loader, "Autoencoder Reconstruction", is_vae=False)
    visualize_reconstructions(vae_model, test_loader, "VAE Reconstruction", is_vae=True)
    
    # Latent space visualization
    visualize_latent_space(ae_model, test_loader, "Autoencoder")
    visualize_latent_space(vae_model, test_loader, "VAE")
    
    # VAE-specific: generation and interpolation
    generate_from_prior(vae_model, n=16)
    interpolate_latent(vae_model, test_loader, n_steps=10)
    
    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(ae_train_losses, label='Train', marker='o')
    axes[0].plot(ae_test_losses, label='Test', marker='s')
    axes[0].set_title('Autoencoder Training Curves')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('BCE Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(vae_train_losses, label='Train', marker='o')
    axes[1].plot(vae_test_losses, label='Test', marker='s')
    axes[1].set_title('VAE Training Curves')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('ELBO Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/agents/output/phase5_01_training_curves.png', dpi=150)
    plt.show()
    print("Saved training curves")
    
    # =============================================================================
    # SECTION 9: FINAL STATISTICS
    # =============================================================================
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Autoencoder final test loss: {ae_test_losses[-1]:.4f}")
    print(f"VAE final test loss:         {vae_test_losses[-1]:.4f}")
    print(f"AE parameters: {sum(p.numel() for p in ae_model.parameters()):,}")
    print(f"VAE parameters: {sum(p.numel() for p in vae_model.parameters()):,}")
    
    # Compute reconstruction MSE on test set
    ae_model.eval()
    vae_model.eval()
    ae_mse_total = 0.0
    vae_mse_total = 0.0
    n_samples = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(DEVICE)
            n = data.size(0)
            
            ae_recon, _ = ae_model(data)
            ae_mse = F.mse_loss(ae_recon, data, reduction='sum').item()
            ae_mse_total += ae_mse
            
            vae_recon, _, _ = vae_model(data)
            vae_mse = F.mse_loss(vae_recon, data, reduction='sum').item()
            vae_mse_total += vae_mse
            
            n_samples += n
    
    print(f"\nTest MSE — Autoencoder: {ae_mse_total / n_samples:.6f}")
    print(f"Test MSE — VAE:         {vae_mse_total / n_samples:.6f}")
    print("\n" + "="*70)
    print("ALL DONE ✓")
    print("="*70)
