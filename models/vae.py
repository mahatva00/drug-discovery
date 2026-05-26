"""
vae.py
------
Variational Autoencoder (VAE) for molecular generation.

Architecture:
    Encoder : GNNEncoder (from gnn.py)
              → outputs mu and log_var

    Reparameterization :
              z = mu + eps * std   (eps ~ N(0,1))

    Decoder : MLP that takes z and reconstructs
              molecular properties (logP, qed, SAS)

Loss:
    Total Loss = Reconstruction Loss + KL Divergence
    - Reconstruction : MSE between predicted and actual properties
    - KL Divergence  : pushes latent space towards N(0,1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gnn import GNNEncoder


# ── Decoder ───────────────────────────────────────────────────────────────────
class MolecularDecoder(nn.Module):
    """
    MLP Decoder that takes a latent vector z and reconstructs
    molecular properties.

    Args:
        latent_dim   (int) : Size of latent vector (default: 128)
        hidden_dim   (int) : Hidden layer size     (default: 256)
        output_dim   (int) : Number of properties to reconstruct
                             logP, qed, SAS = 3    (default: 3)
        dropout      (float): Dropout rate          (default: 0.1)
    """

    def __init__(
        self,
        latent_dim  = 128,
        hidden_dim  = 256,
        output_dim  = 3,
        dropout     = 0.1
    ):
        super(MolecularDecoder, self).__init__()

        self.decoder = nn.Sequential(

            # Layer 1
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            # Layer 2
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            # Layer 3
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),

            # Output layer → molecular properties
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, z):
        """
        Args:
            z : latent vector [batch_size, latent_dim]
        Returns:
            reconstructed properties [batch_size, output_dim]
        """
        return self.decoder(z)


# ── Full VAE Model ────────────────────────────────────────────────────────────
class MolecularVAE(nn.Module):
    """
    Full Molecular VAE combining:
        - GNNEncoder  : graph → mu, log_var
        - Reparameterization trick : mu, log_var → z
        - MolecularDecoder : z → molecular properties

    Args:
        node_feat_dim (int)  : Node feature dimension  (default: 14)
        edge_feat_dim (int)  : Edge feature dimension  (default: 5)
        hidden_dim    (int)  : Hidden layer size        (default: 256)
        latent_dim    (int)  : Latent space dimension   (default: 128)
        output_dim    (int)  : Number of properties     (default: 3)
        num_layers    (int)  : GNN layers               (default: 3)
        dropout       (float): Dropout rate             (default: 0.1)
        beta          (float): KL weight (beta-VAE)     (default: 1.0)
    """

    def __init__(
        self,
        node_feat_dim = 14,
        edge_feat_dim = 5,
        hidden_dim    = 256,
        latent_dim    = 128,
        output_dim    = 3,
        num_layers    = 3,
        dropout       = 0.1,
        beta          = 1.0
    ):
        super(MolecularVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta       = beta

        # ── Encoder (GNN) ─────────────────────────────────────────────────────
        self.encoder = GNNEncoder(
            node_feat_dim = node_feat_dim,
            edge_feat_dim = edge_feat_dim,
            hidden_dim    = hidden_dim,
            latent_dim    = latent_dim,
            num_layers    = num_layers,
            dropout       = dropout
        )

        # ── Decoder (MLP) ─────────────────────────────────────────────────────
        self.decoder = MolecularDecoder(
            latent_dim  = latent_dim,
            hidden_dim  = hidden_dim,
            output_dim  = output_dim,
            dropout     = dropout
        )

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick:
            z = mu + eps * std
            eps ~ N(0, 1)

        This allows gradients to flow through the sampling step.

        Args:
            mu      : [batch_size, latent_dim]
            log_var : [batch_size, latent_dim]
        Returns:
            z       : [batch_size, latent_dim]
        """
        if self.training:
            std = torch.exp(0.5 * log_var)   # std = e^(0.5 * log_var)
            eps = torch.randn_like(std)       # eps ~ N(0, 1)
            return mu + eps * std
        else:
            # During inference, just use the mean
            return mu

    def forward(self, x, edge_index, batch):
        """
        Full forward pass:
            graph → encode → reparameterize → decode → properties

        Args:
            x          : node features  [num_atoms, node_feat_dim]
            edge_index : COO format     [2, num_edges]
            batch      : batch vector   [num_atoms]

        Returns:
            recon      : reconstructed properties [batch_size, output_dim]
            mu         : latent mean              [batch_size, latent_dim]
            log_var    : latent log variance      [batch_size, latent_dim]
            z          : sampled latent vector    [batch_size, latent_dim]
        """

        # ── Encode ────────────────────────────────────────────────────────────
        mu, log_var = self.encoder(x, edge_index, batch)

        # ── Reparameterize ────────────────────────────────────────────────────
        z = self.reparameterize(mu, log_var)

        # ── Decode ────────────────────────────────────────────────────────────
        recon = self.decoder(z)

        return recon, mu, log_var, z

    def encode(self, x, edge_index, batch):
        """Encode only — returns mu and log_var"""
        return self.encoder(x, edge_index, batch)

    def decode(self, z):
        """Decode only — takes z, returns properties"""
        return self.decoder(z)

    def sample(self, num_samples, device):
        """
        Generate new molecules by sampling from the latent space.

        Args:
            num_samples (int)    : Number of molecules to generate
            device               : torch device

        Returns:
            generated properties [num_samples, output_dim]
            z                    [num_samples, latent_dim]
        """
        # Sample z from standard normal distribution
        z = torch.randn(num_samples, self.latent_dim).to(device)

        self.eval()
        with torch.no_grad():
            properties = self.decode(z)

        return properties, z


# ── VAE Loss Function ─────────────────────────────────────────────────────────
def vae_loss(recon, target, mu, log_var, beta=1.0):
    """
    VAE Loss = Reconstruction Loss + beta * KL Divergence

    Args:
        recon   : reconstructed properties [batch_size, output_dim]
        target  : actual properties        [batch_size, output_dim]
        mu      : latent mean              [batch_size, latent_dim]
        log_var : latent log variance      [batch_size, latent_dim]
        beta    : KL weight (default 1.0)

    Returns:
        total_loss  : scalar
        recon_loss  : scalar
        kl_loss     : scalar
    """

    # ── Reconstruction loss (MSE) ─────────────────────────────────────────────
    recon_loss = F.mse_loss(recon, target, reduction='mean')

    # ── KL Divergence ─────────────────────────────────────────────────────────
    # KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kl_loss = -0.5 * torch.mean(
        1 + log_var - mu.pow(2) - log_var.exp()
    )

    # ── Total loss ────────────────────────────────────────────────────────────
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data.dataset import load_zinc_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Using device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    train_loader, _, _ = load_zinc_dataset(max_molecules=500)

    # ── Build model ───────────────────────────────────────────────────────────
    model = MolecularVAE(
        node_feat_dim = 14,
        edge_feat_dim = 5,
        hidden_dim    = 256,
        latent_dim    = 128,
        output_dim    = 3,
        num_layers    = 3,
        dropout       = 0.1,
        beta          = 1.0
    ).to(device)

    print("\n  MolecularVAE Architecture:")
    print(model)

    # ── Count parameters ──────────────────────────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total trainable parameters: {total_params:,}")

    # ── Forward pass test ─────────────────────────────────────────────────────
    for batch in train_loader:
        batch = batch.to(device)

        # Create dummy target properties [batch_size, 3] (logP, qed, SAS)
        target = torch.randn(batch.num_graphs, 3).to(device)

        # Forward pass
        recon, mu, log_var, z = model(batch.x, batch.edge_index, batch.batch)

        # Compute loss
        total_loss, recon_loss, kl_loss = vae_loss(recon, target, mu, log_var)

        print(f"\n  Input  node features : {batch.x.shape}")
        print(f"  Output recon shape   : {recon.shape}")
        print(f"  Output mu shape      : {mu.shape}")
        print(f"  Output z shape       : {z.shape}")
        print(f"\n  Total Loss : {total_loss.item():.4f}")
        print(f"  Recon Loss : {recon_loss.item():.4f}")
        print(f"  KL Loss    : {kl_loss.item():.4f}")
        break

    # ── Sampling test ─────────────────────────────────────────────────────────
    print("\n  Testing molecule generation (sampling)...")
    props, z_samples = model.sample(num_samples=5, device=device)
    print(f"  Generated properties shape : {props.shape}")
    print(f"  Sample z shape             : {z_samples.shape}")
    print(f"  Sample properties (logP, qed, SAS):\n{props}")

    print("\n  vae.py working correctly!")