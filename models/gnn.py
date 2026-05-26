"""
gnn.py
------
Graph Neural Network (GNN) Encoder for molecular graphs.

Architecture:
    - 3 x GIN (Graph Isomorphism Network) convolution layers
    - Batch Normalization after each layer
    - Global Mean Pooling → graph-level embedding
    - Linear projection → outputs mean (mu) and log variance (log_var)
      which are used by the VAE

Input:
    - x          : node features  [num_atoms, 14]
    - edge_index : COO format     [2, num_bonds*2]
    - edge_attr  : edge features  [num_bonds*2, 5]
    - batch      : batch vector   [num_atoms]

Output:
    - mu      : mean of latent distribution        [batch_size, latent_dim]
    - log_var : log variance of latent distribution [batch_size, latent_dim]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, BatchNorm


# ── GNN Encoder ───────────────────────────────────────────────────────────────
class GNNEncoder(nn.Module):
    """
    GNN Encoder that converts a molecular graph into a latent vector.

    Args:
        node_feat_dim (int) : Number of input node features (14 from dataset)
        edge_feat_dim (int) : Number of input edge features (5 from dataset)
        hidden_dim    (int) : Hidden layer size (default: 256)
        latent_dim    (int) : Size of latent space vector (default: 128)
        num_layers    (int) : Number of GIN conv layers (default: 3)
        dropout       (float): Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        node_feat_dim = 14,
        edge_feat_dim = 5,
        hidden_dim    = 256,
        latent_dim    = 128,
        num_layers    = 3,
        dropout       = 0.1
    ):
        super(GNNEncoder, self).__init__()

        self.hidden_dim  = hidden_dim
        self.latent_dim  = latent_dim
        self.num_layers  = num_layers
        self.dropout     = dropout

        # ── Input projection (node features → hidden_dim) ─────────────────────
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        # ── GIN Convolution layers ────────────────────────────────────────────
        # GIN uses a small MLP inside each convolution
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        for _ in range(num_layers):
            # MLP inside GIN conv
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(BatchNorm(hidden_dim))

        # ── Output projection → mu and log_var ───────────────────────────────
        self.fc_mu      = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

        # ── Dropout ───────────────────────────────────────────────────────────
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x, edge_index, batch):
        """
        Forward pass of the GNN Encoder.

        Args:
            x          : node features  [num_atoms, node_feat_dim]
            edge_index : COO format     [2, num_edges]
            batch      : batch vector   [num_atoms]

        Returns:
            mu      : [batch_size, latent_dim]
            log_var : [batch_size, latent_dim]
        """

        # ── Step 1: Project input node features to hidden dim ─────────────────
        x = F.relu(self.input_proj(x))

        # ── Step 2: Apply GIN conv layers ─────────────────────────────────────
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)   # message passing
            x = bn(x)                 # batch normalization
            x = F.relu(x)             # activation
            x = self.drop(x)          # dropout

        # ── Step 3: Global pooling → one vector per molecule ──────────────────
        # Aggregates all atom embeddings into a single graph embedding
        graph_embed = global_mean_pool(x, batch)  # [batch_size, hidden_dim]

        # ── Step 4: Project to latent space (mu and log_var) ──────────────────
        mu      = self.fc_mu(graph_embed)       # [batch_size, latent_dim]
        log_var = self.fc_log_var(graph_embed)  # [batch_size, latent_dim]

        return mu, log_var


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from data.dataset import load_zinc_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Using device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    train_loader, _, _ = load_zinc_dataset(max_molecules=500)

    # ── Build model ───────────────────────────────────────────────────────────
    model = GNNEncoder(
        node_feat_dim = 14,
        edge_feat_dim = 5,
        hidden_dim    = 256,
        latent_dim    = 128,
        num_layers    = 3,
        dropout       = 0.1
    ).to(device)

    print("\n  GNN Encoder Architecture:")
    print(model)

    # ── Count parameters ──────────────────────────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total trainable parameters: {total_params:,}")

    # ── Forward pass test ─────────────────────────────────────────────────────
    for batch in train_loader:
        batch = batch.to(device)

        mu, log_var = model(batch.x, batch.edge_index, batch.batch)

        print(f"\n  Input  node features shape : {batch.x.shape}")
        print(f"  Output mu shape            : {mu.shape}")
        print(f"  Output log_var shape       : {log_var.shape}")
        print(f"\n  mu sample values      : {mu[0][:5]}")
        print(f"  log_var sample values : {log_var[0][:5]}")
        break

    print("\n  gnn.py working correctly!")