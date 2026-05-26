"""
train.py
--------
Training loop for the Molecular GNN-VAE.

Features:
    - GPU accelerated training
    - Full ZINC 250k dataset (250,000 molecules)
    - Real molecular property targets (logP, qed, SAS)
    - Train / Validation loss tracking
    - Best model checkpoint saving
    - Mid-training checkpoint saving every 5 epochs (crash protection)
    - KL annealing to stabilize training
    - Loss history saved to CSV for plotting
"""

import os
import sys
import torch
import torch.optim as optim
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset import load_zinc_dataset
from models.vae import MolecularVAE, vae_loss


# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    # Data
    "max_molecules" : 250000,      # Full ZINC250k dataset
    "batch_size"    : 128,         # Increased from 32 for faster GPU utilization

    # Model
    "node_feat_dim" : 14,
    "edge_feat_dim" : 5,
    "hidden_dim"    : 256,
    "latent_dim"    : 128,
    "output_dim"    : 3,           # logP, qed, SAS
    "num_layers"    : 3,
    "dropout"       : 0.2,

    # Training
    "epochs"        : 50,
    "learning_rate" : 1e-3,
    "weight_decay"  : 1e-5,
    "beta_start"    : 0.0,
    "beta_end"      : 1.0,
    "beta_warmup"   : 15,

    # Saving
    "checkpoint_dir"      : "./checkpoints",
    "loss_log_path"       : "./checkpoints/loss_history.csv",
    "mid_save_every"      : 5,     # Save a mid-training checkpoint every N epochs
}


# ── KL Annealing ──────────────────────────────────────────────────────────────
def get_beta(epoch, beta_start, beta_end, warmup_epochs):
    """
    Linearly anneals beta from beta_start to beta_end over warmup_epochs.
    After warmup, beta stays at beta_end.
    This prevents the KL term from dominating too early in training.
    """
    if epoch >= warmup_epochs:
        return beta_end
    return beta_start + (beta_end - beta_start) * (epoch / warmup_epochs)


# ── Training step ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device, beta):
    """
    Runs one full epoch of training.
    Returns average total, reconstruction, and KL losses.
    """
    model.train()

    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum    = 0.0
    num_batches    = 0

    for batch in tqdm(loader, desc="  Training", leave=False):
        batch = batch.to(device)

        # Real targets from dataset
        target = batch.y.view(batch.num_graphs, -1)   # [batch_size, 3]

        optimizer.zero_grad()
        recon, mu, log_var, z = model(batch.x, batch.edge_index, batch.batch)

        loss, recon_l, kl_l = vae_loss(recon, target, mu, log_var, beta=beta)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss_sum += loss.item()
        recon_loss_sum += recon_l.item()
        kl_loss_sum    += kl_l.item()
        num_batches    += 1

    return (
        total_loss_sum / num_batches,
        recon_loss_sum / num_batches,
        kl_loss_sum    / num_batches
    )


# ── Validation step ───────────────────────────────────────────────────────────
def validate(model, loader, device, beta):
    """
    Runs one full epoch of validation (no gradients).
    Returns average total, reconstruction, and KL losses.
    """
    model.eval()

    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum    = 0.0
    num_batches    = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Validating", leave=False):
            batch  = batch.to(device)
            target = batch.y.view(batch.num_graphs, -1)

            recon, mu, log_var, z = model(batch.x, batch.edge_index, batch.batch)
            loss, recon_l, kl_l   = vae_loss(recon, target, mu, log_var, beta=beta)

            total_loss_sum += loss.item()
            recon_loss_sum += recon_l.item()
            kl_loss_sum    += kl_l.item()
            num_batches    += 1

    return (
        total_loss_sum / num_batches,
        recon_loss_sum / num_batches,
        kl_loss_sum    / num_batches
    )


# ── Save checkpoint helper ────────────────────────────────────────────────────
def save_checkpoint(path, epoch, model, optimizer, val_loss, scaler):
    """
    Saves a full checkpoint to disk including model weights,
    optimizer state, config, and scaler parameters.
    """
    torch.save(
        {
            "epoch"        : epoch,
            "model_state"  : model.state_dict(),
            "optim_state"  : optimizer.state_dict(),
            "val_loss"     : val_loss,
            "config"       : CONFIG,
            "scaler_mean"  : scaler.mean_.tolist(),
            "scaler_scale" : scaler.scale_.tolist(),
        },
        path
    )


# ── Main Training Function ────────────────────────────────────────────────────
def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Using device : {device}")
    if device.type == "cuda":
        print(f"  GPU          : {torch.cuda.get_device_name(0)}")

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    train_loader, val_loader, _, scaler = load_zinc_dataset(
        max_molecules = CONFIG["max_molecules"],
        batch_size    = CONFIG["batch_size"]
    )

    # ── Build model ───────────────────────────────────────────────────────────
    model = MolecularVAE(
        node_feat_dim = CONFIG["node_feat_dim"],
        edge_feat_dim = CONFIG["edge_feat_dim"],
        hidden_dim    = CONFIG["hidden_dim"],
        latent_dim    = CONFIG["latent_dim"],
        output_dim    = CONFIG["output_dim"],
        num_layers    = CONFIG["num_layers"],
        dropout       = CONFIG["dropout"],
        beta          = CONFIG["beta_end"]
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params : {total_params:,}\n")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = optim.Adam(
        model.parameters(),
        lr           = CONFIG["learning_rate"],
        weight_decay = CONFIG["weight_decay"]
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        factor   = 0.5,
        patience = 5,
        verbose  = True
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    loss_history  = []

    print("=" * 65)
    print("  Starting Training on FULL ZINC 250k Dataset...")
    print(f"  Molecules    : {CONFIG['max_molecules']:,}")
    print(f"  Batch size   : {CONFIG['batch_size']}")
    print(f"  Epochs       : {CONFIG['epochs']}")
    print(f"  Mid-save every {CONFIG['mid_save_every']} epochs → ./checkpoints/mid_checkpoint.pt")
    print("=" * 65)

    for epoch in range(1, CONFIG["epochs"] + 1):

        beta = get_beta(
            epoch,
            CONFIG["beta_start"],
            CONFIG["beta_end"],
            CONFIG["beta_warmup"]
        )

        train_total, train_recon, train_kl = train_one_epoch(
            model, train_loader, optimizer, device, beta
        )

        val_total, val_recon, val_kl = validate(
            model, val_loader, device, beta
        )

        scheduler.step(val_total)

        loss_history.append({
            "epoch"       : epoch,
            "beta"        : round(beta, 4),
            "train_total" : round(train_total, 6),
            "train_recon" : round(train_recon, 6),
            "train_kl"    : round(train_kl, 6),
            "val_total"   : round(val_total, 6),
            "val_recon"   : round(val_recon, 6),
            "val_kl"      : round(val_kl, 6),
        })

        print(
            f"  Epoch [{epoch:02d}/{CONFIG['epochs']}] "
            f"| Beta: {beta:.2f} "
            f"| Train Loss: {train_total:.4f} "
            f"(R:{train_recon:.4f} KL:{train_kl:.4f}) "
            f"| Val Loss: {val_total:.4f} "
            f"(R:{val_recon:.4f} KL:{val_kl:.4f})"
        )

        # ── Save best model ───────────────────────────────────────────────────
        if val_total < best_val_loss:
            best_val_loss = val_total
            save_checkpoint(
                path      = os.path.join(CONFIG["checkpoint_dir"], "best_model.pt"),
                epoch     = epoch,
                model     = model,
                optimizer = optimizer,
                val_loss  = best_val_loss,
                scaler    = scaler
            )
            print(f"  ✓ Best model saved! (val_loss={best_val_loss:.4f})")

        # ── Mid-training checkpoint every N epochs (crash protection) ─────────
        if epoch % CONFIG["mid_save_every"] == 0:
            mid_path = os.path.join(
                CONFIG["checkpoint_dir"],
                f"mid_checkpoint_epoch{epoch:02d}.pt"
            )
            save_checkpoint(
                path      = mid_path,
                epoch     = epoch,
                model     = model,
                optimizer = optimizer,
                val_loss  = val_total,
                scaler    = scaler
            )
            print(f"  ↓ Mid checkpoint saved → {mid_path}")

        # ── Save loss history after every epoch (so you can monitor live) ─────
        df = pd.DataFrame(loss_history)
        df.to_csv(CONFIG["loss_log_path"], index=False)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n  Loss history saved to {CONFIG['loss_log_path']}")
    print("\n" + "=" * 65)
    print(f"  Training complete!")
    print(f"  Best validation loss : {best_val_loss:.4f}")
    print(f"  Best model saved to  : {CONFIG['checkpoint_dir']}/best_model.pt")
    print(f"  Mid checkpoints at   : {CONFIG['checkpoint_dir']}/mid_checkpoint_epochXX.pt")
    print("=" * 65)

    return model, scaler


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()