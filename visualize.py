"""
visualize.py
------------
Visualization module for the Molecular GNN-VAE project.

Generates 4 types of visualizations:
    1. Molecule Grid       - 2D structures of top drug candidates
    2. Latent Space        - t-SNE plot of latent space (colored by qed)
    3. Training Curves     - Loss history plot
    4. Property Dashboard  - logP, qed, SAS distributions
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.vae import MolecularVAE


# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH   = "./checkpoints/best_model.pt"
GENERATED_CSV     = "./checkpoints/generated_molecules.csv"
LOSS_CSV          = "./checkpoints/loss_history.csv"
CSV_PATH          = "./data/zinc_raw/raw/zinc250k.csv"
OUTPUT_DIR        = "./visualizations"

# ── Colors ────────────────────────────────────────────────────────────────────
BG_COLOR     = "#0f1117"
PANEL_COLOR  = "#1a1d2e"
ACCENT1      = "#7c3aed"
ACCENT2      = "#06b6d4"
ACCENT3      = "#10b981"
ACCENT4      = "#f59e0b"
TEXT_COLOR   = "#e2e8f0"
GRID_COLOR   = "#2d3748"


# ── 1. Molecule Grid ──────────────────────────────────────────────────────────
def draw_molecule_grid(df_candidates, n_cols=4, n_rows=5, save=True):
    """
    Draws a grid of 2D molecular structures for the top drug candidates.
    Each molecule shows atoms, bonds, and key properties.
    """
    print("\n  [1/4] Drawing molecule grid...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_mols  = min(n_cols * n_rows, len(df_candidates))
    fig     = plt.figure(figsize=(n_cols * 4, n_rows * 3.5))
    fig.patch.set_facecolor(BG_COLOR)

    # Title
    fig.suptitle(
        "Generated Drug Candidate Molecules\nGNN-VAE | ZINC Dataset",
        fontsize   = 18,
        fontweight = "bold",
        color      = TEXT_COLOR,
        y          = 0.98
    )

    mol_images = []

    for _, row in df_candidates.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is not None:
            mol_images.append((mol, row))
        if len(mol_images) >= n_mols:
            break

    for idx, (mol, row) in enumerate(mol_images):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        ax.set_facecolor(PANEL_COLOR)

        # ── FIX: use only valid drawOptions attributes ─────────────────────────
        drawer = rdMolDraw2D.MolDraw2DCairo(300, 200)
        opts = drawer.drawOptions()
        opts.addStereoAnnotation = True
        opts.backgroundColour    = (0.1, 0.11, 0.17, 1)
        # NOTE: atomLabelFontSize does NOT exist in most RDKit versions.
        # Font size is controlled via the drawer size (300x200 above).

        AllChem.Compute2DCoords(mol)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()

        # Convert to PIL image
        bio = io.BytesIO(drawer.GetDrawingText())
        img = Image.open(bio)
        ax.imshow(img)
        ax.axis("off")

        # Property label
        label = (
            f"logP={row['logP_generated']:.2f}  "
            f"qed={row['qed_generated']:.3f}  "
            f"SAS={row['SAS_generated']:.2f}"
        )
        ax.set_title(
            label,
            fontsize   = 7,
            color      = ACCENT2,
            pad        = 3,
            fontfamily = "monospace"
        )

        # Rank badge
        ax.text(
            0.04, 0.96, f"#{idx+1}",
            transform  = ax.transAxes,
            fontsize   = 9,
            fontweight = "bold",
            color      = ACCENT4,
            va         = "top"
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        path = os.path.join(OUTPUT_DIR, "1_molecule_grid.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        print(f"     Saved → {path}")

    plt.close()


# ── 2. Latent Space t-SNE ─────────────────────────────────────────────────────
def plot_latent_space(num_molecules=5000, save=True):
    """
    Encodes molecules into latent space and visualizes using t-SNE.
    Points are colored by qed (drug-likeness score).
    """
    print("\n  [2/4] Plotting latent space (t-SNE)...")

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    config     = checkpoint["config"]

    # Rebuild model
    model = MolecularVAE(
        node_feat_dim = config["node_feat_dim"],
        edge_feat_dim = config["edge_feat_dim"],
        hidden_dim    = config["hidden_dim"],
        latent_dim    = config["latent_dim"],
        output_dim    = config["output_dim"],
        num_layers    = config["num_layers"],
        dropout       = config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Load dataset
    from data.dataset import load_zinc_dataset
    train_loader, _, _, scaler = load_zinc_dataset(max_molecules=num_molecules)

    # Encode molecules → latent vectors
    all_z   = []
    all_qed = []

    with torch.no_grad():
        for batch in train_loader:
            batch  = batch.to(device)
            mu, _  = model.encode(batch.x, batch.edge_index, batch.batch)
            all_z.append(mu.cpu().numpy())

            # Extract qed from batch targets and denormalize
            props    = batch.y.view(batch.num_graphs, -1).cpu().numpy()
            qed_norm = props[:, 1]
            qed_real = qed_norm * scaler.scale_[1] + scaler.mean_[1]
            all_qed.extend(qed_real.tolist())

    z_matrix = np.vstack(all_z)
    qed_arr  = np.array(all_qed[:len(z_matrix)])

    # t-SNE reduction
    print(f"     Running t-SNE on {len(z_matrix)} molecules...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
    z_2d = tsne.fit_transform(z_matrix)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)

    sc = ax.scatter(
        z_2d[:, 0], z_2d[:, 1],
        c          = qed_arr,
        cmap       = "plasma",
        alpha      = 0.7,
        s          = 18,
        edgecolors = "none"
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("QED (Drug-likeness)", color=TEXT_COLOR, fontsize=11)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
    cbar.outline.set_edgecolor(GRID_COLOR)

    ax.set_title(
        "Latent Space Visualization (t-SNE)\nColored by Drug-likeness (QED)",
        fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=15
    )
    ax.set_xlabel("t-SNE Dimension 1", color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", color=TEXT_COLOR, fontsize=11)
    ax.tick_params(colors=TEXT_COLOR)
    ax.spines[:].set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, alpha=0.4, linewidth=0.5)

    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "2_latent_space_tsne.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        print(f"     Saved → {path}")

    plt.close()


# ── 3. Training Loss Curves ───────────────────────────────────────────────────
def plot_training_curves(save=True):
    """
    Plots the training and validation loss curves from the loss history CSV.
    """
    print("\n  [3/4] Plotting training curves...")

    if not os.path.exists(LOSS_CSV):
        print(f"     Loss CSV not found at {LOSS_CSV}. Skipping.")
        return

    df = pd.read_csv(LOSS_CSV)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(
        "Training History — GNN-VAE\nMolecular Drug Discovery",
        fontsize=14, fontweight="bold", color=TEXT_COLOR, y=1.02
    )

    plots = [
        ("train_total", "val_total", "Total Loss",          ACCENT1,   ACCENT2),
        ("train_recon", "val_recon", "Reconstruction Loss", ACCENT3,   ACCENT4),
        ("train_kl",    "val_kl",    "KL Divergence Loss",  "#f43f5e", "#fb923c"),
    ]

    for ax, (train_col, val_col, title, c1, c2) in zip(axes, plots):
        ax.set_facecolor(PANEL_COLOR)

        ax.plot(df["epoch"], df[train_col],
                color=c1, linewidth=2, label="Train", alpha=0.9)
        ax.plot(df["epoch"], df[val_col],
                color=c2, linewidth=2, label="Val",
                linestyle="--", alpha=0.9)

        # Best epoch marker
        best_idx = df[val_col].idxmin()
        ax.axvline(
            df["epoch"][best_idx],
            color="white", linewidth=1,
            linestyle=":", alpha=0.5,
            label=f"Best epoch {df['epoch'][best_idx]}"
        )

        ax.set_title(title, color=TEXT_COLOR, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", color=TEXT_COLOR, fontsize=10)
        ax.set_ylabel("Loss", color=TEXT_COLOR, fontsize=10)
        ax.tick_params(colors=TEXT_COLOR)
        ax.spines[:].set_color(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, alpha=0.4, linewidth=0.5)

        legend = ax.legend(fontsize=9, facecolor=BG_COLOR, labelcolor=TEXT_COLOR)
        legend.get_frame().set_edgecolor(GRID_COLOR)

    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "3_training_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        print(f"     Saved → {path}")

    plt.close()


# ── 4. Property Distribution Dashboard ───────────────────────────────────────
def plot_property_dashboard(save=True):
    """
    Plots the distribution of logP, qed, SAS for:
        - Real ZINC molecules (blue)
        - Generated molecules (orange)
    Side by side for comparison.
    """
    print("\n  [4/4] Plotting property distributions...")

    if not os.path.exists(GENERATED_CSV):
        print(f"     Generated CSV not found. Run generate.py first.")
        return

    df_real = pd.read_csv(CSV_PATH).head(250000)
    df_gen  = pd.read_csv(GENERATED_CSV)

    props = [
        ("logP", "logP_generated", "logP (Lipophilicity)",    "(-2 to 5 ideal)"),
        ("qed",  "qed_generated",  "QED (Drug-likeness)",     "(>0.5 ideal)"),
        ("SAS",  "SAS_generated",  "SAS (Ease of Synthesis)", "(<6.0 ideal)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(
        "Molecular Property Distributions\nReal ZINC vs GNN-VAE Generated",
        fontsize=14, fontweight="bold", color=TEXT_COLOR, y=1.02
    )

    for ax, (real_col, gen_col, title, subtitle) in zip(axes, props):
        ax.set_facecolor(PANEL_COLOR)

        real_vals = df_real[real_col].dropna().values
        gen_vals  = df_gen[gen_col].dropna().values

        ax.hist(real_vals, bins=50, alpha=0.6,
                color=ACCENT2, label="Real ZINC",
                density=True, edgecolor="none")

        ax.hist(gen_vals, bins=30, alpha=0.8,
                color=ACCENT4, label="Generated",
                density=True, edgecolor="none")

        ax.set_title(
            f"{title}\n{subtitle}",
            color=TEXT_COLOR, fontsize=11, fontweight="bold"
        )
        ax.set_xlabel(real_col, color=TEXT_COLOR, fontsize=10)
        ax.set_ylabel("Density", color=TEXT_COLOR, fontsize=10)
        ax.tick_params(colors=TEXT_COLOR)
        ax.spines[:].set_color(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, alpha=0.4, linewidth=0.5)

        legend = ax.legend(fontsize=9, facecolor=BG_COLOR, labelcolor=TEXT_COLOR)
        legend.get_frame().set_edgecolor(GRID_COLOR)

        # Stats annotation
        stats = (
            f"Real  μ={real_vals.mean():.2f} σ={real_vals.std():.2f}\n"
            f"Gen   μ={gen_vals.mean():.2f} σ={gen_vals.std():.2f}"
        )
        ax.text(
            0.97, 0.97, stats,
            transform  = ax.transAxes,
            fontsize   = 8,
            color      = TEXT_COLOR,
            va         = "top",
            ha         = "right",
            fontfamily = "monospace",
            bbox       = dict(boxstyle="round", facecolor=BG_COLOR, alpha=0.7)
        )

    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "4_property_distributions.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        print(f"     Saved → {path}")

    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def run_all_visualizations():
    """
    Runs all 4 visualizations in sequence.
    """
    print("\n" + "=" * 55)
    print("  VISUALIZATION PIPELINE")
    print("  GNN-VAE Molecular Drug Discovery")
    print("=" * 55)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(GENERATED_CSV):
        print("\n  ERROR: Run generate.py first to get candidates!")
        return

    df_candidates = pd.read_csv(GENERATED_CSV)

    draw_molecule_grid(df_candidates)
    plot_latent_space(num_molecules=5000)
    plot_training_curves()
    plot_property_dashboard()

    print("\n" + "=" * 55)
    print("  All visualizations saved to ./visualizations/")
    print("  Files:")
    print("    1_molecule_grid.png            — 2D molecular structures")
    print("    2_latent_space_tsne.png        — t-SNE latent space")
    print("    3_training_curves.png          — Loss history")
    print("    4_property_distributions.png   — Property comparison")
    print("=" * 55)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_all_visualizations()