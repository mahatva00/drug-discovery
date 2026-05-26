"""
generate.py
-----------
Generates novel drug candidate molecules using the trained GNN-VAE.

Steps:
    1. Load trained model from checkpoint
    2. Sample random vectors from latent space
    3. Decode into molecular properties (logP, qed, SAS)
    4. Denormalize properties back to real scale
    5. Filter candidates by drug-likeness rules (Lipinski)
    6. Match to nearest real ZINC molecule via property similarity
    7. Save results to CSV
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.vae import MolecularVAE


# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "./checkpoints/best_model.pt"
CSV_PATH        = "./data/zinc_raw/raw/zinc250k.csv"
OUTPUT_PATH     = "./checkpoints/generated_molecules.csv"


# ── Lipinski Drug-likeness Filter ─────────────────────────────────────────────
def lipinski_filter(logP, qed, SAS):
    """
    Filters molecules based on drug-likeness rules:
        - logP  <= 5.0   (lipophilicity — absorption)
        - qed   >= 0.5   (drug-likeness score)
        - SAS   <= 6.0   (synthetic accessibility — ease of synthesis)

    Returns True if molecule passes all filters.
    """
    return (
        logP <= 5.0 and
        logP >= -2.0 and
        qed  >= 0.5 and
        SAS  <= 6.0
    )


# ── Main generation function ──────────────────────────────────────────────────
def generate_molecules(
    num_samples   = 10000,
    top_k         = 20,
    save_results  = True
):
    """
    Generates novel drug candidate molecules.

    Args:
        num_samples  (int) : Number of latent vectors to sample
        top_k        (int) : Number of top candidates to return
        save_results (bool): Save results to CSV

    Returns:
        DataFrame of top drug candidates
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Using device : {device}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print("\n  Loading trained model from checkpoint...")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}\n"
            f"Please run train.py first!"
        )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    config     = checkpoint["config"]

    print(f"  Checkpoint from epoch : {checkpoint['epoch']}")
    print(f"  Best val loss         : {checkpoint['val_loss']:.4f}")

    # ── Rebuild model ─────────────────────────────────────────────────────────
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
    print("  Model loaded successfully!")

    # ── Rebuild scaler ────────────────────────────────────────────────────────
    scaler         = StandardScaler()
    scaler.mean_   = np.array(checkpoint["scaler_mean"])
    scaler.scale_  = np.array(checkpoint["scaler_scale"])
    scaler.var_    = scaler.scale_ ** 2
    scaler.n_features_in_ = 3

    # ── Sample from latent space ──────────────────────────────────────────────
    print(f"\n  Sampling {num_samples} molecules from latent space...")

    with torch.no_grad():
        # Sample z ~ N(0, I)
        z = torch.randn(num_samples, config["latent_dim"]).to(device)

        # Decode z → normalized properties
        props_normalized = model.decode(z).cpu().numpy()

    # ── Denormalize properties ────────────────────────────────────────────────
    props_real = scaler.inverse_transform(props_normalized)

    logP_vals = props_real[:, 0]
    qed_vals  = props_real[:, 1]
    SAS_vals  = props_real[:, 2]

    print(f"  Generated property ranges:")
    print(f"    logP : {logP_vals.min():.2f} to {logP_vals.max():.2f}")
    print(f"    qed  : {qed_vals.min():.2f}  to {qed_vals.max():.2f}")
    print(f"    SAS  : {SAS_vals.min():.2f}  to {SAS_vals.max():.2f}")

    # ── Apply Lipinski filter ─────────────────────────────────────────────────
    print(f"\n  Applying Lipinski drug-likeness filter...")

    candidates = []
    for i in range(num_samples):
        logP = float(logP_vals[i])
        qed  = float(qed_vals[i])
        SAS  = float(SAS_vals[i])

        if lipinski_filter(logP, qed, SAS):
            candidates.append({
                "sample_id"       : i,
                "logP_generated"  : round(logP, 4),
                "qed_generated"   : round(qed,  4),
                "SAS_generated"   : round(SAS,  4),
                "z_vector"        : z[i].cpu().numpy()
            })

    print(f"  Passed filter : {len(candidates)} / {num_samples} molecules")

    if len(candidates) == 0:
        print("  No molecules passed the filter! Try increasing num_samples.")
        return None

    # ── Load real ZINC molecules for nearest neighbor matching ────────────────
    print(f"\n  Loading ZINC database for nearest neighbor matching...")
    df_zinc = pd.read_csv(CSV_PATH)

    zinc_props = df_zinc[["logP", "qed", "SAS"]].values.astype(np.float32)
    zinc_smiles = df_zinc["smiles"].tolist()

    # ── Match each candidate to nearest real molecule ─────────────────────────
    print(f"  Finding nearest real molecules for {len(candidates)} candidates...")

    results = []
    for cand in candidates:
        gen_props = np.array([[
            cand["logP_generated"],
            cand["qed_generated"],
            cand["SAS_generated"]
        ]])

        # Compute euclidean distance to all ZINC molecules
        dists   = euclidean_distances(gen_props, zinc_props)[0]
        nearest = int(np.argmin(dists))

        results.append({
            "sample_id"      : cand["sample_id"],
            "smiles"         : zinc_smiles[nearest],
            "logP_generated" : cand["logP_generated"],
            "qed_generated"  : cand["qed_generated"],
            "SAS_generated"  : cand["SAS_generated"],
            "logP_real"      : round(float(zinc_props[nearest, 0]), 4),
            "qed_real"       : round(float(zinc_props[nearest, 1]), 4),
            "SAS_real"       : round(float(zinc_props[nearest, 2]), 4),
            "distance"       : round(float(dists[nearest]), 4),
        })

    # ── Sort by qed (higher = more drug-like) and distance (lower = closer) ──
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(
        by        = ["qed_generated", "distance"],
        ascending = [False, True]
    ).reset_index(drop=True)

    # ── Take top K ────────────────────────────────────────────────────────────
    df_top = df_results.head(top_k)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  TOP {top_k} GENERATED DRUG CANDIDATES")
    print(f"{'='*65}")
    print(f"  {'#':<4} {'SMILES':<35} {'logP':>6} {'qed':>6} {'SAS':>6} {'Dist':>6}")
    print(f"  {'-'*65}")

    for i, row in df_top.iterrows():
        smiles_short = row["smiles"][:33] + ".." if len(row["smiles"]) > 33 else row["smiles"]
        print(
            f"  {i+1:<4} {smiles_short:<35} "
            f"{row['logP_generated']:>6.2f} "
            f"{row['qed_generated']:>6.3f} "
            f"{row['SAS_generated']:>6.2f} "
            f"{row['distance']:>6.3f}"
        )

    print(f"{'='*65}")
    print(f"\n  Column guide:")
    print(f"    logP : Lipophilicity     (ideal: -2 to 5)")
    print(f"    qed  : Drug-likeness     (ideal: > 0.5, max=1.0)")
    print(f"    SAS  : Synthetic access  (ideal: < 6.0, lower=easier)")
    print(f"    Dist : Property distance (lower = closer to real molecule)")

    # ── Save results ──────────────────────────────────────────────────────────
    if save_results:
        df_top.drop(columns=[], inplace=False).to_csv(OUTPUT_PATH, index=False)
        print(f"\n  Results saved to: {OUTPUT_PATH}")

    return df_top


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = generate_molecules(
        num_samples  = 10000,
        top_k        = 20,
        save_results = True
    )

    if df is not None:
        print(f"\n  Generation complete! {len(df)} candidates ready for visualization.")