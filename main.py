"""
main.py
-------
Entry point for the Molecular Drug Discovery GNN-VAE pipeline.

Real-World Use Case:
    A company wants to develop a better or cheaper version of an existing drug.
    They input the drug name → the model finds similar, novel, drug-like molecules
    → outputs the most feasible candidates as a clean PNG with structures,
    chemical formulas, and property scores.

Steps:
    1. User inputs a drug name (e.g. "Aspirin")
    2. PubChem API fetches the drug's SMILES string (two-step: name → CID → properties)
    3. SMILES is converted to a molecular graph
    4. Trained GNN-VAE encodes the graph → latent vector (mu)
    5. New molecules are sampled from a neighborhood around mu
    6. Decoded → properties → denormalized → Lipinski filtered
    7. Top 3 candidates matched to real ZINC molecules
    8. Saved as a clean PNG

Usage:
    python main.py
    python main.py --drug "Ibuprofen" --samples 500 --top 3 --noise 0.5
"""

import os
import sys
import argparse
import requests
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.vae import MolecularVAE
from data.dataset import smiles_to_graph


# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "./checkpoints/best_model.pt"
CSV_PATH        = "./data/zinc_raw/raw/zinc250k.csv"
OUTPUT_DIR      = "./visualizations"

# ── Colors ────────────────────────────────────────────────────────────────────
BG_COLOR    = "#0f1117"
PANEL_COLOR = "#1a1d2e"
ACCENT1     = "#7c3aed"
ACCENT2     = "#06b6d4"
ACCENT3     = "#10b981"
ACCENT4     = "#f59e0b"
TEXT_COLOR  = "#e2e8f0"
GRID_COLOR  = "#2d3748"
RED_COLOR   = "#f43f5e"


# ── Step 1: Fetch SMILES from PubChem (two-step) ──────────────────────────────
def fetch_smiles_from_pubchem(drug_name):
    """
    Fetches the SMILES string for a drug from PubChem API.
    Uses a two-step approach: name → CID → properties.
    Tries all known SMILES key names PubChem may return.

    Args:
        drug_name (str): Name of the drug (e.g. "Aspirin", "Ibuprofen")

    Returns:
        smiles  (str): SMILES string
        iupac   (str): IUPAC name
        formula (str): Molecular formula
        cid     (int): PubChem Compound ID
    """
    print(f"\n  Fetching '{drug_name}' from PubChem API...")

    # ── Step 1a: Get CID from drug name ───────────────────────────────────────
    cid_url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{requests.utils.quote(drug_name)}/cids/JSON"
    )

    try:
        r = requests.get(cid_url, timeout=15)
        r.raise_for_status()
        cid = r.json()["IdentifierList"]["CID"][0]
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Could not connect to PubChem. Please check your internet connection."
        )
    except Exception:
        raise ValueError(
            f"Drug '{drug_name}' not found on PubChem.\n"
            f"Try: Aspirin, Ibuprofen, Paracetamol, Metformin, Caffeine, Penicillin"
        )

    # ── Step 1b: Get properties using CID ─────────────────────────────────────
    # Request all possible SMILES variants PubChem may return
    prop_url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}"
        f"/property/CanonicalSMILES,IsomericSMILES,ConnectivitySMILES,"
        f"IUPACName,MolecularFormula,MolecularWeight/JSON"
    )

    try:
        r = requests.get(prop_url, timeout=15)
        r.raise_for_status()
        props = r.json()["PropertyTable"]["Properties"][0]
    except Exception:
        raise ValueError(
            f"Could not fetch properties for '{drug_name}' (CID={cid}) from PubChem."
        )

    # Try all known SMILES key names in order of preference
    smiles = (
        props.get("CanonicalSMILES")
        or props.get("IsomericSMILES")
        or props.get("ConnectivitySMILES")
    )

    iupac   = props.get("IUPACName", drug_name)
    formula = props.get("MolecularFormula", "Unknown")
    mw      = props.get("MolecularWeight", "Unknown")

    if smiles is None:
        raise ValueError(
            f"PubChem returned no SMILES for '{drug_name}'.\n"
            f"Available keys: {list(props.keys())}"
        )

    print(f"  ✓ Found on PubChem!")
    print(f"    CID              : {cid}")
    print(f"    IUPAC Name       : {iupac}")
    print(f"    Formula          : {formula}")
    print(f"    Molecular Weight : {mw} g/mol")
    print(f"    SMILES           : {smiles[:60]}{'...' if len(smiles) > 60 else ''}")

    return smiles, iupac, formula, cid


# ── Step 2: Encode input drug into latent space ───────────────────────────────
def encode_drug(smiles, model, scaler, device):
    """
    Converts a SMILES string into a latent vector using the trained GNN encoder.

    Args:
        smiles  : SMILES string of the input drug
        model   : trained MolecularVAE
        scaler  : fitted StandardScaler (from training)
        device  : torch device

    Returns:
        mu (np.ndarray): latent mean vector [1, latent_dim]
    """
    dummy_props = [0.0, 0.0, 0.0]
    graph = smiles_to_graph(smiles, dummy_props)

    if graph is None:
        raise ValueError(
            f"Could not parse SMILES: {smiles}\n"
            f"The molecule may be too complex for the current model."
        )

    graph = graph.to(device)

    # Batch vector for a single molecule — all zeros
    batch = torch.zeros(graph.x.shape[0], dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        mu, log_var = model.encode(graph.x, graph.edge_index, batch)

    return mu.cpu().numpy()  # [1, latent_dim]


# ── Step 3: Sample neighbors in latent space ──────────────────────────────────
def sample_neighbors(mu, model, scaler, device, num_samples=500, noise_scale=0.5):
    """
    Samples molecules from a neighborhood around the input drug's latent vector.
    Instead of sampling from N(0,1) randomly, we sample near the input drug
    to get SIMILAR molecules.

    Args:
        mu          : latent vector of input drug [1, latent_dim]
        model       : trained MolecularVAE
        scaler      : fitted StandardScaler
        device      : torch device
        num_samples : how many candidates to generate
        noise_scale : controls exploration radius
                      (smaller = more similar, larger = more diverse)

    Returns:
        props_real : denormalized properties [num_samples, 3]
        z_samples  : latent vectors          [num_samples, latent_dim]
    """
    latent_dim = mu.shape[1]
    mu_tensor  = torch.tensor(mu, dtype=torch.float).to(device)

    # Sample noise around mu (anchored to input drug, not origin)
    noise = torch.randn(num_samples, latent_dim).to(device) * noise_scale
    z     = mu_tensor + noise  # [num_samples, latent_dim]

    model.eval()
    with torch.no_grad():
        props_normalized = model.decode(z).cpu().numpy()

    # Denormalize → real property values
    props_real = scaler.inverse_transform(props_normalized)

    return props_real, z.cpu().numpy()


# ── Step 4: Lipinski filter ───────────────────────────────────────────────────
def lipinski_filter(logP, qed, SAS):
    """
    Returns True if the molecule passes drug-likeness rules:
        - logP between -2.0 and 5.0  (lipophilicity)
        - qed  >= 0.5                (drug-likeness score)
        - SAS  <= 6.0                (synthetic accessibility)
    """
    return (
        -2.0 <= logP <= 5.0 and
        qed  >= 0.5         and
        SAS  <= 6.0
    )


# ── Step 5: Match to real ZINC molecules ──────────────────────────────────────
def match_to_zinc(candidates, top_k=3):
    """
    Matches generated property vectors to the nearest real molecules in ZINC.
    Returns top_k candidates sorted by QED (drug-likeness).

    Args:
        candidates : list of dicts with logP, qed, SAS
        top_k      : number of top candidates to return

    Returns:
        DataFrame of top candidates with SMILES and properties
    """
    print(f"\n  Matching {len(candidates)} candidates to real ZINC molecules...")

    df_zinc     = pd.read_csv(CSV_PATH)
    zinc_props  = df_zinc[["logP", "qed", "SAS"]].values.astype(np.float32)
    zinc_smiles = df_zinc["smiles"].tolist()

    results = []
    for cand in candidates:
        gen_props = np.array([[
            cand["logP"],
            cand["qed"],
            cand["SAS"]
        ]])

        dists   = euclidean_distances(gen_props, zinc_props)[0]
        nearest = int(np.argmin(dists))

        results.append({
            "smiles"   : zinc_smiles[nearest],
            "logP"     : cand["logP"],
            "qed"      : cand["qed"],
            "SAS"      : cand["SAS"],
            "distance" : round(float(dists[nearest]), 4),
        })

    # Sort by QED descending then distance ascending, remove duplicates
    df = pd.DataFrame(results)
    df = df.sort_values(
        by        = ["qed", "distance"],
        ascending = [False, True]
    ).drop_duplicates(subset="smiles").reset_index(drop=True)

    return df.head(top_k)


# ── Step 6: Draw molecule PNG ─────────────────────────────────────────────────
def draw_molecule(smiles, size=(400, 300)):
    """Renders a SMILES string to a PIL Image using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts   = drawer.drawOptions()
    opts.backgroundColour    = (0.1, 0.11, 0.17, 1)
    opts.addStereoAnnotation = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    bio = io.BytesIO(drawer.GetDrawingText())
    return Image.open(bio)


def get_formula(smiles):
    """Returns the molecular formula for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Unknown"
    return rdMolDescriptors.CalcMolFormula(mol)


# ── Step 7: Save final PNG ────────────────────────────────────────────────────
def save_result_png(drug_name, input_smiles, input_formula, df_candidates, output_path):
    """
    Saves a clean PNG showing:
        - Input drug (left panel)
        - Top 3 generated candidates (right panels)
        - Chemical formula + properties for each
    """
    n_candidates = len(df_candidates)
    total_cols   = 1 + n_candidates

    fig = plt.figure(figsize=(total_cols * 4.5, 7))
    fig.patch.set_facecolor(BG_COLOR)

    # ── Main title ─────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Drug Discovery Pipeline — GNN-VAE\n"
        f"Input: {drug_name}  →  Top {n_candidates} Novel Candidates",
        fontsize   = 16,
        fontweight = "bold",
        color      = TEXT_COLOR,
        y          = 1.01
    )

    gs = gridspec.GridSpec(
        2, total_cols,
        figure        = fig,
        height_ratios = [0.85, 0.15],
        hspace        = 0.05,
        wspace        = 0.08
    )

    # ── Input drug panel ───────────────────────────────────────────────────────
    ax_input = fig.add_subplot(gs[0, 0])
    ax_input.set_facecolor(PANEL_COLOR)

    img_input = draw_molecule(input_smiles, size=(400, 300))
    if img_input:
        ax_input.imshow(img_input)
    ax_input.axis("off")

    for spine in ax_input.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(ACCENT1)
        spine.set_linewidth(2.5)

    ax_input.set_title(
        f"INPUT DRUG\n{drug_name}",
        fontsize   = 11,
        fontweight = "bold",
        color      = ACCENT1,
        pad        = 8
    )

    # Formula below input
    ax_label_input = fig.add_subplot(gs[1, 0])
    ax_label_input.set_facecolor(PANEL_COLOR)
    ax_label_input.axis("off")
    ax_label_input.text(
        0.5, 0.5,
        f"Formula: {input_formula}",
        transform  = ax_label_input.transAxes,
        fontsize   = 10,
        color      = TEXT_COLOR,
        ha         = "center",
        va         = "center",
        fontfamily = "monospace"
    )

    # ── Arrow ──────────────────────────────────────────────────────────────────
    fig.text(
        (1 / total_cols) + 0.01,
        0.55,
        "→",
        fontsize   = 28,
        color      = ACCENT4,
        ha         = "center",
        va         = "center",
        fontweight = "bold"
    )

    # ── Candidate panels ───────────────────────────────────────────────────────
    for i, (_, row) in enumerate(df_candidates.iterrows()):
        col    = i + 1
        ax_mol = fig.add_subplot(gs[0, col])
        ax_mol.set_facecolor(PANEL_COLOR)

        img_mol = draw_molecule(row["smiles"], size=(400, 300))
        if img_mol:
            ax_mol.imshow(img_mol)
        ax_mol.axis("off")

        for spine in ax_mol.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(ACCENT3)
            spine.set_linewidth(2)

        formula = get_formula(row["smiles"])
        ax_mol.set_title(
            f"CANDIDATE #{i+1}\nFormula: {formula}",
            fontsize   = 10,
            fontweight = "bold",
            color      = ACCENT3,
            pad        = 8
        )

        ax_mol.text(
            0.04, 0.96, f"#{i+1}",
            transform  = ax_mol.transAxes,
            fontsize   = 11,
            fontweight = "bold",
            color      = ACCENT4,
            va         = "top"
        )

        # Properties below candidate
        ax_label = fig.add_subplot(gs[1, col])
        ax_label.set_facecolor(PANEL_COLOR)
        ax_label.axis("off")

        props_text = (
            f"logP={row['logP']:.2f}   "
            f"QED={row['qed']:.3f}   "
            f"SAS={row['SAS']:.2f}   "
            f"Dist={row['distance']:.3f}"
        )
        ax_label.text(
            0.5, 0.5,
            props_text,
            transform  = ax_label.transAxes,
            fontsize   = 8.5,
            color      = ACCENT2,
            ha         = "center",
            va         = "center",
            fontfamily = "monospace"
        )

    # ── Legend ─────────────────────────────────────────────────────────────────
    fig.text(
        0.5, -0.04,
        "logP: Lipophilicity (-2 to 5 ideal)   |   "
        "QED: Drug-likeness (>0.5, max=1.0)   |   "
        "SAS: Synthetic Accessibility (<6.0, lower=easier)   |   "
        "Dist: Property distance to real molecule (lower=closer)",
        ha        = "center",
        fontsize  = 7.5,
        color     = GRID_COLOR,
        fontstyle = "italic"
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(
        output_path,
        dpi         = 180,
        bbox_inches = "tight",
        facecolor   = BG_COLOR
    )
    plt.close()
    print(f"\n  ✓ Result saved → {output_path}")


# ── Main Pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(drug_name, num_samples=2000, top_k=3, noise_scale=0.5):
    """
    Full pipeline: drug name → novel candidate molecules → PNG output.

    Args:
        drug_name    : Name of the input drug (e.g. "Aspirin")
        num_samples  : Number of latent samples to generate
        top_k        : Number of top candidates to show
        noise_scale  : Exploration radius in latent space
                       (0.3=very similar, 0.5=balanced, 1.0=diverse)
    """

    print("\n" + "=" * 60)
    print("  MOLECULAR DRUG DISCOVERY PIPELINE")
    print("  GNN-VAE | ZINC Dataset | PubChem Lookup")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    print("\n  Loading trained model...")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"No checkpoint found at {CHECKPOINT_PATH}\n"
            f"Please run train.py first!"
        )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    config     = checkpoint["config"]

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

    # Rebuild scaler from checkpoint
    scaler                = StandardScaler()
    scaler.mean_          = np.array(checkpoint["scaler_mean"])
    scaler.scale_         = np.array(checkpoint["scaler_scale"])
    scaler.var_           = scaler.scale_ ** 2
    scaler.n_features_in_ = 3

    print(f"  ✓ Model loaded (trained for {checkpoint['epoch']} epochs, "
          f"val_loss={checkpoint['val_loss']:.4f})")

    # ── Step 1: PubChem lookup ─────────────────────────────────────────────────
    input_smiles, iupac_name, input_formula, cid = fetch_smiles_from_pubchem(drug_name)

    # ── Step 2: Encode input drug ──────────────────────────────────────────────
    print(f"\n  Encoding '{drug_name}' into latent space...")
    mu = encode_drug(input_smiles, model, scaler, device)
    print(f"  ✓ Latent vector shape : {mu.shape}")

    # ── Step 3: Sample neighbors ───────────────────────────────────────────────
    print(f"\n  Sampling {num_samples} molecules near '{drug_name}' in latent space...")
    print(f"  Exploration radius (noise_scale) : {noise_scale}")
    props_real, z_samples = sample_neighbors(
        mu, model, scaler, device,
        num_samples = num_samples,
        noise_scale = noise_scale
    )

    # ── Step 4: Apply Lipinski filter ──────────────────────────────────────────
    print(f"\n  Applying Lipinski drug-likeness filter...")
    candidates = []
    for i in range(len(props_real)):
        logP = float(props_real[i, 0])
        qed  = float(props_real[i, 1])
        SAS  = float(props_real[i, 2])

        if lipinski_filter(logP, qed, SAS):
            candidates.append({
                "logP" : round(logP, 4),
                "qed"  : round(qed,  4),
                "SAS"  : round(SAS,  4),
            })

    print(f"  Passed filter : {len(candidates)} / {num_samples} molecules")

    if len(candidates) == 0:
        print("\n  No candidates passed the filter.")
        print("  Try: --samples 1000  or  --noise 0.8")
        return

    # ── Step 5: Match to ZINC ──────────────────────────────────────────────────
    df_top = match_to_zinc(candidates, top_k=top_k)

    # ── Step 6: Print summary table ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TOP {len(df_top)} CANDIDATES SIMILAR TO '{drug_name.upper()}'")
    print(f"{'='*60}")
    print(f"  {'#':<4} {'Formula':<14} {'logP':>6} {'QED':>6} {'SAS':>6} {'Dist':>6}")
    print(f"  {'-'*50}")
    for i, (_, row) in enumerate(df_top.iterrows()):
        formula = get_formula(row["smiles"])
        print(
            f"  {i+1:<4} {formula:<14} "
            f"{row['logP']:>6.2f} "
            f"{row['qed']:>6.3f} "
            f"{row['SAS']:>6.2f} "
            f"{row['distance']:>6.3f}"
        )
    print(f"{'='*60}")

    # ── Step 7: Save PNG ───────────────────────────────────────────────────────
    safe_name   = drug_name.replace(" ", "_").lower()
    output_path = os.path.join(OUTPUT_DIR, f"drug_candidates_{safe_name}.png")

    save_result_png(
        drug_name     = drug_name,
        input_smiles  = input_smiles,
        input_formula = input_formula,
        df_candidates = df_top,
        output_path   = output_path
    )

    print(f"\n  Pipeline complete!")
    print(f"  Input drug   : {drug_name} ({input_formula})")
    print(f"  Candidates   : {len(df_top)} novel drug-like molecules")
    print(f"  Output saved : {output_path}")
    print("=" * 60)


# ── CLI Entry Point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "GNN-VAE Drug Discovery — "
            "find novel candidates similar to a known drug"
        )
    )
    parser.add_argument(
        "--drug",
        type    = str,
        default = None,
        help    = "Name of the input drug (e.g. 'Aspirin', 'Ibuprofen')"
    )
    parser.add_argument(
        "--samples",
        type    = int,
        default = 2000,
        help    = "Number of latent samples to generate (default: 500)"
    )
    parser.add_argument(
        "--top",
        type    = int,
        default = 3,
        help    = "Number of top candidates to output (default: 3)"
    )
    parser.add_argument(
        "--noise",
        type    = float,
        default = 0.5,
        help    = "Exploration radius — 0.3=similar, 1.0=diverse (default: 0.5)"
    )

    args = parser.parse_args()

    # If no drug name via CLI, prompt interactively
    if args.drug is None:
        print("\n" + "=" * 60)
        print("  MOLECULAR DRUG DISCOVERY — GNN-VAE")
        print("=" * 60)
        print("  Enter the name of a known drug to find similar,")
        print("  novel, drug-like candidates.\n")
        print("  Examples: Aspirin, Ibuprofen, Paracetamol,")
        print("            Metformin, Caffeine, Penicillin")
        print("=" * 60)
        drug_name = input("\n  Enter drug name: ").strip()
    else:
        drug_name = args.drug

    run_pipeline(
        drug_name   = drug_name,
        num_samples = args.samples,
        top_k       = args.top,
        noise_scale = args.noise
    )