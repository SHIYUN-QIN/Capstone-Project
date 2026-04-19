import numpy as np
import pandas as pd
import torch
import json
import pickle
import sys
from pathlib import Path
from tqdm import tqdm

# Add src to path if needed (assuming script is run from project root)
sys.path.append(str(Path(__file__).parent.parent))

from src.model.inference import ModelManager
from src.pipeline.drug_features import DrugFeaturePipeline
from src.pipeline.protein_features import ProteinFeaturePipeline
from src.uncertainty.scorer import UncertaintyScorer

def calibrate_dataset(dataset_name: str, 
                      zenodo_path: Path, 
                      data_path: Path, 
                      model_dir: Path, 
                      output_dir: Path,
                      sample_size: int = 1000):
    """
    1. Load model and lookups
    2. Pick samples from the original CSV with labels
    3. Run inference
    4. Compute accuracy per entropy bin
    5. Save results to JSON
    """
    print(f"\n--- Calibrating {dataset_name} ---")
    
    # Paths
    if dataset_name == "drugbank":
        csv_path = zenodo_path / "bond_angle/bond_angle/bond_angle/total_cid_unid_csv.csv"
        s_col, u_col = "cid", "Uniprot ID" # CID might actually point to SMILES in data mapping?
        # Actually prepare_data.py uses CID as index to d_3d_raw which has SMILES
    elif dataset_name == "davis":
        csv_path = zenodo_path / "davis/davis/davis_total_cid_unid.csv"
        s_col, u_col = "smiles", "uid"
    elif dataset_name == "kiba":
        csv_path = zenodo_path / "KIBA/KIBA_total_cid_unid.csv"
        s_col, u_col = "smiles", "uid"
    else:
        print(f"Unknown dataset {dataset_name}")
        return

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return

    # Load data
    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    if dataset_name == "drugbank":
        # DrugBank CID column doesn't have smiles, but the d_3d_raw has it.
        # But prepare_data.py already mapped everything to lookups.
        # We need a way to link CSV rows to lookups.
        # Let's hope the index in CSV matches the index in lookups?
        # No, lookups are keyed by CANONICAL SMILES.
        # Let's read the raw npy to get SMILES for DrugBank CID.
        d3_raw_path = zenodo_path / "bond_angle/bond_angle/bond_angle/d_3D_feature.npy"
        d3_raw = np.load(d3_raw_path, allow_pickle=True)
        # Add smiles column to df
        smiles_list = []
        for i in range(len(df)):
            smiles_list.append(d3_raw[i].get('smiles', ''))
        df['smiles'] = smiles_list
        s_col = 'smiles'
        del d3_raw

    # Sample for performance
    if len(df) > sample_size:
        print(f"Sampling {sample_size} rows from {len(df)}...")
        df = df.sample(sample_size, random_state=42)

    # Initialize components
    print("Initializing system components...")
    manager = ModelManager(model_dir=model_dir)
    manager.load_model(dataset_name)
    
    with open(data_path / "drug_2d_lookup.pkl", "rb") as f:
        d2_lookup = pickle.load(f)
    with open(data_path / "drug_3d_lookup.pkl", "rb") as f:
        d3_lookup = pickle.load(f)
    with open(data_path / "protein_lookup.pkl", "rb") as f:
        prot_lookup = pickle.load(f)

    from src.pipeline.drug_features import canonical_smiles

    all_probs = []
    all_labels = []
    all_entropies = []
    
    print(f"Running inference on {len(df)} samples...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        smiles = str(row[s_col])
        uid = str(row[u_col])
        label = int(row['label'])
        
        canon = canonical_smiles(smiles)
        d2 = d2_lookup.get(canon)
        d3 = d3_lookup.get(canon)
        prot = prot_lookup.get(uid)
        
        if d2 is None or d3 is None or prot is None:
            continue
            
        # We need to convert from float16 to float32 if lookups are f16
        d2 = d2.astype(np.float32)
        d3 = d3.astype(np.float32)
        prot = prot.astype(np.float32)
            
        res = manager.predict_single(d2, d3, prot, dataset_name)
        probs = res["probs"]
        entropy = UncertaintyScorer.prediction_entropy(probs)
        
        all_probs.append(probs)
        all_labels.append(label)
        all_entropies.append(entropy)

    if not all_probs:
        print("No valid samples processed.")
        return

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_entropies = np.array(all_entropies)
    
    # Calculate accuracy per bin
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    bin_accuracies = []
    bin_counts = []
    
    for i in range(len(bins)-1):
        mask = (all_entropies >= bins[i]) & (all_entropies < bins[i+1])
        if np.sum(mask) > 0:
            preds = np.argmax(all_probs[mask], axis=1)
            acc = np.mean(preds == all_labels[mask])
            bin_accuracies.append(float(acc))
            bin_counts.append(int(np.sum(mask)))
        else:
            bin_accuracies.append(0.0)
            bin_counts.append(0)
            
    # Heuristic for thresholds
    # High: accuracy > 90%
    # Medium: accuracy > 70%
    high_thresh = 0.15
    for i, acc in enumerate(bin_accuracies):
        if acc < 0.90:
            high_thresh = bins[i]
            break
            
    med_thresh = 0.40
    for i, acc in enumerate(bin_accuracies):
        if acc < 0.70:
            med_thresh = bins[i]
            break

    result = {
        "dataset": dataset_name,
        "bins": bins,
        "accuracy": bin_accuracies,
        "counts": bin_counts,
        "thresholds": {
            "high": high_thresh,
            "medium": med_thresh
        }
    }
    
    out_file = output_dir / f"calibration_{dataset_name}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Calibration saved to {out_file}")
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--samples", type=int, default=200) # Small for speed
    args = parser.parse_args()
    
    root = Path("DTIsystem")
    zenodo = Path("zenodo")
    datasets = ["drugbank", "davis", "kiba"] if args.dataset == "all" else [args.dataset]
    
    for ds in datasets:
        calibrate_dataset(
            ds, 
            zenodo, 
            root / "data", 
            root / "weights", 
            root / "data",
            sample_size=args.samples
        )
