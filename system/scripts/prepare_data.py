import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
import gc

# Keys from notebook Cell 2
ATOM_SCALAR_KEYS = [
    'atomic_num', 'chiral_tag', 'degree', 'explicit_valence',
    'formal_charge', 'hybridization', 'implicit_valence',
    'is_aromatic', 'total_numHs', 'mass'
]
ATOM_VEC_KEYS = ['atom_pos']
BOND_SCALAR_KEYS = ['bond_dir', 'bond_type', 'is_in_ring', 'bond_length']
ANGLE_SCALAR_KEYS = ['bond_angle', 'Ba_bond_angle', 'Bl_bond_length']
AD_DIST_KEY = 'Ad_atom_dist'
FIXED_FP_KEYS = ['morgan_fp', 'maccs_fp', 'daylight_fg_counts']
D3_MORGAN = 200
D3_MACCS = 167
D3_DFG = 127
D3_DIM = 515

def safe_mean(arr, fallback=0.0):
    return float(arr.mean()) if len(arr) > 0 else fallback

def safe_mean_vec(arr, ndim, fallback=0.0):
    if arr.shape[0] == 0:
        return np.zeros(ndim, dtype=np.float32)
    return arr.mean(axis=0).flatten().astype(np.float32)

def dict_to_vec(d):
    parts = []
    for k in ATOM_SCALAR_KEYS:
        v = d.get(k, None)
        if v is None: parts.append(np.zeros(1, dtype=np.float32))
        else:
            arr = np.asarray(v, dtype=np.float32).flatten()
            parts.append(np.array([safe_mean(arr)], dtype=np.float32))

    for k in ATOM_VEC_KEYS:
        v = d.get(k, None)
        if v is None: parts.append(np.zeros(3, dtype=np.float32))
        else:
            arr = np.asarray(v, dtype=np.float32)
            if arr.ndim == 1: arr = arr.reshape(-1, 1)
            parts.append(safe_mean_vec(arr, arr.shape[1] if arr.ndim > 1 else 1))

    for k in BOND_SCALAR_KEYS:
        v = d.get(k, None)
        if v is None: parts.append(np.zeros(1, dtype=np.float32))
        else:
            arr = np.asarray(v, dtype=np.float32).flatten()
            parts.append(np.array([safe_mean(arr)], dtype=np.float32))

    for k in ANGLE_SCALAR_KEYS:
        v = d.get(k, None)
        if v is None: parts.append(np.zeros(1, dtype=np.float32))
        else:
            arr = np.asarray(v, dtype=np.float32).flatten()
            parts.append(np.array([safe_mean(arr)], dtype=np.float32))

    v = d.get(AD_DIST_KEY, None)
    if v is None: parts.append(np.zeros(1, dtype=np.float32))
    else:
        arr = np.asarray(v, dtype=np.float32).flatten()
        parts.append(np.array([safe_mean(arr)], dtype=np.float32))

    for k, expected_dim in zip(FIXED_FP_KEYS, [D3_MORGAN, D3_MACCS, D3_DFG]):
        v = d.get(k, None)
        if v is None: parts.append(np.zeros(expected_dim, dtype=np.float32))
        else:
            arr = np.asarray(v, dtype=np.float32).flatten()
            if len(arr) != expected_dim:
                tmp = np.zeros(expected_dim, dtype=np.float32)
                tmp[:min(len(arr), expected_dim)] = arr[:expected_dim]
                arr = tmp
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            parts.append(arr)

    result = np.concatenate(parts).astype(np.float32)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    assert len(result) == D3_DIM, f'Expected {D3_DIM}, got {len(result)}'
    return result

def canonical_smiles(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol else smiles
    except: return smiles

def build_lookups(zenodo_dir: Path, output_dir: Path, use_float16=True):
    print(f"Building lookups from Zenodo... (FP16={use_float16})")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    d2_lookup = {}
    d3_lookup = {}
    protein_lookup = {}

    datasets = [
        {
            "name": "drugbank",
            "dir": zenodo_dir / "bond_angle/bond_angle/bond_angle",
            "d2": "d_feature.npy", "d3": "d_3D_feature.npy", "t": "t_feature.npy",
            "csv": "total_cid_unid_csv.csv"
        },
        {
            "name": "davis",
            "dir": zenodo_dir / "davis/davis",
            "d2": "davis_d_2D_features.npy", "d3": "davis_d_3d_feature.npy", "t": "davis_t_feature.npy",
            "csv": "davis_total_cid_unid.csv"
        },
        {
            "name": "kiba",
            "dir": zenodo_dir / "KIBA",
            "d2": "KIBA_d_2d_feature.npy", "d3": "KIBA_d_3d_feature.npy", "t": "KIBA_t_1D_feature.npy",
            "csv": "KIBA_total_cid_unid.csv"
        }
    ]

    for ds in datasets:
        name = ds["name"]
        path = ds["dir"]
        if not path.exists():
            print(f"Warning: Path {path} not found. Skipping.")
            continue
            
        try:
            print(f"Processing {name}...")
            csv_path = path / ds["csv"]
            if not csv_path.exists(): csv_path = list(path.glob("*.csv"))[0]
            df = pd.read_csv(csv_path)
            
            # Identify columns
            cols = [c.lower() for c in df.columns]
            s_col = None
            if 'smiles' in cols: s_col = df.columns[cols.index('smiles')]
            elif 'smile' in cols: s_col = df.columns[cols.index('smile')]
            
            u_col = None
            for cand in ['uid', 'uniprot id', 'target_name', 'uniprot_id']:
                if cand in cols:
                    u_col = df.columns[cols.index(cand)]
                    break
            
            # 1. Process Drug Features
            print(f"  Indexing Drug features...")
            d_2d_raw = np.load(path / ds["d2"], allow_pickle=True)
            d_3d_raw = np.load(path / ds["d3"], allow_pickle=True)
            
            for i, row in df.iterrows():
                smiles = None
                if s_col: smiles = str(row[s_col])
                else:
                    # Fallback to dictionary
                    d_dict = d_3d_raw[i]
                    smiles = d_dict.get('smiles', None)
                
                if not smiles or smiles == '1' or smiles == '0':
                    # Last resort fallback (try all d_3d_raw[i] items if i index is correct)
                    d_dict = d_3d_raw[i]
                    smiles = d_dict.get('smiles', str(i)) 
                
                canon = canonical_smiles(smiles)
                if canon not in d2_lookup: d2_lookup[canon] = d_2d_raw[i].astype(np.float16 if use_float16 else np.float32)
                if canon not in d3_lookup: d3_lookup[canon] = dict_to_vec(d_3d_raw[i]).astype(np.float16 if use_float16 else np.float32)
            
            del d_2d_raw, d_3d_raw
            gc.collect()

            # 2. Process Proteins
            print(f"  Finding unique proteins...")
            unique_uids = df[u_col].unique()
            needed_uids = [uid for uid in unique_uids if str(uid) not in protein_lookup]
            
            if needed_uids:
                print(f"  Loading {len(needed_uids)} new proteins...")
                t_feat_raw = np.load(path / ds["t"], allow_pickle=True)
                
                # Manual map to avoid slow index lookups in loop
                uid_to_first_idx = {}
                for i, uid in enumerate(df[u_col]):
                    if uid not in uid_to_first_idx:
                        uid_to_first_idx[uid] = i
                
                for uid in needed_uids:
                    idx = uid_to_first_idx[uid]
                    p_emb = t_feat_raw[idx]
                    if not isinstance(p_emb, np.ndarray): p_emb = np.array(p_emb)
                    if p_emb.ndim == 1 and len(p_emb) > 1024: p_emb = p_emb.reshape(-1, 1024)
                    protein_lookup[str(uid)] = p_emb.astype(np.float16 if use_float16 else np.float32)
                
                del t_feat_raw
                gc.collect()

            del df
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {name}: {e}")

    print(f"Saving indexed data...")
    for label, data, fn in [("d2", d2_lookup, "drug_2d_lookup.pkl"), 
                            ("d3", d3_lookup, "drug_3d_lookup.pkl"), 
                            ("prot", protein_lookup, "protein_lookup.pkl")]:
        with open(output_dir / fn, "wb") as f:
            pickle.dump(data, f)
        print(f"  Saved {len(data)} items to {fn}")
        del data
        gc.collect()
    
    print("Indexing Complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--zenodo", type=Path, default=Path("zenodo"))
    parser.add_argument("--out", type=Path, default=Path("DTIsystem/data"))
    parser.add_argument("--float32", action="store_true")
    args = parser.parse_args()
    build_lookups(args.zenodo, args.out, use_float16=not args.float32)
