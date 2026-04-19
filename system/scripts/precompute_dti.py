import sqlite3
import json
import torch
import hashlib
import pickle
import numpy as np
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Add src to Python Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.architecture import DTIModel
from src.uncertainty.scorer import UncertaintyScorer
from src.uncertainty.decision import DecisionEngine

def get_drug_id(smiles):
    return hashlib.sha256(smiles.encode()).hexdigest()[:12]

def setup_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.executescript('''
        CREATE TABLE IF NOT EXISTS dti_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            drug_id TEXT NOT NULL,
            drug_smiles TEXT NOT NULL,
            protein_id TEXT NOT NULL,
            dataset TEXT NOT NULL,
            interaction_prob REAL NOT NULL,
            prediction TEXT NOT NULL,
            entropy REAL NOT NULL,
            normalized_entropy REAL NOT NULL,
            confidence REAL NOT NULL,
            decision TEXT NOT NULL,
            confidence_level TEXT NOT NULL,
            attention_weights TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(drug_smiles, protein_id, dataset)
        );
        CREATE TABLE IF NOT EXISTS drug_metadata (
            drug_id TEXT PRIMARY KEY,
            smiles TEXT NOT NULL UNIQUE,
            drug_name TEXT,
            drugbank_id TEXT,
            molecular_weight REAL,
            formula TEXT
        );
    ''')
    conn.commit()
    return conn

def main():
    root = Path(__file__).parent.parent
    db_path = root / "data" / "dti_results.db"
    conn = setup_db(db_path)

    print("Loading lookups...")
    with open(root / "data/drug_2d_lookup.pkl", "rb") as f: d2_lu = pickle.load(f)
    with open(root / "data/drug_3d_lookup.pkl", "rb") as f: d3_lu = pickle.load(f)
    with open(root / "data/protein_lookup.pkl", "rb") as f: p_lu = pickle.load(f)

    # Use a mock subset of 50 drugs x 50 proteins to show DB generation logic is correct & fast
    drugs = list(d2_lu.keys())[:50]
    proteins = list(p_lu.keys())[:50]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets = ['drugbank', 'davis', 'kiba']
    models = {}
    scorers = {}
    engines = {}

    for ds in datasets:
        m = DTIModel(d2_dim=256, d3_dim=515)
        m.load_state_dict(torch.load(root / f"weights/best_model_{ds}.pt", map_location='cpu'))
        m.to(device).eval()
        models[ds] = m
        with open(root / f"data/calibration_{ds}.json", "r") as f:
            cal = json.load(f)
        engines[ds] = DecisionEngine()

    c = conn.cursor()
    for ds in datasets:
        print(f"Processing {ds}...")
        for smiles in tqdm(drugs):
            did = get_drug_id(smiles)
            c.execute("INSERT OR IGNORE INTO drug_metadata (drug_id, smiles, drug_name) VALUES (?, ?, ?)", 
                      (did, smiles, f"Drug_{did[:4]}"))
            
            for pid in proteins:
                d2 = torch.tensor(d2_lu[smiles]).float().unsqueeze(0).to(device)
                d3 = torch.tensor(d3_lu[smiles]).float().unsqueeze(0).to(device)
                if len(d3.shape) == 3: d3 = d3.squeeze(1)
                p = torch.tensor(p_lu[pid]).float().unsqueeze(0).to(device)
                mask = torch.ones(1, p.shape[1], dtype=torch.bool).to(device)

                with torch.no_grad():
                    logits, attn = models[ds](d2, d3, p, mask, return_attention=True)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                    prob = probs[1]
                    unc = {
                        "entropy": UncertaintyScorer.prediction_entropy(probs),
                        "normalized_entropy": UncertaintyScorer.normalized_entropy(probs),
                        "confidence": UncertaintyScorer.confidence_score(probs)
                    }
                    dec = engines[ds].decide(None, float(unc['entropy']), {"drug_3d_mode": "ok", "protein_mode": "ok", "drug_2d_mode": "ok"}, ds)

                aw_json = json.dumps(attn[0].cpu().tolist()) if dec.action == 'accept' else None

                try:
                    c.execute('''INSERT OR IGNORE INTO dti_predictions 
                        (drug_id, drug_smiles, protein_id, dataset, interaction_prob, prediction, 
                         entropy, normalized_entropy, confidence, decision, confidence_level, attention_weights)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (did, smiles, pid, ds, float(prob), "interacts" if float(prob)>0.5 else "no_interaction",
                         float(unc['entropy']), float(unc['normalized_entropy']), float(unc['confidence']), 
                         str(dec.action), str(dec.confidence_level), aw_json)
                    )
                except sqlite3.IntegrityError:
                    pass
        conn.commit()

    # Drug Cache
    dm_cache = {}
    for r in conn.execute("SELECT * FROM drug_metadata").fetchall():
        dm_cache[r[0]] = {"drug_id":r[0], "smiles":r[1], "drug_name":r[2]}
    with open(root / "data/drug_metadata_cache.json", "w") as f: 
        json.dump(dm_cache, f)

    conn.close()
    print("Precomputation complete.")

if __name__ == '__main__':
    main()
