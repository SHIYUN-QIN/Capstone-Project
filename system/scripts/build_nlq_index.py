import sqlite3
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm

def build_index():
    root = Path(__file__).parent.parent
    db_path = root / "data" / "dti_results.db"
    
    if not db_path.exists():
        print("DB not found, skipping NLQ build.")
        return

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        SELECT p.drug_id, p.protein_id, p.dataset, p.interaction_prob, p.decision, 
               d.drug_name, d.smiles
        FROM dti_predictions p
        LEFT JOIN drug_metadata d ON p.drug_id = d.drug_id
        WHERE p.decision IN ("accept", "review")
    ''')
    rows = c.fetchall()
    
    docs = []
    texts = []
    
    for r in tqdm(rows, desc="Formatting docs"):
        meta = {
            "drug_id": r[0], "protein_id": r[1], "dataset": r[2], 
            "interaction_prob": r[3], "decision": r[4], 
            "drug_name": r[5] or r[6][:10]
        }
        text_repr = f"Drug {meta['drug_name']} has a {meta['decision']} interaction with protein {meta['protein_id']} on {meta['dataset']} dataset with probability {meta['interaction_prob']:.2f}."
        docs.append({"meta": meta, "text": text_repr})
        texts.append(text_repr)
    
    if len(texts) > 0:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embs = model.encode(texts, show_progress_bar=True)
        
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        
        faiss.write_index(index, str(root / "data" / "nlq_index.faiss"))
        with open(root / "data" / "nlq_documents.json", "w") as f:
            json.dump(docs, f)
            
        print(f"Index built with {len(docs)} documents.")

if __name__ == "__main__":
    build_index()
