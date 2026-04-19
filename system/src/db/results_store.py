import sqlite3
import json

class ResultsStore:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def get_pair_all_datasets(self, drug_smiles: str, protein_id: str) -> list[dict]:
        c = self.conn.cursor()
        c.execute('''SELECT * FROM dti_predictions 
                     WHERE drug_smiles = ? AND protein_id = ?''', (drug_smiles, protein_id))
        return [dict(r) for r in c.fetchall()]

    def get_pair_attention(self, drug_smiles: str, protein_id: str, dataset: str) -> list[float]:
        c = self.conn.cursor()
        c.execute('''SELECT attention_weights FROM dti_predictions 
                     WHERE drug_smiles = ? AND protein_id = ? AND dataset = ?''', 
                  (drug_smiles, protein_id, dataset))
        row = c.fetchone()
        if row and row['attention_weights']:
            return json.loads(row['attention_weights'])
        return []

    def list_drugs(self, limit=100, offset=0, search: str = None) -> list[dict]:
        c = self.conn.cursor()
        if search:
            query = f"%{search}%"
            c.execute('SELECT * FROM drug_metadata WHERE drug_id LIKE ? OR smiles LIKE ? OR drug_name LIKE ? LIMIT ? OFFSET ?', 
                      (query, query, query, limit, offset))
        else:
            c.execute('SELECT * FROM drug_metadata LIMIT ? OFFSET ?', (limit, offset))
        return [dict(r) for r in c.fetchall()]

    def list_proteins(self, limit=100, offset=0, search: str = None) -> list[dict]:
        c = self.conn.cursor()
        if search:
            query = f"%{search}%"
            c.execute('SELECT DISTINCT protein_id FROM dti_predictions WHERE protein_id LIKE ? LIMIT ? OFFSET ?', 
                      (query, limit, offset))
        else:
            c.execute('SELECT DISTINCT protein_id FROM dti_predictions LIMIT ? OFFSET ?', (limit, offset))
        return [{"protein_id": r['protein_id']} for r in c.fetchall()]

    def get_protein_drugs(self, protein_id: str, dataset: str, top_k=50) -> list[dict]:
        c = self.conn.cursor()
        c.execute('''SELECT p.*, d.drug_name FROM dti_predictions p
                     LEFT JOIN drug_metadata d ON p.drug_id = d.drug_id
                     WHERE p.protein_id = ? AND p.dataset = ? 
                     ORDER BY p.interaction_prob DESC LIMIT ?''', 
                  (protein_id, dataset, top_k))
        return [dict(r) for r in c.fetchall()]

    def get_drug_proteins(self, drug_smiles: str, dataset: str, top_k=50) -> list[dict]:
        c = self.conn.cursor()
        c.execute('''SELECT * FROM dti_predictions 
                     WHERE drug_smiles = ? AND dataset = ? 
                     ORDER BY interaction_prob DESC LIMIT ?''', 
                  (drug_smiles, dataset, top_k))
        return [dict(r) for r in c.fetchall()]

    def get_valid_options(self) -> dict:
        c = self.conn.cursor()
        c.execute('''SELECT DISTINCT p.drug_id, d.drug_name FROM dti_predictions p
                     LEFT JOIN drug_metadata d ON p.drug_id = d.drug_id
                     WHERE p.attention_weights IS NOT NULL''')
        drugs = [{"id": r['drug_id'], "label": r['drug_name'] or r['drug_id']} for r in c.fetchall()]
        
        c.execute('''SELECT DISTINCT protein_id FROM dti_predictions 
                     WHERE attention_weights IS NOT NULL''')
        proteins = [{"id": r['protein_id'], "label": r['protein_id']} for r in c.fetchall()]
        
        return {"drugs": drugs, "proteins": proteins}

    def get_statistics(self) -> dict:
        c = self.conn.cursor()
        c.execute('SELECT dataset, COUNT(*) as c FROM dti_predictions GROUP BY dataset')
        stats = {r['dataset']: r['c'] for r in c.fetchall()}
        c.execute('SELECT COUNT(DISTINCT drug_id) as c FROM drug_metadata')
        row = c.fetchone()
        d_count = row['c'] if row else 0
        return {"pairs": stats, "total_drugs": d_count}

    def close(self):
        self.conn.close()
