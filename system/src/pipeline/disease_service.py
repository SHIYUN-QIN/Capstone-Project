import json
from pathlib import Path

class DiseaseService:
    def __init__(self, cache_dir: str = "data"):
        root = Path(__file__).parent.parent.parent
        self.dt_path = root / cache_dir / "disease_target_cache.json"
        self.dpi_path = root / cache_dir / "disease_protein_index.json"
        
        self.disease_target = {}
        self.disease_index = {}
        
        if self.dt_path.exists():
            with open(self.dt_path, "r", encoding="utf-8") as f:
                self.disease_target = json.load(f)
        if self.dpi_path.exists():
            with open(self.dpi_path, "r", encoding="utf-8") as f:
                self.disease_index = json.load(f)

    def search_diseases(self, keyword: str) -> list[dict]:
        kw = keyword.lower()
        res = []
        for efo, data in self.disease_index.items():
            if kw in data["disease_name"].lower() or kw in efo.lower():
                res.append({"efo_id": efo, "disease_name": data["disease_name"], "target_count": len(data["targets"])})
        return res

    def get_disease_targets(self, efo_id: str) -> dict:
        if efo_id not in self.disease_index: return None
        data = self.disease_index[efo_id]
        
        targets = []
        for p in data["targets"]:
            score = 0.0
            for d in self.disease_target.get(p, {}).get("diseases", []):
                if d["efo_id"] == efo_id:
                    score = d["score"]
                    break
            targets.append({"protein_id": p, "association_score": score})
        
        return {"disease_name": data["disease_name"], "targets": targets}

    def get_disease_drug_candidates(self, efo_id: str, results_store) -> list[dict]:
        targets = self.get_disease_targets(efo_id)
        if not targets: return []
        
        candidates = []
        for t in targets["targets"]:
            pid = t["protein_id"]
            asc = t["association_score"]
            drugs = results_store.get_protein_drugs(pid, "drugbank", top_k=10)
            for d in drugs:
                if d["decision"] == "accept":
                    candidates.append({
                        "drug_id": d["drug_id"],
                        "drug_name": d["drug_name"] or d.get("drug_smiles", ""),
                        "target_protein": pid,
                        "interaction_prob": d["interaction_prob"],
                        "association_score": asc,
                        "combined_score": asc * d["interaction_prob"]
                    })
        
        candidates.sort(key=lambda x: x["combined_score"], reverse=True)
        return candidates[:50]

    def get_protein_diseases(self, protein_id: str) -> list[dict]:
        res = []
        for d in self.disease_target.get(protein_id, {}).get("diseases", []):
            res.append({
                "efo_id": d.get("efo_id", ""),
                "disease_name": d.get("disease_name", ""),
                "association_score": d.get("score", 0.0)
            })
        if not res:
            # Let's search inside the index just in case it's reverse mapped without being in target directly
            for efo, data in self.disease_index.items():
                if protein_id in data["targets"]:
                    # score might not be available directly here
                    res.append({
                        "efo_id": efo,
                        "disease_name": data["disease_name"],
                        "association_score": 0.0
                    })
        return res
