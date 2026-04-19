import json
import logging
import time
from pathlib import Path
from src.db.results_store import ResultsStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProteinMetadataService:
    def __init__(self, cache_file="data/protein_metadata_cache.json"):
        root = Path(__file__).parent.parent.parent
        self.cache_path = root / cache_file
        if self.cache_path.exists():
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    def get(self, protein_id: str) -> dict:
        return self.cache.get(protein_id)

    def search(self, query: str) -> list[dict]:
        query = query.lower()
        results = []
        for pid, data in self.cache.items():
            if (query in (data.get("protein_name") or "").lower() or 
                query in (data.get("gene_name") or "").lower() or
                query in pid.lower()):
                results.append({"protein_id": pid, **data})
        return results

if __name__ == "__main__":
    pass
