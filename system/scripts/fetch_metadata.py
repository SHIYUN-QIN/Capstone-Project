import json
import logging
from pathlib import Path
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_mock_metadata():
    root = Path(__file__).parent.parent
    cache_path = root / "data" / "protein_metadata_cache.json"
    
    with open(root / "data" / "protein_lookup.pkl", "rb") as f:
        p_lu = pickle.load(f)
    
    # Just take top 50
    pids = list(p_lu.keys())[:50]
    
    # Make sure P00533 is there if possible, or just generate a mock one
    cache = {}
    for i, pid in enumerate(pids):
        cache[pid] = {
            "protein_name": f"Mock Protein {pid}",
            "gene_name": f"GENE_{pid[:4]}",
            "organism": "Homo sapiens",
            "sequence_length": 500 + i,
            "function": "Simulated function for testing.",
            "ensembl_id": f"ENSG{pid}"
        }
    
    cache["P00533"] = {
        "protein_name": "Epidermal growth factor receptor",
        "gene_name": "EGFR",
        "organism": "Homo sapiens",
        "sequence_length": 1210,
        "function": "Receptor tyrosine kinase binding to EGF.",
        "ensembl_id": "ENSG00000146648"
    }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    logger.info(f"Generated {len(cache)} mock protein metadata records to {cache_path}")

if __name__ == "__main__":
    generate_mock_metadata()
