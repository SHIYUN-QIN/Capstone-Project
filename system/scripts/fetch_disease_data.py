import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_mock_disease_data():
    root = Path(__file__).parent.parent
    p_cache = root / "data" / "protein_metadata_cache.json"
    
    with open(p_cache, "r", encoding="utf-8") as f:
        proteins = json.load(f)

    disease_target = {}
    disease_index = {}

    # Mock cancer data
    efo_cancer = "EFO_0000311"
    disease_index[efo_cancer] = {"disease_name": "Cancer", "targets": []}

    for pid, pdata in proteins.items():
        disease_target[pid] = {
            "ensembl_id": pdata.get("ensembl_id", ""),
            "diseases": [
                {"efo_id": efo_cancer, "disease_name": "Cancer", "score": 0.85}
            ]
        }
        disease_index[efo_cancer]["targets"].append(pid)

    with open(root / "data" / "disease_target_cache.json", "w") as f:
        json.dump(disease_target, f, indent=2)
    with open(root / "data" / "disease_protein_index.json", "w") as f:
        json.dump(disease_index, f, indent=2)

    logger.info("Generated mock disease data.")

if __name__ == "__main__":
    generate_mock_disease_data()
