import pickle
import logging
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger("dti.feature_store")

class FeatureStore:
    """
    Manages precomputed feature files for drugs and proteins.
    Loads lookup dictionaries from disk and providing search capabilities.
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.drug_2d_lookup: Dict[str, Any] = {}
        self.drug_3d_lookup: Dict[str, Any] = {}
        self.protein_lookup: Dict[str, Any] = {}
        self._loaded = False

    def load(self, force: bool = False):
        """Loads all lookup files from the data directory."""
        if self._loaded and not force:
            return

        try:
            self._load_drug_2d()
            self._load_drug_3d()
            self._load_protein()
            self._loaded = True
            logger.info("FeatureStore initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to load feature lookups: {str(e)}")
            raise

    def _load_drug_2d(self):
        path = self.data_dir / "drug_2d_lookup.pkl"
        if path.exists():
            with open(path, "rb") as f:
                self.drug_2d_lookup = pickle.load(f)
            logger.info(f"Loaded {len(self.drug_2d_lookup)} entries for drug 2D features.")
        else:
            logger.warning(f"Drug 2D lookup file not found: {path}")

    def _load_drug_3d(self):
        path = self.data_dir / "drug_3d_lookup.pkl"
        if path.exists():
            with open(path, "rb") as f:
                self.drug_3d_lookup = pickle.load(f)
            logger.info(f"Loaded {len(self.drug_3d_lookup)} entries for drug 3D features.")
        else:
            logger.warning(f"Drug 3D lookup file not found: {path}")

    def _load_protein(self):
        path = self.data_dir / "protein_lookup.pkl"
        if path.exists():
            with open(path, "rb") as f:
                self.protein_lookup = pickle.load(f)
            logger.info(f"Loaded {len(self.protein_lookup)} entries for protein features.")
        else:
            logger.warning(f"Protein lookup file not found: {path}")

    def get_drug_2d(self, smiles: str) -> Optional[Any]:
        return self.drug_2d_lookup.get(smiles)

    def get_drug_3d(self, smiles: str) -> Optional[Any]:
        return self.drug_3d_lookup.get(smiles)

    def get_protein(self, protein_id: str) -> Optional[Any]:
        return self.protein_lookup.get(protein_id)
    
    def list_known_proteins(self):
        return sorted(list(self.protein_lookup.keys()))
