import numpy as np
from .feature_store import FeatureStore

class ProteinFeaturePipeline:
    def __init__(self, feature_store: FeatureStore):
        self.store = feature_store
    
    def get_embedding(self, protein_id: str) -> tuple[np.ndarray | None, str]:
        emb = self.store.get_protein(protein_id)
        if emb is not None:
            return emb.astype(np.float32), "precomputed"
        return None, "not_found"
    
    def list_known_proteins(self) -> list[str]:
        """Return all available protein IDs for the dropdown."""
        return self.store.list_known_proteins()
