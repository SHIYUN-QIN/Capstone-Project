import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from .feature_store import FeatureStore

def canonical_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)  # RDKit canonical form

class DrugFeaturePipeline:
    def __init__(self, feature_store: FeatureStore):
        self.store = feature_store
    
    def extract(self, smiles: str) -> dict:
        canon = canonical_smiles(smiles)
        if canon is None:
            return {"d2": None, "d3": None, "d2_mode": "failed", "d3_mode": "failed",
                    "error": "Invalid SMILES string"}
        
        # D2: precomputed MG-BERT or Morgan FP proxy
        d2 = self.store.get_drug_2d(canon)
        if d2 is not None:
            d2_mode = "precomputed"
        else:
            # Fallback to Morgan FP as proxy for unknown drugs
            mol = Chem.MolFromSmiles(canon)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256)
            d2, d2_mode = np.array(fp, np.float32), "proxy_rdkit"
        
        # D3: precomputed ONLY -- refuse if missing
        d3 = self.store.get_drug_3d(canon)
        if d3 is not None:
            d3_mode = "precomputed"
        else:
            d3, d3_mode = None, "not_available"
        
        return {"d2": d2, "d3": d3, "d2_mode": d2_mode, "d3_mode": d3_mode}
