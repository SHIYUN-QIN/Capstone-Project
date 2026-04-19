import torch
import numpy as np
from pathlib import Path
from .architecture import DTIModel
from ..config import settings

MODEL_PARAMS = dict(d2_dim=256, d3_dim=515, p_dim=1024,
                    drug_out=256, prot_out=256, hidden=512, dropout=0.2)
EXPECTED_PARAM_COUNT = 21_249_794
MAX_PROT_LEN = 800

class ModelManager:
    def __init__(self, model_dir: Path, device: str = "cpu"):
        self.model_dir = model_dir
        self.device = torch.device(device)
        self.models = {}

    def load_model(self, dataset: str) -> None:
        if dataset in self.models:
            return
        
        path = self.model_dir / f"best_model_{dataset}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Model weights for {dataset} not found at {path}")

        model = DTIModel(**MODEL_PARAMS)
        # Risk Fix #4: weights_only=True is safe for state_dicts
        sd = torch.load(path, map_location=self.device, weights_only=True)
        model.load_state_dict(sd, strict=True)
        
        # Verify parameter count
        actual = sum(p.numel() for p in model.parameters())
        assert actual == EXPECTED_PARAM_COUNT, f"Param count {actual} != {EXPECTED_PARAM_COUNT}"
        
        model.to(self.device)
        model.eval()
        self.models[dataset] = model

    def predict_single(self, d2: np.ndarray, d3: np.ndarray,
                       protein_emb: np.ndarray, dataset: str) -> dict:
        """Single pair inference with proper padding.
        
        Args:
            d2: (256,) float32
            d3: (515,) float32  
            protein_emb: (L, 1024) float32 \u2014 variable length, truncated to 800
        """
        if dataset not in self.models:
            self.load_model(dataset)
            
        # Truncate protein to MAX_PROT_LEN
        protein_emb = protein_emb[:MAX_PROT_LEN]
        L = protein_emb.shape[0]
        
        # Build batch-of-1 tensors
        d2_t = torch.from_numpy(d2).unsqueeze(0).float().to(self.device)    # [1, 256]
        d3_t = torch.from_numpy(d3).unsqueeze(0).float().to(self.device)    # [1, 515]
        p_t  = torch.from_numpy(protein_emb).unsqueeze(0).float().to(self.device)  # [1, L, 1024]
        mask = torch.ones(1, L, dtype=torch.bool).to(self.device)           # [1, L] all True
        
        with torch.no_grad():
            logits = self.models[dataset](d2_t, d3_t, p_t, mask)  # [1, 2]
        probs = torch.softmax(logits.float(), dim=1)
        return {"logits": logits[0].cpu().numpy(), "probs": probs[0].cpu().numpy()}
    
    def predict_batch(self, items: list[dict], dataset: str) -> list[dict]:
        """Batch inference with collate_fn padding."""
        if dataset not in self.models:
            self.load_model(dataset)

        # 1. Truncate all proteins to 800
        for item in items:
            item["protein_emb"] = item["protein_emb"][:MAX_PROT_LEN]
        
        # 2. Stack d2 \u2192 [B, 256], d3 \u2192 [B, 515]
        d2 = torch.stack([torch.from_numpy(x["d2"]) for x in items]).float().to(self.device)
        d3 = torch.stack([torch.from_numpy(x["d3"]) for x in items]).float().to(self.device)
        
        # 3. Pad proteins to max_L in THIS batch
        lens = [x["protein_emb"].shape[0] for x in items]
        max_L = max(lens)
        B = len(items)
        p_pad = torch.zeros(B, max_L, 1024).to(self.device)
        mask  = torch.zeros(B, max_L, dtype=torch.bool).to(self.device)
        for i, (emb, L) in enumerate(zip([x["protein_emb"] for x in items], lens)):
            p_pad[i, :L] = torch.from_numpy(emb).to(self.device)
            mask[i, :L]  = True
        
        with torch.no_grad():
            logits = self.models[dataset](d2, d3, p_pad.float(), mask)
        probs = torch.softmax(logits.float(), dim=1)
        return [{"logits": logits[i].cpu().numpy(), "probs": probs[i].cpu().numpy()} for i in range(B)]
