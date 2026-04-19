import numpy as np

class EnsemblePredictor:
    """Compare predictions across 3 models (DrugBank, Davis, KIBA)."""
    
    def ensemble_analyze(self, predictions: list[dict]) -> dict:
        """Analyze cross-dataset consensus score.
        
        predictions: list of model outputs [{"dataset": "db", "prob": 0.8}, ...]
        """
        probs = [p["prob"] for p in predictions if p["prob"] is not None]
        if not probs:
            return {"consensus": "unavailable", "std": 0.0, "mean": 0.0}
            
        std = float(np.std(probs))
        mean = float(np.mean(probs))
        
        # Cross-dataset Consensus Score
        # "disagree" if std(probs) > 0.3 (adjustable threshold)
        consensus = "agree" if std < 0.15 else "partial" if std < 0.3 else "disagree"
        
        return {
            "consensus": consensus,
            "std": std,
            "mean": mean,
            "n_models": len(probs)
        }
