import numpy as np

class UncertaintyScorer:
    """Compute uncertainty metrics from model output probabilities."""
    
    @staticmethod
    def prediction_entropy(probs: np.ndarray) -> float:
        """Shannon entropy: H = -p0 * log(p0) - p1 * log(p1)
        Range: [0, ln(2) ≈ 0.693]. Higher = more uncertain.
        """
        # epsilon to avoid log(0)
        eps = 1e-9
        p = np.clip(probs, eps, 1.0 - eps)
        return float(-np.sum(p * np.log(p)))
    
    @staticmethod
    def confidence_score(probs: np.ndarray) -> float:
        """max(p0, p1). Range: [0.5, 1.0]. Higher = more confident."""
        return float(np.max(probs))
    
    @staticmethod
    def normalized_entropy(probs: np.ndarray) -> float:
        """Entropy / ln(2). Range: [0, 1]. Easier to interpret."""
        return float(UncertaintyScorer.prediction_entropy(probs) / np.log(2.0))
