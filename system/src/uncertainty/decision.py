from dataclasses import dataclass
from typing import Literal, Optional
from ..config import settings
from .calibration import CalibrationAnalyzer

@dataclass
class DecisionResult:
    action: str          # "accept" | "review" | "reject" | "cannot_predict"
    confidence_level: str  # "high" | "medium" | "low" | "none"
    explanation: str
    threshold_used: float

class DecisionEngine:
    def __init__(self):
        self.analyzer = CalibrationAnalyzer()
        self.default_thresholds = {
            "high":   settings.ENTROPY_THRESHOLD_HIGH, 
            "medium": settings.ENTROPY_THRESHOLD_MEDIUM
        }

    def decide(self, probs, entropy, feature_info: dict, dataset: str = "drugbank") -> DecisionResult:
        """Apply decision rules based on model outputs and feature quality."""
        
        # 0. Load dynamic thresholds
        thresholds = self.analyzer.get_thresholds(dataset)
        high_max = thresholds.get("high", self.default_thresholds["high"])
        med_max = thresholds.get("medium", self.default_thresholds["medium"])

        # 1. Critical feature gate
        if feature_info.get("drug_3d_mode") == "not_available" or feature_info.get("drug_3d_mode") == "failed":
            return DecisionResult(
                action="cannot_predict",
                confidence_level="none",
                explanation="Drug 3D features not available. This drug is not in the training set and 3D feature computation requires the GEM pipeline.",
                threshold_used=0.0
            )

        if feature_info.get("protein_mode") == "not_found":
            return DecisionResult(
                action="cannot_predict",
                confidence_level="none",
                explanation="Protein not found in precomputed library.",
                threshold_used=0.0
            )

        # 2. Entropy-based level
        level = "low"
        if entropy <= high_max:
            level = "high"
        elif entropy <= med_max:
            level = "medium"

        # 3. Apply feature quality degradation
        final_level = level
        quality_msg = ""
        if feature_info.get("drug_2d_mode") == "proxy_rdkit":
            quality_msg = " (Downgraded due to proxy features)"
            if level == "high":
                final_level = "medium"
            elif level == "medium":
                final_level = "low"

        # 4. Map level to action
        action_map = {
            "high": "accept",
            "medium": "review",
            "low": "reject"
        }
        action = action_map.get(final_level, "reject")
        
        explanation = f"Confidence {final_level} based on prediction entropy ({entropy:.3f}){quality_msg}."
        
        return DecisionResult(
            action=action,
            confidence_level=final_level,
            explanation=explanation,
            threshold_used=high_max if final_level == "high" else med_max if final_level == "medium" else 0.693
        )
