from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='DTI_')

    MODEL_DIR: Path = Path("weights")
    DATA_DIR: Path = Path("data")
    STATIC_DIR: Path = Path("static")
    AVAILABLE_DATASETS: List[str] = ["drugbank", "davis", "kiba"]
    DEFAULT_DATASET: str = "drugbank"
    MAX_PROT_LEN: int = 800
    D2_DIM: int = 256
    D3_DIM: int = 515
    P_EMB_DIM: int = 1024
    DEVICE: str = "cpu"  # "cuda" if GPU available
    BATCH_SIZE: int = 8   # Small for CPU; prevents OOM on free tier
    
    # Uncertainty Thresholds (Initial defaults, will be calibrated)
    ENTROPY_THRESHOLD_HIGH: float = 0.15
    ENTROPY_THRESHOLD_MEDIUM: float = 0.40
    CONFIDENCE_THRESHOLD_HIGH: float = 0.90
    CONFIDENCE_THRESHOLD_MEDIUM: float = 0.70

settings = Settings()
