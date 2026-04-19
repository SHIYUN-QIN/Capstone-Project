from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class DTIResult(BaseModel):
    dataset: str
    interaction_prob: float
    entropy: float
    decision: str

class PairResponse(BaseModel):
    drug_info: dict
    protein_info: dict
    predictions: List[DTIResult]

class AttentionResponse(BaseModel):
    residue_indices: List[int]
    attention_weights: List[float]
    protein_sequence: str = ""

class DrugListResponse(BaseModel):
    drugs: List[dict]
    total: int

class ProteinListResponse(BaseModel):
    proteins: List[dict]
    total: int

class DiseaseSearchResponse(BaseModel):
    diseases: List[dict]

class DiseaseDetailResponse(BaseModel):
    disease_name: str
    targets: List[dict]

class NLQRequest(BaseModel):
    query: str
    top_k: int = 10

class NLQResponse(BaseModel):
    query: str
    results: List[dict]
    summary: str
    search_time_ms: int

class HealthResponse(BaseModel):
    status: str
    version: str
