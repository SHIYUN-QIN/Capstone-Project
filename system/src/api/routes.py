from fastapi import APIRouter, Depends, HTTPException, Query, Response
from typing import List
import csv
from io import StringIO
from .schemas import *
from ..db.results_store import ResultsStore
from ..pipeline.protein_metadata import ProteinMetadataService
from ..pipeline.disease_service import DiseaseService
from ..pipeline.nlq_service import NLQService
from pathlib import Path
import os

router = APIRouter()

# Setup dependencies (global for fast access)
root = Path(__file__).parent.parent.parent
db_path = root / "data" / "dti_results.db"
results_store = ResultsStore(str(db_path))
metadata_service = ProteinMetadataService()
disease_service = DiseaseService()
nlq_service = NLQService()

@router.get("/pair/{drug_id}/{protein_id}", response_model=PairResponse)
def get_pair(drug_id: str, protein_id: str):
    # Lookup drug info
    drugs = results_store.list_drugs(search=drug_id)
    drug_info = drugs[0] if drugs else {"drug_id": drug_id, "drug_name": "Unknown"}
    
    # Lookup protein info
    protein_info = metadata_service.get(protein_id) or {"protein_id": protein_id}
    
    # Get predictions
    preds = results_store.get_pair_all_datasets(drug_info.get('smiles', ''), protein_id)
    
    return {
        "drug_info": drug_info,
        "protein_info": protein_info,
        "predictions": [
            {
                "dataset": p["dataset"],
                "interaction_prob": p["interaction_prob"],
                "entropy": p["entropy"],
                "decision": p["decision"]
            } for p in preds
        ]
    }

@router.get("/pair/{drug_id}/{protein_id}/attention", response_model=AttentionResponse)
def get_attention(drug_id: str, protein_id: str, dataset: str = Query(..., description="Dataset name e.g. drugbank")):
    drugs = results_store.list_drugs(search=drug_id)
    if not drugs: raise HTTPException(404, "Drug not found")
    smiles = drugs[0]['smiles']
    
    attn = results_store.get_pair_attention(smiles, protein_id, dataset)
    if not attn: raise HTTPException(404, "Attention map not found or not accepted.")
    
    return {
        "residue_indices": list(range(1, len(attn) + 1)),
        "attention_weights": attn,
        "protein_sequence": ""
    }

@router.get("/drugs", response_model=DrugListResponse)
def get_drugs(search: str = None, limit: int = 20, offset: int = 0):
    drugs = results_store.list_drugs(limit=limit, offset=offset, search=search)
    return {"drugs": drugs, "total": len(drugs)}

@router.get("/proteins", response_model=ProteinListResponse)
def get_proteins(search: str = None, limit: int = 20, offset: int = 0):
    if search:
        m_results = metadata_service.search(search)
        return {"proteins": m_results[offset:offset+limit], "total": len(m_results)}
    pr = results_store.list_proteins(limit=limit, offset=offset)
    res = []
    for p in pr:
        res.append(metadata_service.get(p["protein_id"]) or p)
    return {"proteins": res, "total": 9999} # Mock total

@router.get("/protein/{protein_id}/top_drugs")
def get_protein_top_drugs(protein_id: str, dataset: str = "drugbank", limit: int = 20):
    pairs = results_store.get_protein_drugs(protein_id, dataset, top_k=limit)
    return {"pairs": pairs}

@router.get("/drug/{drug_id}/top_proteins")
def get_drug_top_proteins(drug_id: str, dataset: str = "drugbank", limit: int = 20):
    drugs = results_store.list_drugs(search=drug_id)
    if not drugs: raise HTTPException(404, "Drug not found")
    smiles = drugs[0]['smiles']
    pairs = results_store.get_drug_proteins(smiles, dataset, top_k=limit)
    return {"pairs": pairs}

@router.get("/valid_options")
def get_valid_options():
    res = results_store.get_valid_options()
    for p in res["proteins"]:
        info = metadata_service.get(p["id"])
        if info and "protein_name" in info:
            p["label"] = f'{info["protein_name"]} ({p["id"]})'
    return res

@router.get("/stats")
def get_stats():
    return results_store.get_statistics()

@router.get("/disease/search", response_model=DiseaseSearchResponse)
def search_disease(q: str):
    res = disease_service.search_diseases(q)
    return {"diseases": res}

@router.get("/disease/{efo_id}", response_model=DiseaseDetailResponse)
def get_disease(efo_id: str):
    res = disease_service.get_disease_targets(efo_id)
    if not res: raise HTTPException(404, "Disease not found")
    return res

@router.get("/disease/{efo_id}/drug_candidates")
def get_disease_drugs(efo_id: str):
    res = disease_service.get_disease_drug_candidates(efo_id, results_store)
    return {"candidates": res}

@router.post("/nlq", response_model=NLQResponse)
def run_nlq(req: NLQRequest):
    return nlq_service.query(req.query, req.top_k)

@router.get("/protein/{protein_id}/diseases")
def get_protein_diseases(protein_id: str):
    return {"diseases": disease_service.get_protein_diseases(protein_id)}

@router.get("/export/pair/{drug_id}/{protein_id}")
def export_pair(drug_id: str, protein_id: str, format: str = Query("json", description="json or csv")):
    preds = results_store.get_pair_all_datasets(drug_id, protein_id)
    if not preds: # perhaps drug_id was NOT smiles but SHA256 ID, let's auto-fetch smiles
        drugs = results_store.list_drugs(search=drug_id)
        if drugs:
            preds = results_store.get_pair_all_datasets(drugs[0]['smiles'], protein_id)
    
    if format.lower() == "csv":
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["dataset", "interaction_prob", "entropy", "decision"])
        for p in preds:
            writer.writerow([p["dataset"], p["interaction_prob"], p["entropy"], p["decision"]])
        return Response(content=output.getvalue(), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=pair_{drug_id}_{protein_id}.csv"})
    
    return {"drug_id": drug_id, "protein_id": protein_id, "predictions": preds}

@router.get("/export/protein/{protein_id}")
def export_protein(protein_id: str, format: str = Query("csv"), dataset: str = Query("drugbank")):
    pairs = results_store.get_protein_drugs(protein_id, dataset, top_k=5000) # get all or many
    
    if format.lower() == "csv":
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["drug_id", "drug_name", "interaction_prob", "entropy", "decision"])
        for p in pairs:
            writer.writerow([p["drug_id"], p.get("drug_name", ""), p["interaction_prob"], p["entropy"], p["decision"]])
        return Response(content=output.getvalue(), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=protein_{protein_id}_{dataset}.csv"})
    
    return {"pairs": pairs}

@router.get("/health", response_model=HealthResponse)
def health_check():
    return {"status": "healthy", "version": "2.0-static"}
