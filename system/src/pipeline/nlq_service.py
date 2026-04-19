import faiss
import json
import logging
from pathlib import Path
from src.db.results_store import ResultsStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLQService:
    def __init__(self, index_path="data/nlq_index.faiss", docs_path="data/nlq_documents.json", model_name="all-MiniLM-L6-v2"):
        root = Path(__file__).parent.parent.parent
        self.idx_path = root / index_path
        self.docs_path = root / docs_path
        self.documents = []
        self.index = None
        self.model = None

        if self.docs_path.exists() and self.idx_path.exists():
            from sentence_transformers import SentenceTransformer
            self.documents = json.load(open(self.docs_path, "r", encoding="utf-8"))
            self.index = faiss.read_index(str(self.idx_path))
            self.model = SentenceTransformer(model_name)
    
    def query(self, question: str, top_k: int = 10) -> dict:
        import time
        start = time.time()
        
        if not self.model or not self.index:
            return {"query": question, "results": [], "summary": "Index not found.", "search_time_ms": 0}

        q_emb = self.model.encode([question])
        scores, I = self.index.search(q_emb, k=top_k)
        
        results = []
        for i, idx in enumerate(I[0]):
            if idx == -1: continue
            doc = self.documents[idx]
            match = doc["meta"]
            match["relevance_score"] = float(scores[0][i])
            results.append(match)
            
        ms = int((time.time() - start) * 1000)
        summary = f"Found {len(results)} matches for your query."
        return {"query": question, "results": results, "summary": summary, "search_time_ms": ms}
