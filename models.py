# models.py - Definições de modelos Pydantic para a API

from pydantic import BaseModel
from typing import List, Dict, Optional

# --- Modelos para a API de Busca Vetorial ---

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    use_cache: Optional[bool] = True

class SearchResult(BaseModel):
    id: int
    text: str
    similarity: float
    metadata: Dict

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    search_time_ms: float
    system_stats: Dict
    cache_hit: bool

# --- Modelo para a API RAG ---

class RAGRequest(BaseModel):
    query: str
    top_k: Optional[int] = 6

# --- Modelo para Health Check ---

class HealthResponse(BaseModel):
    status: str
    vector_db_loaded: bool
    model_loaded: bool
    rag_pipeline_ready: bool
    total_vectors: int
    system_resources: Dict
    uptime_seconds: float