# vector_api.py - API Server para consultas ao Vector Database e RAG
import os
import asyncio
import logging
import gc
import json
from datetime import datetime
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Optional

import psutil
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sentence_transformers import SentenceTransformer

# Importa os modelos Pydantic do arquivo centralizado
from models import SearchRequest, SearchResult, SearchResponse, RAGRequest, HealthResponse

# Importa nossas classes customizadas
from pdf_processor import HierarchicalVectorTape
from rag_pipeline import RAGPipeline

# --- Configuração de Logging ---
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Classe Principal do Servidor API ---

class VectorAPIServer:
    """Servidor API para consultas vetoriais otimizado para Raspberry Pi"""
    
    def __init__(self, vector_db_path: str, model_name: str):
        self.vector_db_path = vector_db_path
        self.model_name = model_name
        self.vector_db: Optional[HierarchicalVectorTape] = None
        self.model: Optional[SentenceTransformer] = None
        self.executor = ThreadPoolExecutor(max_workers=int(os.getenv("MAX_WORKERS", "2")))
        
        # Cache de consultas
        self.query_cache: Dict[str, Dict] = {}
        self.cache_max_size = int(os.getenv("CACHE_MAX_SIZE", "100"))
        self.cache_ttl = int(os.getenv("CACHE_TTL", "300"))
        
        # Estatísticas
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'startup_time': datetime.now().timestamp(),
        }
        
        # Otimizações
        self.gc_frequency = int(os.getenv("GC_FREQUENCY", "10"))

    async def initialize(self):
        """Inicialização assíncrona dos componentes principais."""
        logger.info("Inicializando Vector API Server...")
        
        logger.info(f"Carregando modelo de embedding: {self.model_name}")
        self.model = SentenceTransformer(
            self.model_name,
            device='cpu',
            cache_folder=os.getenv('MODEL_CACHE_DIR', './model_cache')
        )
        logger.info(f"Modelo carregado. Dimensão: {self.model.get_sentence_embedding_dimension()}")
        
        logger.info(f"Carregando vector database de: {self.vector_db_path}")
        Path(self.vector_db_path).mkdir(exist_ok=True)
        self.vector_db = HierarchicalVectorTape(
            self.vector_db_path, 
            self.model.get_sentence_embedding_dimension()
        )
        self.vector_db.load()
        gc.collect()
        logger.info("Inicialização concluída com sucesso!")

    def _get_system_stats(self) -> Dict:
        memory = psutil.virtual_memory()
        temp = None
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = round(float(f.read().strip()) / 1000.0, 1)
        except (FileNotFoundError, ValueError):
            pass
        
        return {
            'cpu_percent': round(psutil.cpu_percent(interval=0.1), 1),
            'memory_percent': round(memory.percent, 1),
            'temperature_c': temp
        }
    
    def _get_from_cache(self, key: str) -> Optional[List]:
        if key in self.query_cache:
            cached = self.query_cache[key]
            if (datetime.now().timestamp() - cached['timestamp']) < self.cache_ttl:
                return cached['result']
            else:
                del self.query_cache[key]
        return None

    def _add_to_cache(self, key: str, result: List):
        if len(self.query_cache) >= self.cache_max_size:
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
        
        self.query_cache[key] = {
            'result': result,
            'timestamp': datetime.now().timestamp()
        }

    async def search_vectors(self, request: SearchRequest) -> SearchResponse:
        start_time = datetime.now().timestamp()
        
        cache_key = f"{request.query}_{request.top_k}"
        cached_result = self._get_from_cache(cache_key) if request.use_cache else None

        if cached_result:
            self.stats['cache_hits'] += 1
            results_data = cached_result
            cache_hit = True
        else:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                self._search_sync,
                request
            )
            # Corrigido: Acessando metadados corretamente do registro
            results_data = [
                {
                    'id': r[0], 
                    'text': r[2], 
                    'similarity': round(r[1], 4), 
                    'metadata': r[3] 
                } for r in results
            ]
            if request.use_cache:
                self._add_to_cache(cache_key, results_data)
            cache_hit = False

        self.stats['total_queries'] += 1
        if self.stats['total_queries'] % self.gc_frequency == 0:
            gc.collect()

        search_time_ms = round((datetime.now().timestamp() - start_time) * 1000, 2)
        
        return SearchResponse(
            query=request.query,
            results=[SearchResult(**r) for r in results_data],
            search_time_ms=search_time_ms,
            system_stats=self._get_system_stats(),
            cache_hit=cache_hit
        )

    def _search_sync(self, request: SearchRequest) -> List:
        if not self.model or not self.vector_db:
            raise RuntimeError("Servidor não inicializado corretamente.")
        
        query_embedding = self.model.encode([request.query])[0]
        return self.vector_db.search(
            query_embedding=query_embedding,
            query_text=request.query,
            top_k=request.top_k
        )


# --- Inicialização Global e Gerenciador de Lifespan ---

api_server = VectorAPIServer(
    vector_db_path=os.getenv('VECTOR_DB_PATH', './vector_db'),
    model_name=os.getenv('MODEL_NAME', 'all-MiniLM-L6-v2')
)
rag_pipeline: Optional[RAGPipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline
    logger.info("Iniciando a aplicação...")
    await api_server.initialize()
    
    try:
        rag_pipeline = RAGPipeline(
            vector_server=api_server,
            ollama_model=os.getenv('OLLAMA_MODEL', 'gemma3'),
            ollama_host=os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        )
        logger.info("RAG Pipeline inicializado com sucesso.")
    except Exception as e:
        logger.critical(f"Falha ao inicializar o RAG Pipeline: {e}. O endpoint /rag/query não funcionará.")
    
    yield
    
    logger.info("Finalizando a aplicação...")
    if api_server.executor:
        api_server.executor.shutdown(wait=True)
    logger.info("Recursos liberados.")


# --- Configuração do App FastAPI ---

app = FastAPI(
    title="Edge RAG API",
    description="API otimizada para RAG com busca vetorial em Raspberry Pi",
    version="1.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Endpoints da API ---

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Edge RAG API is running. See /docs for documentation."}

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    uptime = datetime.now().timestamp() - api_server.stats['startup_time']
    return HealthResponse(
        status="healthy",
        vector_db_loaded=api_server.vector_db is not None,
        model_loaded=api_server.model is not None,
        rag_pipeline_ready=rag_pipeline is not None,
        total_vectors=api_server.vector_db.record_count if api_server.vector_db else 0,
        system_resources=api_server._get_system_stats(),
        uptime_seconds=round(uptime, 2)
    )

@app.post("/search", response_model=SearchResponse, tags=["Vector Search"])
async def search(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query não pode estar vazia")
    return await api_server.search_vectors(request)

@app.post("/rag/query", tags=["RAG"])
async def rag_query(request: RAGRequest):
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline não está disponível.")
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query não pode estar vazia")

    return StreamingResponse(
        rag_pipeline.generate_answer_stream(request.query, request.top_k),
        media_type="text/event-stream"
    )


# --- Execução do Servidor ---

if __name__ == "__main__":
    uvicorn.run(
        "vector_api:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        workers=1,
        reload=os.getenv("DEVELOPMENT", "false").lower() == "true"
    )