# rag_pipeline.py - Orquestrador do pipeline RAG
import ollama
import logging
import json
from typing import List, Dict, Any, TYPE_CHECKING

# ==============================================================================
# IMPORTAÇÃO CORRIGIDA: Importamos SearchRequest do nosso novo arquivo 'models.py'
# Esta é a mudança CRÍTICA que quebra o ciclo de importação.
# ==============================================================================
from models import SearchRequest

# Esta parte permanece para ajudar o editor de código, mas não roda de verdade
if TYPE_CHECKING:
    from vector_api import VectorAPIServer

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Orquestra o processo de Retrieval-Augmented Generation.
    1. Recebe uma query.
    2. Busca por contexto relevante no Vector Database.
    3. Constrói um prompt com o contexto.
    4. Envia o prompt para um LLM (Ollama/Gemma) para gerar uma resposta.
    """
    def __init__(self, 
                 vector_server: 'VectorAPIServer', # A referência com aspas permanece
                 ollama_model: str = "gemma3", 
                 ollama_host: str = "http://localhost:11434"):
        self.vector_server = vector_server
        self.ollama_model = ollama_model
        self.client = ollama.Client(host=ollama_host)
        logger.info(f"Pipeline RAG inicializado com o modelo Ollama: {self.ollama_model}")

        try:
            self.client.show(self.ollama_model)
        except ollama.ResponseError as e:
            logger.error(f"O modelo '{self.ollama_model}' não foi encontrado no Ollama. "
                         f"Por favor, execute 'ollama pull {self.ollama_model}' no terminal. Erro: {e}")
            raise

    def _construct_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        if not context_chunks:
            return query

        context_str = "\n\n---\n\n".join([chunk['text'] for chunk in context_chunks])
        
        prompt_template = f"""
        Você é um assistente de pesquisa que responde perguntas baseando-se estritamente em um texto fornecido. Siga estas regras rigorosamente:
        1. Responda à "PERGUNTA" usando APENAS as informações encontradas no "CONTEXTO".
        2. Sua resposta DEVE ser em Português.
        3. Se a resposta não estiver explicitamente no "CONTEXTO", responda exatamente com: "A informação não foi encontrada nos documentos fornecidos."
        4. Não adicione nenhuma informação que não esteja no texto. Seja direto e conciso.
        ##IMPORTANTE:resposta DEVE ser em Português Brasil.

        ### CONTEXTO
        {context_str}

        ### PERGUNTA
        {query}

        ### RESPOSTA
        """
        return prompt_template.strip()

    async def generate_answer_stream(self, query: str, top_k: int = 3):
        logger.info(f"RAG: Recebida nova query: '{query[:50]}...'")
        
        search_request = SearchRequest(query=query, top_k=top_k)
        search_response = await self.vector_server.search_vectors(search_request)
        
        retrieved_chunks = [result.dict() for result in search_response.results]
        
        prompt = self._construct_prompt(query, retrieved_chunks)
        
        sources = [
            {
                "source": chunk['metadata'].get('source', 'N/A'),
                "chunk_id": chunk['metadata'].get('chunk_id', -1),
                "similarity": chunk['similarity']
            }
            for chunk in retrieved_chunks if chunk['metadata'] # Adicionado para segurança
        ]
        
        logger.info(f"RAG: Gerando resposta com {len(retrieved_chunks)} pedaços de contexto.")
        
        try:
            stream = self.client.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': prompt}],
                stream=True,
                options={'temperature': 0.2, 'num_ctx': 2048}
            )

            metadata_package = {"type": "sources", "data": sources}
            yield f"data: {json.dumps(metadata_package)}\n\n"

            for chunk in stream:
                if 'content' in chunk['message']:
                    token = chunk['message']['content']
                    message_package = {"type": "llm_token", "data": token}
                    yield f"data: {json.dumps(message_package)}\n\n"
        
        except Exception as e:
            logger.error(f"RAG: Erro ao comunicar com o Ollama: {e}")
            error_package = {"type": "error", "data": "Ocorreu um erro ao gerar a resposta."}
            yield f"data: {json.dumps(error_package)}\n\n"