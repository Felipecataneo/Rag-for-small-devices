# pdf_processor.py - Sistema de processamento e indexação de PDFs
import os
import mmap
import struct
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import PyPDF2
from sentence_transformers import SentenceTransformer
from bloom_filter2 import BloomFilter
import json
import logging
from tqdm import tqdm
import re
import pickle
import base64

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VectorRecord:
    """Estrutura de um registro vetorial"""
    id: int
    text: str
    embedding: np.ndarray
    metadata: Dict
    fingerprint: str
    
class VectorDNA:
    """Sistema de fingerprinting para embeddings"""
    
    @staticmethod
    def generate_fingerprint(embedding: np.ndarray, text: str) -> str:
        """Gera fingerprint único baseado no embedding e texto"""
        mean_val = np.mean(embedding)
        std_val = np.std(embedding)
        max_val = np.max(embedding)
        min_val = np.min(embedding)
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        stats_str = f"{mean_val:.4f}_{std_val:.4f}_{max_val:.4f}_{min_val:.4f}_{text_hash}"
        return hashlib.sha256(stats_str.encode()).hexdigest()[:16]
    
    @staticmethod
    def similarity_by_fingerprint(fp1: str, fp2: str) -> float:
        """Estimativa rápida de similaridade baseada em fingerprints"""
        diff = sum(c1 != c2 for c1, c2 in zip(fp1, fp2))
        return 1.0 - (diff / len(fp1))

class AdaptiveQuantizer:
    """Sistema de quantização adaptativa para reduzir tamanho dos vetores"""
    
    def __init__(self, target_bits: int = 8):
        self.target_bits = target_bits
        self.scale_factor = None
        self.offset = None
        
    def fit(self, embeddings: List[np.ndarray]):
        all_vals = np.concatenate(embeddings)
        self.offset = np.min(all_vals)
        self.scale_factor = (np.max(all_vals) - self.offset) / (2**self.target_bits - 1)
        logger.info(f"Quantização: offset={self.offset:.4f}, scale={self.scale_factor:.6f}")
    
    def quantize(self, embedding: np.ndarray) -> np.ndarray:
        if self.scale_factor is None:
            raise ValueError("Quantizer não foi ajustado. Execute fit() primeiro.")
        quantized = ((embedding - self.offset) / self.scale_factor).astype(np.uint8)
        return quantized
    
    def dequantize(self, quantized: np.ndarray) -> np.ndarray:
        return quantized.astype(np.float32) * self.scale_factor + self.offset

class HierarchicalVectorTape:
    """Sistema principal de armazenamento vetorial hierárquico"""
    
    def __init__(self, base_path: str, embedding_dim: int = 384, use_quantization: bool = True):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        self.embedding_dim = embedding_dim
        self.use_quantization = use_quantization
        
        self.tape_file = self.base_path / "vectors.tape"
        self.meta_file = self.base_path / "vectors.meta"
        self.bloom_file = self.base_path / "vectors.bloom"  # Changed to .bloom instead of .bloom.json
        self.config_file = self.base_path / "config.json"
        
        self.quantizer = AdaptiveQuantizer() if use_quantization else None
        self.bloom_filter: Optional[BloomFilter] = None
        self.bloom_max_elements = 0
        self.bloom_error_rate = 0.0
        self.record_count = 0
        self.fingerprint_cache = {}
        
        # Carrega apenas a configuração básica no __init__
        self._load_config()

    def load(self):
        """Carrega todos os componentes do banco de dados do disco para a memória."""
        logger.info("Carregando componentes do HierarchicalVectorTape...")
        self._load_config()
        self._load_bloom_filter()
        self._load_fingerprints()
        if self.bloom_filter is None:
            logger.error("FALHA CRÍTICA: Filtro de Bloom não pôde ser carregado e é None.")
        else:
            logger.info("Componentes carregados com sucesso.")

    def _load_config(self):
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.record_count = config.get('record_count', 0)
                self.embedding_dim = config.get('embedding_dim', self.embedding_dim)
                self.bloom_max_elements = config.get('bloom_max_elements', 0)
                self.bloom_error_rate = config.get('bloom_error_rate', 0.1)
                self.use_quantization = config.get('use_quantization', True)

                # --- ADICIONAR ESTA PARTE ---
                # Carrega os parâmetros do quantizador, se existirem
                if self.use_quantization and self.quantizer:
                    self.quantizer.offset = config.get('quantizer_offset')
                    self.quantizer.scale_factor = config.get('quantizer_scale')
                    if self.quantizer.offset is not None:
                        logger.info(f"Parâmetros de quantização carregados: offset={self.quantizer.offset:.4f}, scale={self.quantizer.scale_factor:.6f}")
                # -------------------------
        else:
            self._save_config()
    
    def _save_config(self):
        config = {
            'record_count': self.record_count, 
            'embedding_dim': self.embedding_dim, 
            'use_quantization': self.use_quantization,
            'bloom_max_elements': self.bloom_max_elements,
            'bloom_error_rate': self.bloom_error_rate,
            
            # --- MODIFICAR ESTAS LINHAS ---
            # Converte os valores para float nativo do Python antes de salvar
            'quantizer_offset': float(self.quantizer.offset) if self.quantizer and self.quantizer.offset is not None else None,
            'quantizer_scale': float(self.quantizer.scale_factor) if self.quantizer and self.quantizer.scale_factor is not None else None,
            # ----------------------------
        }
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)


    def _load_bloom_filter(self):
        """Carrega o objeto BloomFilter inteiro do disco usando pickle."""
        if not self.bloom_file.exists():
            logger.warning("Arquivo de filtro de Bloom não encontrado. Será criado um novo ao adicionar registros.")
            self.bloom_filter = None
            return

        try:
            with open(self.bloom_file, 'rb') as f:
                # Carrega o objeto inteiro diretamente. É a forma mais simples e robusta.
                self.bloom_filter = pickle.load(f)
            
            # Atualiza os parâmetros na classe a partir do objeto carregado
            if hasattr(self.bloom_filter, 'max_elements'):
                self.bloom_max_elements = self.bloom_filter.max_elements
                self.bloom_error_rate = self.bloom_filter.error_rate
            
            logger.info(f"Filtro de Bloom carregado com sucesso de {self.bloom_file}")
            
        except Exception as e:
            logger.error(f"Falha crítica ao carregar o filtro de Bloom com pickle de {self.bloom_file}: {e}")
            self.bloom_filter = None


    def _save_bloom_filter(self):
        """Salva o objeto BloomFilter inteiro no disco usando pickle."""
        if not self.bloom_filter:
            return

        try:
            with open(self.bloom_file, 'wb') as f:
                # Salva o objeto inteiro diretamente.
                pickle.dump(self.bloom_filter, f)
            logger.info(f"Filtro de Bloom salvo com sucesso em {self.bloom_file}")

        except Exception as e:
            logger.error(f"Não foi possível salvar o filtro de Bloom com pickle: {e}")
    

    def _load_fingerprints(self):
        if self.meta_file.exists():
            with open(self.meta_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        meta = json.loads(line.strip())
                        if 'id' in meta and 'fingerprint' in meta:
                            self.fingerprint_cache[meta['id']] = meta['fingerprint']
                    except json.JSONDecodeError:
                        continue # Pula linhas malformadas
            logger.info(f"Cache de {len(self.fingerprint_cache)} fingerprints carregado.")
    
    def _init_bloom_filter(self, expected_items: int, error_rate: float = 0.1):
        """Inicializa o filtro de Bloom usando 'max_elements' (API correta para bloom-filter2)."""
        logger.info(f"Inicializando Bloom Filter para ~{expected_items} itens.")
        
        # O nome do parâmetro correto para a biblioteca instalada é 'max_elements'
        self.bloom_filter = BloomFilter(max_elements=expected_items, error_rate=error_rate)
        
        # Guarda os parâmetros para referência
        self.bloom_max_elements = expected_items
        self.bloom_error_rate = error_rate
    
    def _tokenize_text(self, text: str) -> List[str]:
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        tokens = [t for t in tokens if 2 <= len(t) <= 20]
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
        return tokens + bigrams
    
    def add_records(self, records: List[VectorRecord]):
        if not records:
            return
        
        logger.info(f"Adicionando {len(records)} registros...")
        
        if self.bloom_filter is None:
            estimated_total = self.record_count + len(records)
            # Aqui, passamos a estimativa para 'expected_items'
            self._init_bloom_filter(expected_items=estimated_total * 2)
        
        if self.quantizer and self.record_count == 0:
            embeddings = [r.embedding for r in records]
            self.quantizer.fit(embeddings)
        
        with open(self.tape_file, 'ab') as tape_f, open(self.meta_file, 'a', encoding='utf-8') as meta_f:
            for record in tqdm(records, desc="Indexando"):
                self._write_record(tape_f, meta_f, record)
        
        self._save_config()
        self._save_bloom_filter()
        logger.info(f"Total de registros: {self.record_count}")

    def _write_record(self, tape_f, meta_f, record: VectorRecord):
        if self.quantizer:
            embedding_data = self.quantizer.quantize(record.embedding)
            data_type = 'uint8'
        else:
            embedding_data = record.embedding.astype(np.float32)
            data_type = 'float32'
        
        tape_f.write(embedding_data.tobytes())
        
        tokens = self._tokenize_text(record.text)
        if self.bloom_filter:
            for token in tokens:
                self.bloom_filter.add(token)
        
        self.fingerprint_cache[record.id] = record.fingerprint
        
        meta_record = {
            'id': record.id,
            'text': record.text,
            'metadata': record.metadata,
            'fingerprint': record.fingerprint,
            'tokens': tokens[:50],
            'data_type': data_type,
            'offset': self.record_count * len(embedding_data.tobytes())
        }
        meta_f.write(json.dumps(meta_record, ensure_ascii=False) + '\n')
        
        self.record_count += 1
    
    def search(self, query_embedding: np.ndarray, query_text: str, top_k: int = 10) -> List[Tuple[int, float, str]]:
        if self.record_count == 0:
            return []
        
        logger.info(f"Buscando top-{top_k} para query: '{query_text[:50]}...'")
        
        query_tokens = self._tokenize_text(query_text)
        candidate_ids = self._bloom_filter_candidates(query_tokens)
        
        if not candidate_ids:
            logger.warning("Nenhum candidato encontrado pelo Bloom Filter")
            return []
        
        logger.info(f"Bloom Filter encontrou {len(candidate_ids)} candidatos")
        
        query_fingerprint = VectorDNA.generate_fingerprint(query_embedding, query_text)
        refined_candidates = self._fingerprint_filter(candidate_ids, query_fingerprint)
        
        results = self._vector_similarity_search(query_embedding, refined_candidates, top_k)
        
        logger.info(f"Retornando {len(results)} resultados")
        return results
    
    def _bloom_filter_candidates(self, query_tokens: List[str]) -> List[int]:
        if not self.bloom_filter:
            logger.error("Tentativa de busca com Filtro de Bloom não inicializado.")
            # Fallback: retorna todos os IDs se o filtro não estiver carregado
            return list(range(self.record_count))

        candidates = set()
        with open(self.meta_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    meta = json.loads(line.strip())
                    for token in query_tokens:
                        if token in self.bloom_filter:
                            if any(token in doc_token for doc_token in meta.get('tokens', [])):
                                candidates.add(meta['id'])
                                break
                except json.JSONDecodeError:
                    continue
        return list(candidates)
    
    def _fingerprint_filter(self, candidate_ids: List[int], query_fingerprint: str, threshold: float = 0.3) -> List[int]:
        if not self.fingerprint_cache:
            return candidate_ids
        
        refined = []
        for cand_id in candidate_ids:
            if cand_id in self.fingerprint_cache:
                cand_fp = self.fingerprint_cache[cand_id]
                fp_similarity = VectorDNA.similarity_by_fingerprint(query_fingerprint, cand_fp)
                if fp_similarity > threshold:
                    refined.append(cand_id)
        
        return refined if refined else candidate_ids[:100] # Limita o fallback

    def _vector_similarity_search(self, query_embedding: np.ndarray, candidate_ids: List[int], top_k: int) -> List[Tuple[int, float, str]]:
        results = []
        meta_records_by_id = {}
        with open(self.meta_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    meta = json.loads(line.strip())
                    meta_records_by_id[meta['id']] = meta
                except json.JSONDecodeError:
                    continue

        with open(self.tape_file, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for cand_id in candidate_ids:
                if cand_id not in meta_records_by_id:
                    continue
                
                meta = meta_records_by_id[cand_id]
                offset = meta['offset']
                
                mm.seek(offset)
                if meta.get('data_type') == 'uint8':
                    cand_data = np.frombuffer(mm.read(self.embedding_dim), dtype=np.uint8)
                    cand_embedding = self.quantizer.dequantize(cand_data) if self.quantizer else cand_data.astype(np.float32)
                else:
                    bytes_per_record = self.embedding_dim * 4
                    cand_data = np.frombuffer(mm.read(bytes_per_record), dtype=np.float32)
                    cand_embedding = cand_data
                
                similarity = self._cosine_similarity(query_embedding, cand_embedding)
                results.append((meta['id'], similarity, meta['text'], meta['metadata']))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b) if norm_a != 0 and norm_b != 0 else 0.0

class PDFProcessor:
    """Processador de PDFs para extração e indexação de conteúdo"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Modelo carregado: {model_name} (dim={self.embedding_dim})")
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        chunks = []
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                full_text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text() or ""
                    full_text += f"\n[Page {page_num + 1}]\n{page_text}"
                
                chunk_size = 500
                words = full_text.split()
                
                for i in range(0, len(words), chunk_size):
                    chunk_words = words[i:i + chunk_size]
                    chunk_text = " ".join(chunk_words)
                    chunks.append({
                        'text': chunk_text,
                        'metadata': {'source': os.path.basename(pdf_path), 'chunk_id': len(chunks)}
                    })
        except Exception as e:
            logger.error(f"Erro ao processar PDF {pdf_path}: {e}")
        return chunks
    
    def process_pdf_folder(self, folder_path: str, vector_db_path: str):
        folder = Path(folder_path)
        pdf_files = list(folder.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"Nenhum PDF encontrado em {folder_path}")
            return
        
        logger.info(f"Processando {len(pdf_files)} arquivos PDF...")
        vector_db = HierarchicalVectorTape(vector_db_path, self.embedding_dim)
        
        all_records = []
        record_id = 0
        
        for pdf_file in tqdm(pdf_files, desc="Processando PDFs"):
            logger.info(f"Processando: {pdf_file.name}")
            chunks = self.extract_text_from_pdf(str(pdf_file))
            
            if not chunks:
                continue
            
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.model.encode(texts, show_progress_bar=False, batch_size=32)
            
            for chunk, embedding in zip(chunks, embeddings):
                fingerprint = VectorDNA.generate_fingerprint(embedding, chunk['text'])
                record = VectorRecord(
                    id=record_id,
                    text=chunk['text'],
                    embedding=embedding,
                    metadata=chunk['metadata'],
                    fingerprint=fingerprint
                )
                all_records.append(record)
                record_id += 1
        
        if all_records:
            vector_db.add_records(all_records)
        
        logger.info("Processamento de PDFs concluído!")

if __name__ == "__main__":
    import shutil
    db_path = "./vector_db"
    
    # Limpa o banco de dados antigo antes de reprocessar
    if os.path.exists(db_path):
        logger.info(f"Limpando banco de dados antigo em {db_path}...")
        shutil.rmtree(db_path)
        
    processor = PDFProcessor()
    
    processor.process_pdf_folder(
        folder_path="./pdfs",
        vector_db_path=db_path
    )
    
    print("\nProcessamento concluído! O Vector Database foi criado/atualizado em './vector_db'")