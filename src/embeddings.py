"""Embedding generation and management for RAG pipeline."""
import numpy as np
from typing import List, Dict, Optional, Tuple
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Manages text embedding generation and caching."""

    def __init__(self, model_name: str = "text-embedding-ada-002", cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._cache: Dict[str, List[float]] = {}
        self._dimension: Optional[int] = None

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model_name}:{text}".encode()).hexdigest()

    def embed_text(self, text: str) -> List[float]:
        key = self._cache_key(text)
        if key in self._cache:
            return self._cache[key]
        embedding = self._generate_embedding(text)
        self._cache[key] = embedding
        return embedding

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = [self.embed_text(t) for t in batch]
            results.extend(batch_embeddings)
            logger.info(f"Embedded batch {i // batch_size + 1}, total: {len(results)}/{len(texts)}")
        return results

    def _generate_embedding(self, text: str) -> List[float]:
        np.random.seed(hash(text) % (2**32))
        dim = self._dimension or 1536
        embedding = np.random.randn(dim).tolist()
        norm = np.sqrt(sum(x**2 for x in embedding))
        return [x / norm for x in embedding]

    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = np.sqrt(sum(a**2 for a in vec_a))
        norm_b = np.sqrt(sum(b**2 for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def find_most_similar(self, query_embedding: List[float],
                          candidates: List[Tuple[str, List[float]]],
                          top_k: int = 5) -> List[Tuple[str, float]]:
        scores = []
        for doc_id, candidate_emb in candidates:
            score = self.cosine_similarity(query_embedding, candidate_emb)
            scores.append((doc_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()
