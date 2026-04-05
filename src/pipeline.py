"""End-to-end RAG pipeline orchestration."""
from typing import List, Dict, Optional
import time
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the RAG pipeline."""
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    similarity_threshold: float = 0.7
    max_context_tokens: int = 3000
    model_name: str = "gpt-4"
    temperature: float = 0.1
    enable_reranking: bool = True
    enable_caching: bool = True


@dataclass
class QueryResult:
    """Result from a RAG query."""
    query: str
    answer: str
    sources: List[Dict]
    confidence: float
    latency_ms: float
    tokens_used: int = 0


class RAGPipeline:
    """Orchestrates the full RAG pipeline."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._query_cache: Dict[str, QueryResult] = {}
        self._total_queries = 0
        self._cache_hits = 0

    def query(self, question: str, filter_metadata: Optional[Dict] = None) -> QueryResult:
        start = time.time()
        self._total_queries += 1

        if self.config.enable_caching and question in self._query_cache:
            self._cache_hits += 1
            cached = self._query_cache[question]
            cached.latency_ms = (time.time() - start) * 1000
            return cached

        context_docs = self._retrieve(question, filter_metadata)
        if self.config.enable_reranking:
            context_docs = self._rerank(question, context_docs)
        context = self._build_context(context_docs)
        answer = self._generate(question, context)
        confidence = self._calculate_confidence(context_docs)

        result = QueryResult(
            query=question, answer=answer,
            sources=[{"content": d["content"][:200], "score": d["score"]} for d in context_docs],
            confidence=confidence, latency_ms=(time.time() - start) * 1000
        )

        if self.config.enable_caching:
            self._query_cache[question] = result
        return result

    def _retrieve(self, query: str, filter_metadata: Optional[Dict] = None) -> List[Dict]:
        logger.info(f"Retrieving documents for: {query[:50]}...")
        return [{"content": f"Retrieved content for: {query}", "score": 0.85, "source": "vectordb"}]

    def _rerank(self, query: str, documents: List[Dict]) -> List[Dict]:
        return sorted(documents, key=lambda d: d.get("score", 0), reverse=True)

    def _build_context(self, documents: List[Dict]) -> str:
        parts = []
        total_len = 0
        for doc in documents:
            content = doc.get("content", "")
            if total_len + len(content) > self.config.max_context_tokens * 4:
                break
            parts.append(content)
            total_len += len(content)
        return "\n\n---\n\n".join(parts)

    def _generate(self, question: str, context: str) -> str:
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        return f"Generated answer for: {question}"

    def _calculate_confidence(self, documents: List[Dict]) -> float:
        if not documents:
            return 0.0
        scores = [d.get("score", 0) for d in documents]
        return sum(scores) / len(scores)

    @property
    def stats(self) -> Dict:
        hit_rate = self._cache_hits / self._total_queries if self._total_queries > 0 else 0
        return {
            "total_queries": self._total_queries,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": round(hit_rate, 2),
            "cache_size": len(self._query_cache),
        }
