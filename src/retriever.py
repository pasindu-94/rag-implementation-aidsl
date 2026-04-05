"""Document retrieval engine for RAG pipeline."""
from typing import List, Dict, Optional, Tuple
import numpy as np
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document with content and metadata."""
    doc_id: str
    content: str
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    score: float = 0.0


@dataclass
class RetrievalResult:
    """Result from a retrieval query."""
    query: str
    documents: List[Document]
    total_candidates: int
    retrieval_time_ms: float = 0.0


class VectorStore:
    """In-memory vector store for document retrieval."""

    def __init__(self):
        self._documents: Dict[str, Document] = {}
        self._embeddings: Dict[str, List[float]] = {}

    def add_document(self, doc: Document) -> None:
        self._documents[doc.doc_id] = doc
        if doc.embedding:
            self._embeddings[doc.doc_id] = doc.embedding
            logger.debug(f"Added document {doc.doc_id} with embedding")

    def add_documents(self, docs: List[Document]) -> int:
        added = 0
        for doc in docs:
            self.add_document(doc)
            added += 1
        logger.info(f"Added {added} documents to vector store")
        return added

    def search(self, query_embedding: List[float], top_k: int = 5,
               filter_metadata: Optional[Dict] = None) -> List[Document]:
        candidates = []
        for doc_id, embedding in self._embeddings.items():
            doc = self._documents[doc_id]
            if filter_metadata:
                if not all(doc.metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue
            score = self._cosine_similarity(query_embedding, embedding)
            doc_copy = Document(
                doc_id=doc.doc_id, content=doc.content,
                metadata=doc.metadata, embedding=doc.embedding, score=score
            )
            candidates.append(doc_copy)
        candidates.sort(key=lambda d: d.score, reverse=True)
        return candidates[:top_k]

    def delete_document(self, doc_id: str) -> bool:
        if doc_id in self._documents:
            del self._documents[doc_id]
            self._embeddings.pop(doc_id, None)
            return True
        return False

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = np.sqrt(sum(x**2 for x in a))
        norm_b = np.sqrt(sum(x**2 for x in b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    @property
    def size(self) -> int:
        return len(self._documents)

    def get_document(self, doc_id: str) -> Optional[Document]:
        return self._documents.get(doc_id)

    def list_documents(self, limit: int = 100) -> List[Document]:
        return list(self._documents.values())[:limit]
