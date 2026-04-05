"""Text chunking strategies for document processing."""
from typing import List, Dict, Optional
import re
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text chunk with metadata."""
    text: str
    chunk_index: int
    source_doc_id: str
    start_char: int
    end_char: int
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TextChunker:
    """Configurable text chunking with multiple strategies."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50,
                 strategy: str = "sliding_window"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy
        self._strategies = {
            "sliding_window": self._sliding_window,
            "sentence": self._sentence_based,
            "paragraph": self._paragraph_based,
            "semantic": self._semantic_based,
        }

    def chunk(self, text: str, doc_id: str = "unknown") -> List[Chunk]:
        if self.strategy not in self._strategies:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        chunks = self._strategies[self.strategy](text, doc_id)
        logger.info(f"Chunked document {doc_id} into {len(chunks)} chunks using {self.strategy}")
        return chunks

    def _sliding_window(self, text: str, doc_id: str) -> List[Chunk]:
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(Chunk(
                text=text[start:end], chunk_index=idx,
                source_doc_id=doc_id, start_char=start, end_char=end
            ))
            start += self.chunk_size - self.overlap
            idx += 1
        return chunks

    def _sentence_based(self, text: str, doc_id: str) -> List[Chunk]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        start_char = 0
        idx = 0
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(Chunk(
                    text=current_chunk.strip(), chunk_index=idx,
                    source_doc_id=doc_id, start_char=start_char,
                    end_char=start_char + len(current_chunk)
                ))
                idx += 1
                start_char += len(current_chunk)
                current_chunk = ""
            current_chunk += sentence + " "
        if current_chunk.strip():
            chunks.append(Chunk(
                text=current_chunk.strip(), chunk_index=idx,
                source_doc_id=doc_id, start_char=start_char,
                end_char=start_char + len(current_chunk)
            ))
        return chunks

    def _paragraph_based(self, text: str, doc_id: str) -> List[Chunk]:
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        pos = 0
        for idx, para in enumerate(paragraphs):
            para = para.strip()
            if para:
                chunks.append(Chunk(
                    text=para, chunk_index=idx,
                    source_doc_id=doc_id, start_char=pos, end_char=pos + len(para)
                ))
            pos += len(para) + 2
        return chunks

    def _semantic_based(self, text: str, doc_id: str) -> List[Chunk]:
        return self._sentence_based(text, doc_id)


class RecursiveChunker(TextChunker):
    """Recursively splits text using multiple separators."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50,
                 separators: Optional[List[str]] = None):
        super().__init__(chunk_size, overlap, "recursive")
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, text: str, doc_id: str = "unknown") -> List[Chunk]:
        chunks = self._recursive_split(text, self.separators)
        result = []
        pos = 0
        for idx, chunk_text in enumerate(chunks):
            result.append(Chunk(
                text=chunk_text, chunk_index=idx,
                source_doc_id=doc_id, start_char=pos, end_char=pos + len(chunk_text)
            ))
            pos += len(chunk_text)
        return result

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return [text[:self.chunk_size]] if len(text) > self.chunk_size else [text]
        sep = separators[0]
        remaining_seps = separators[1:]
        if sep:
            parts = text.split(sep)
        else:
            parts = list(text)
        chunks = []
        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > self.chunk_size:
                    chunks.extend(self._recursive_split(part, remaining_seps))
                else:
                    current = part
        if current:
            chunks.append(current)
        return chunks
