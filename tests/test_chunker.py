"""Tests for text chunking module."""
import pytest
from src.chunker import TextChunker, RecursiveChunker, Chunk


class TestTextChunker:
    def test_sliding_window_basic(self):
        chunker = TextChunker(chunk_size=10, overlap=2)
        text = "Hello world this is a test"
        chunks = chunker.chunk(text, "test-doc")
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_sliding_window_overlap(self):
        chunker = TextChunker(chunk_size=20, overlap=5)
        text = "A" * 50
        chunks = chunker.chunk(text, "doc1")
        assert len(chunks) >= 3

    def test_sentence_based(self):
        chunker = TextChunker(chunk_size=50, strategy="sentence")
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk(text, "doc2")
        assert len(chunks) >= 1

    def test_paragraph_based(self):
        chunker = TextChunker(strategy="paragraph")
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunker.chunk(text, "doc3")
        assert len(chunks) == 3

    def test_empty_text(self):
        chunker = TextChunker()
        chunks = chunker.chunk("", "empty")
        assert len(chunks) <= 1

    def test_invalid_strategy(self):
        chunker = TextChunker(strategy="invalid")
        with pytest.raises(ValueError):
            chunker.chunk("test", "doc")


class TestRecursiveChunker:
    def test_recursive_split(self):
        chunker = RecursiveChunker(chunk_size=50)
        text = "Para one content here.\n\nPara two content here.\n\nPara three."
        chunks = chunker.chunk(text, "rdoc")
        assert len(chunks) >= 1
        assert all(len(c.text) <= 50 or len(c.text.split()) == 1 for c in chunks)

    def test_preserves_content(self):
        chunker = RecursiveChunker(chunk_size=100)
        text = "Some text that should be preserved entirely."
        chunks = chunker.chunk(text, "rdoc2")
        combined = "".join(c.text for c in chunks)
        assert text in combined or len(chunks) == 1
