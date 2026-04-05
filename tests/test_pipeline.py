"""Tests for RAG pipeline module."""
import pytest
from src.pipeline import RAGPipeline, PipelineConfig, QueryResult


class TestRAGPipeline:
    def test_basic_query(self):
        pipeline = RAGPipeline()
        result = pipeline.query("What is RAG?")
        assert isinstance(result, QueryResult)
        assert result.query == "What is RAG?"
        assert result.answer
        assert result.latency_ms > 0

    def test_caching(self):
        config = PipelineConfig(enable_caching=True)
        pipeline = RAGPipeline(config)
        result1 = pipeline.query("test question")
        result2 = pipeline.query("test question")
        assert pipeline.stats["cache_hits"] == 1

    def test_no_caching(self):
        config = PipelineConfig(enable_caching=False)
        pipeline = RAGPipeline(config)
        pipeline.query("q1")
        pipeline.query("q1")
        assert pipeline.stats["cache_hits"] == 0

    def test_stats(self):
        pipeline = RAGPipeline()
        pipeline.query("q1")
        pipeline.query("q2")
        stats = pipeline.stats
        assert stats["total_queries"] == 2

    def test_confidence_score(self):
        pipeline = RAGPipeline()
        result = pipeline.query("test")
        assert 0.0 <= result.confidence <= 1.0

    def test_custom_config(self):
        config = PipelineConfig(top_k=10, temperature=0.5, model_name="gpt-3.5-turbo")
        pipeline = RAGPipeline(config)
        assert pipeline.config.top_k == 10
        assert pipeline.config.model_name == "gpt-3.5-turbo"
