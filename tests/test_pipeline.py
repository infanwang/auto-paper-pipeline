#!/usr/bin/env python3
"""Test suite for paper pipeline modules."""

import pytest
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


class TestAntiCrawl:
    """Test anti-crawl module."""
    
    def test_import(self):
        from anti_crawl import AntiCrawl
        assert AntiCrawl is not None
    
    def test_get_headers(self):
        from anti_crawl import AntiCrawl
        ac = AntiCrawl()
        headers = ac.get_headers()
        assert "User-Agent" in headers
        assert "Accept" in headers
    
    def test_wait(self):
        import time
        from anti_crawl import AntiCrawl
        ac = AntiCrawl(min_delay=0.1, max_delay=0.2)
        start = time.time()
        ac.wait()
        elapsed = time.time() - start
        assert elapsed >= 0.1


class TestDedup:
    """Test dedup module."""
    
    def test_import(self):
        from dedup import Deduplicator
        assert Deduplicator is not None
    
    def test_register_and_check(self, tmp_path):
        from dedup import Deduplicator
        dedup = Deduplicator(index_path=tmp_path / "test.json")
        
        # Register a paper
        dedup.register("2607.00001", "Test Paper")
        
        # Check duplicate
        assert dedup.is_duplicate("2607.00001")
        assert not dedup.is_duplicate("2607.00002")
    
    def test_title_similarity(self, tmp_path):
        from dedup import Deduplicator
        dedup = Deduplicator(index_path=tmp_path / "test.json")
        
        dedup.register("2607.00001", "A Novel Approach to Machine Learning")
        
        # Similar title should be detected as duplicate
        assert dedup.is_duplicate("2607.00002", "A Novel Approach to Machine Learning Systems")


class TestMultilingual:
    """Test multilingual module."""
    
    def test_import(self):
        from multilingual import LanguageDetector, Language
        assert LanguageDetector is not None
    
    def test_english_detection(self):
        from multilingual import LanguageDetector, Language
        detector = LanguageDetector()
        lang, confidence = detector.detect("We propose a novel method for machine learning.")
        assert lang == Language.ENGLISH
    
    def test_chinese_detection(self):
        from multilingual import LanguageDetector, Language
        detector = LanguageDetector()
        lang, confidence = detector.detect("我们提出了一种新的机器学习方法。")
        assert lang == Language.CHINESE
    
    def test_spanish_detection(self):
        from multilingual import LanguageDetector, Language
        detector = LanguageDetector()
        lang, confidence = detector.detect("Proponemos un nuevo método para el aprendizaje automático.")
        assert lang == Language.SPANISH


class TestEnhancedScorer:
    """Test enhanced scorer module."""
    
    def test_import(self):
        from enhanced_scorer import EnhancedLLMScorer
        assert EnhancedLLMScorer is not None
    
    def test_score_paper(self):
        from enhanced_scorer import EnhancedLLMScorer
        scorer = EnhancedLLMScorer()
        scores = scorer.score_paper(
            "Test Paper Title",
            "This is a test abstract with some keywords like novel and method.",
            10
        )
        assert scores.overall > 0
        assert scores.novelty > 0


class TestKnowledgeGraph:
    """Test knowledge graph module."""
    
    def test_import(self):
        from knowledge_graph import PaperKnowledgeGraph
        assert PaperKnowledgeGraph is not None
    
    def test_add_paper(self, tmp_path):
        from knowledge_graph import PaperKnowledgeGraph
        kg = PaperKnowledgeGraph(data_dir=str(tmp_path))
        
        paper = {
            "id": "2607.00001",
            "title": "Test Paper",
            "authors": ["Author One"],
            "topics": ["AI"],
            "year": "2026",
        }
        kg.add_paper(paper)
        
        assert "2607.00001" in kg.graph["papers"]


class TestRecommender:
    """Test recommender module."""
    
    def test_import(self):
        from recommender import PaperRecommender
        assert PaperRecommender is not None


class TestTranslator:
    """Test translator module."""
    
    def test_import(self):
        from translator import AbstractTranslator
        assert AbstractTranslator is not None


class TestMultiSourceCrawler:
    """Test multi-source crawler module."""
    
    def test_import(self):
        from multi_source_crawler import MultiSourceCrawler
        assert MultiSourceCrawler is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
