"""单元测试 - 过滤漏斗"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.funnel import TFFilter, LLMScorer, FunnelPipeline


class TestTFFilter:
    """TF-IDF过滤器测试"""
    
    def test_init(self, domain_keywords):
        """测试初始化"""
        filter = TFFilter(domain_keywords)
        assert len(filter.domain_keywords) > 0
    
    def test_compute_score_high(self, domain_keywords):
        """测试高分计算"""
        filter = TFFilter(domain_keywords)
        text = "This paper proposes a novel LLM agent framework with attention mechanism"
        score = filter._compute_score(text)
        assert score > 0.3
    
    def test_compute_score_low(self, domain_keywords):
        """测试低分计算"""
        filter = TFFilter(domain_keywords)
        text = "This paper is about cooking recipes and food preparation"
        score = filter._compute_score(text)
        assert score < 0.3
    
    def test_filter(self, domain_keywords, sample_papers):
        """测试过滤"""
        filter = TFFilter(domain_keywords)
        filtered = filter.filter(sample_papers, threshold=0.1)
        assert len(filtered) <= len(sample_papers)
        assert all('tfidf_score' in p for p in filtered)


class TestLLMScorer:
    """LLM评分器测试"""
    
    def test_init(self):
        """测试初始化"""
        scorer = LLMScorer()
        assert scorer.model == "gpt-4o-mini"
    
    def test_heuristic_score(self, sample_paper):
        """测试启发式评分"""
        scorer = LLMScorer()
        score = scorer._heuristic_score(sample_paper)
        assert 1.0 <= score <= 10.0
    
    def test_extract_summary(self):
        """测试摘要提取"""
        scorer = LLMScorer()
        abstract = "This is a test abstract about LLM agents and their applications in various domains."
        summary = scorer._extract_summary(abstract)
        assert len(summary) <= 110
        assert summary.endswith("...")
    
    def test_extract_tags(self):
        """测试标签提取"""
        scorer = LLMScorer()
        abstract = "This paper proposes a novel LLM agent framework with multimodal capabilities."
        tags = scorer._extract_tags(abstract)
        assert "LLM" in tags
        assert "Agent" in tags
        assert "多模态" in tags
    
    def test_score_batch(self, sample_papers):
        """测试批量评分"""
        scorer = LLMScorer()
        scored = scorer.score_batch(sample_papers[:3])
        assert len(scored) == 3
        assert all('llm_score' in p for p in scored)
        assert all('llm_summary' in p for p in scored)
        assert all('llm_tags' in p for p in scored)


class TestFunnelPipeline:
    """过滤漏斗测试"""
    
    def test_init(self, domain_keywords):
        """测试初始化"""
        pipeline = FunnelPipeline(domain_keywords)
        assert pipeline.stage1 is not None
        assert pipeline.stage2 is not None
    
    def test_run(self, domain_keywords, sample_papers):
        """测试运行"""
        pipeline = FunnelPipeline(domain_keywords)
        result = pipeline.run(sample_papers, tfidf_threshold=0.1, llm_top_n=3)
        assert len(result) <= 3
        assert all('llm_score' in p for p in result)
