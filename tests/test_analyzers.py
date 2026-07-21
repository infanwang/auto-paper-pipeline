"""单元测试 - 论文分析器"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analyzers.paper_analyzer import PaperAnalyzer, ReproducibilityAssessor


class TestPaperAnalyzer:
    """论文分析器测试"""
    
    def test_init(self):
        """测试初始化"""
        analyzer = PaperAnalyzer()
        assert analyzer is not None
    
    def test_analyze(self, sample_paper):
        """测试分析"""
        analyzer = PaperAnalyzer()
        analysis = analyzer.analyze(sample_paper)
        
        assert 'paper_id' in analysis
        assert 'title' in analysis
        assert 'problem' in analysis
        assert 'method' in analysis
        assert 'strengths' in analysis
        assert 'weaknesses' in analysis
        assert 'improvements' in analysis
    
    def test_extract_problem(self):
        """测试问题提取"""
        analyzer = PaperAnalyzer()
        abstract = "This paper addresses the challenge of long-context reasoning in LLMs. The problem is that existing methods fail to scale."
        problem = analyzer._extract_problem(abstract)
        assert 'challenge' in problem.lower() or 'problem' in problem.lower()
    
    def test_extract_method(self):
        """测试方法提取"""
        analyzer = PaperAnalyzer()
        abstract = "We propose a novel framework for efficient attention mechanism that reduces complexity."
        method = analyzer._extract_method(abstract)
        assert 'propose' in method.lower() or 'novel' in method.lower()
    
    def test_extract_strengths(self):
        """测试优点提取"""
        analyzer = PaperAnalyzer()
        abstract = "This novel approach achieves state-of-the-art performance and is highly efficient."
        strengths = analyzer._extract_strengths(abstract)
        assert 'novelty' in strengths
        assert 'performance' in strengths
    
    def test_extract_weaknesses(self):
        """测试缺点提取"""
        analyzer = PaperAnalyzer()
        abstract = "The method is computationally expensive and requires large datasets for training."
        weaknesses = analyzer._infer_weaknesses(abstract)
        assert 'computational_cost' in weaknesses or 'data_dependency' in weaknesses
    
    def test_suggest_improvements(self, sample_paper):
        """测试改进建议"""
        analyzer = PaperAnalyzer()
        improvements = analyzer._suggest_improvements(sample_paper)
        assert len(improvements) > 0
        assert len(improvements) <= 3


class TestReproducibilityAssessor:
    """可复现性评估器测试"""
    
    def test_assess(self, sample_paper):
        """测试评估"""
        analyzer = PaperAnalyzer()
        assessor = ReproducibilityAssessor()
        
        analysis = analyzer.analyze(sample_paper)
        result = assessor.assess(sample_paper, analysis)
        
        assert 'score' in result
        assert 'factors' in result
        assert 'verdict' in result
        assert 0 <= result['score'] <= 10
        assert result['verdict'] in ['高', '中', '低']
    
    def test_assess_with_github(self):
        """测试有GitHub链接的评估"""
        assessor = ReproducibilityAssessor()
        paper = {'github_repo': 'https://github.com/test/repo', 'abstract': 'A detailed method with dataset and baseline.'}
        analysis = {'experiments': {'datasets': ['ImageNet'], 'baselines': ['ResNet']}, 'method': 'A very detailed method description that is more than 100 characters long for testing purposes.'}
        
        result = assessor.assess(paper, analysis)
        assert result['score'] >= 5  # 有GitHub链接和详细方法
