#!/usr/bin/env python3
"""Enhanced LLM paper scoring with multi-dimensional evaluation."""

import json
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PaperScore:
    """Multi-dimensional paper score."""
    novelty: float = 0.0      # 创新性 (0-10)
    impact: float = 0.0       # 影响力 (0-10)
    rigor: float = 0.0        # 严谨性 (0-10)
    clarity: float = 0.0      # 清晰度 (0-10)
    reproducibility: float = 0.0  # 可复现性 (0-10)
    overall: float = 0.0      # 综合评分 (0-10)
    
    def to_dict(self) -> Dict:
        return {
            "novelty": self.novelty,
            "impact": self.impact,
            "rigor": self.rigor,
            "clarity": self.clarity,
            "reproducibility": self.reproducibility,
            "overall": self.overall,
        }


class EnhancedLLMScorer:
    """Enhanced LLM paper scorer with multi-dimensional evaluation."""
    
    def __init__(self):
        # Keywords for each dimension
        self.novelty_keywords = [
            "novel", "new", "first", "propose", "introduce", "创新", "首次",
            "state-of-the-art", "outperform", "surpass", "突破",
        ]
        
        self.impact_keywords = [
            "significant", "substantial", "important", "crucial", "重要",
            "wide range", "many applications", "practical", "实用",
            "benchmark", "leaderboard", "top performance",
        ]
        
        self.rigor_keywords = [
            "theoretical", "proof", "analysis", "formal", "理论", "证明",
            "ablation", "statistical", "significance", "消融",
            "control", "baseline", "comparison", "对比",
        ]
        
        self.clarity_keywords = [
            "clear", "well-organized", "comprehensive", "detailed", "清晰",
            "figure", "table", "example", "illustration", "图表",
            "appendix", "supplementary", "附录",
        ]
        
        self.reproducibility_keywords = [
            "code", "open-source", "github", "implementation", "代码", "开源",
            "dataset", "pretrained", "checkpoint", "数据集",
            "reproduce", "replicate", "复现",
        ]
    
    def score_novelty(self, title: str, abstract: str) -> float:
        """Score novelty based on keywords."""
        text = f"{title} {abstract}".lower()
        score = 0.0
        
        for keyword in self.novelty_keywords:
            if keyword.lower() in text:
                score += 1.0
        
        return min(10.0, score * 2.0)
    
    def score_impact(self, title: str, abstract: str, citation_count: int = 0) -> float:
        """Score impact based on keywords and citations."""
        text = f"{title} {abstract}".lower()
        score = 0.0
        
        for keyword in self.impact_keywords:
            if keyword.lower() in text:
                score += 1.0
        
        # Citation bonus
        if citation_count > 100:
            score += 3.0
        elif citation_count > 50:
            score += 2.0
        elif citation_count > 10:
            score += 1.0
        
        return min(10.0, score * 2.0)
    
    def score_rigor(self, title: str, abstract: str) -> float:
        """Score rigor based on keywords."""
        text = f"{title} {abstract}".lower()
        score = 0.0
        
        for keyword in self.rigor_keywords:
            if keyword.lower() in text:
                score += 1.0
        
        return min(10.0, score * 2.0)
    
    def score_clarity(self, title: str, abstract: str) -> float:
        """Score clarity based on abstract structure."""
        score = 5.0  # Base score
        
        # Check for structured abstract
        if any(marker in abstract.lower() for marker in ["method", "result", "conclusion"]):
            score += 2.0
        
        # Check for figures/tables mention
        if any(marker in abstract.lower() for marker in ["figure", "table", "fig."]):
            score += 1.0
        
        # Check abstract length (good abstracts are 150-300 words)
        word_count = len(abstract.split())
        if 150 <= word_count <= 300:
            score += 2.0
        elif 100 <= word_count <= 400:
            score += 1.0
        
        return min(10.0, score)
    
    def score_reproducibility(self, title: str, abstract: str) -> float:
        """Score reproducibility based on keywords."""
        text = f"{title} {abstract}".lower()
        score = 0.0
        
        for keyword in self.reproducibility_keywords:
            if keyword.lower() in text:
                score += 1.5
        
        return min(10.0, score * 2.0)
    
    def calculate_overall(self, scores: PaperScore) -> float:
        """Calculate overall score with weights."""
        weights = {
            "novelty": 0.25,
            "impact": 0.25,
            "rigor": 0.20,
            "clarity": 0.15,
            "reproducibility": 0.15,
        }
        
        overall = (
            scores.novelty * weights["novelty"] +
            scores.impact * weights["impact"] +
            scores.rigor * weights["rigor"] +
            scores.clarity * weights["clarity"] +
            scores.reproducibility * weights["reproducibility"]
        )
        
        return round(overall, 1)
    
    def score_paper(
        self,
        title: str,
        abstract: str,
        citation_count: int = 0,
    ) -> PaperScore:
        """
        Score a paper across multiple dimensions.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            citation_count: Citation count
            
        Returns:
            PaperScore object
        """
        scores = PaperScore(
            novelty=self.score_novelty(title, abstract),
            impact=self.score_impact(title, abstract, citation_count),
            rigor=self.score_rigor(title, abstract),
            clarity=self.score_clarity(title, abstract),
            reproducibility=self.score_reproducibility(title, abstract),
        )
        
        scores.overall = self.calculate_overall(scores)
        
        return scores
    
    def get_strengths(self, scores: PaperScore) -> List[str]:
        """Get paper strengths based on scores."""
        strengths = []
        
        if scores.novelty >= 7:
            strengths.append("novelty")
        if scores.impact >= 7:
            strengths.append("impact")
        if scores.rigor >= 7:
            strengths.append("rigor")
        if scores.clarity >= 7:
            strengths.append("clarity")
        if scores.reproducibility >= 7:
            strengths.append("reproducibility")
        
        return strengths if strengths else ["general"]
    
    def get_improvements(self, scores: PaperScore) -> List[str]:
        """Get improvement suggestions based on scores."""
        improvements = []
        
        if scores.novelty < 5:
            improvements.append("可以提出更创新的方法")
        if scores.impact < 5:
            improvements.append("可以扩大应用场景")
        if scores.rigor < 5:
            improvements.append("可以增加理论分析和消融实验")
        if scores.clarity < 5:
            improvements.append("可以改进论文结构和图表")
        if scores.reproducibility < 5:
            improvements.append("可以开源代码和数据集")
        
        return improvements if improvements else ["持续改进"]


# Global instance
enhanced_scorer = EnhancedLLMScorer()


if __name__ == "__main__":
    # Test enhanced scoring
    print("Testing enhanced LLM scoring...")
    
    test_paper = {
        "title": "PoTRE: Test-Time Reasoning inspired by Cognitive Heterogeneity",
        "abstract": "While Large Language Models (LLMs) excel at many tasks, they frequently struggle with complex reasoning that requires long-horizon planning and iterative error correction. We introduce PoTRE, a novel heterogeneous framework that decouples inference into four agents. We evaluate PoTRE on three frontier benchmarks and achieve state-of-the-art accuracy.",
        "citation_count": 0,
    }
    
    scores = enhanced_scorer.score_paper(
        test_paper["title"],
        test_paper["abstract"],
        test_paper["citation_count"],
    )
    
    print(f"\nPaper: {test_paper['title']}")
    print(f"Scores: {scores.to_dict()}")
    print(f"Strengths: {enhanced_scorer.get_strengths(scores)}")
    print(f"Improvements: {enhanced_scorer.get_improvements(scores)}")
