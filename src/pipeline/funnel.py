"""多级过滤漏斗 - V2.0"""

import json
import re
from typing import List, Dict, Tuple
from pathlib import Path


class TFFilter:
    """Stage 1: TF-IDF 快速过滤"""
    
    def __init__(self, domain_keywords: List[str]):
        self.domain_keywords = [kw.lower() for kw in domain_keywords]
    
    def _compute_score(self, text: str) -> float:
        """计算领域相关性分数"""
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        # 关键词匹配
        keyword_matches = sum(1 for kw in self.domain_keywords if kw in text_lower)
        keyword_score = min(keyword_matches / 5, 1.0)
        
        # 重要词汇匹配
        important_terms = ['novel', 'propose', 'state-of-the-art', 'outperform', 
                          'significant', 'improve', 'achieve', 'demonstrate']
        term_matches = sum(1 for t in important_terms if t in text_lower)
        term_score = min(term_matches / 3, 1.0)
        
        return keyword_score * 0.7 + term_score * 0.3
    
    def filter(self, papers: List[Dict], threshold: float = 0.2) -> List[Dict]:
        """过滤论文"""
        filtered = []
        
        for paper in papers:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            score = self._compute_score(text)
            
            if score >= threshold:
                paper['tfidf_score'] = score
                filtered.append(paper)
        
        # 按分数排序
        filtered.sort(key=lambda x: x.get('tfidf_score', 0), reverse=True)
        
        return filtered


class LLMScorer:
    """Stage 2: LLM 深度评分"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
    
    def score_batch(self, papers: List[Dict], batch_size: int = 5) -> List[Dict]:
        """批量评分（合并多篇到一次请求）"""
        scored_papers = []
        
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i+batch_size]
            
            # 构建批量评分prompt
            papers_text = ""
            for j, p in enumerate(batch, 1):
                papers_text += f"\n{j}. 标题: {p['title']}\n   摘要: {p['abstract'][:200]}...\n"
            
            prompt = f"""你是AI领域资深审稿人。评估以下{len(batch)}篇论文的相关性和重要性。

论文列表：
{papers_text}

请为每篇论文评分（1-10分）并给出一句话总结。返回JSON格式：
{{
  "scores": [
    {{"index": 1, "score": 8, "summary": "...", "tags": ["tag1", "tag2"]}},
    ...
  ]
}}"""
            
            # 模拟LLM调用（实际使用时替换为真实API）
            for j, p in enumerate(batch):
                # 简单启发式评分
                score = self._heuristic_score(p)
                p['llm_score'] = score
                p['llm_summary'] = self._extract_summary(p['abstract'])
                p['llm_tags'] = self._extract_tags(p['abstract'])
                scored_papers.append(p)
        
        return scored_papers
    
    def _heuristic_score(self, paper: Dict) -> float:
        """启发式评分（替代LLM调用）"""
        abstract = paper.get('abstract', '').lower()
        score = 5.0
        
        # 新颖性指标
        if 'novel' in abstract or 'propose' in abstract or 'introduce' in abstract:
            score += 1.0
        
        # 性能指标
        if 'state-of-the-art' in abstract or 'sota' in abstract:
            score += 1.5
        if 'outperform' in abstract or 'surpass' in abstract:
            score += 1.0
        
        # 引用数加成
        citations = paper.get('citation_count', 0)
        if citations > 100:
            score += 1.0
        elif citations > 50:
            score += 0.5
        
        # 作者数量（多人合作通常更可靠）
        if len(paper.get('authors', [])) > 3:
            score += 0.3
        
        return min(max(score, 1.0), 10.0)
    
    def _extract_summary(self, abstract: str) -> str:
        """提取摘要"""
        # 取前100个字符作为简短摘要
        return abstract[:100].strip() + "..."
    
    def _extract_tags(self, abstract: str) -> List[str]:
        """提取标签"""
        tags = []
        tag_keywords = {
            'LLM': ['llm', 'large language model', 'language model'],
            'Agent': ['agent', 'agentic', 'tool use'],
            '推理优化': ['inference', 'optimization', 'efficient', 'fast'],
            '量化': ['quantization', 'quantize', 'int4', 'int8'],
            '多模态': ['multimodal', 'vision', 'image', 'video'],
            '代码生成': ['code generation', 'programming', 'software'],
            '芯片验证': ['verification', 'formal', 'circuit'],
            '5G': ['5g', 'wireless', 'communication', 'mimo'],
        }
        
        abstract_lower = abstract.lower()
        for tag, keywords in tag_keywords.items():
            if any(kw in abstract_lower for kw in keywords):
                tags.append(tag)
        
        return tags[:3]  # 最多3个标签


class FunnelPipeline:
    """多级过滤漏斗管线"""
    
    def __init__(self, domain_keywords: List[str]):
        self.stage1 = TFFilter(domain_keywords)
        self.stage2 = LLMScorer()
    
    def run(self, papers: List[Dict], 
            tfidf_threshold: float = 0.2,
            llm_top_n: int = 20) -> List[Dict]:
        """运行漏斗"""
        print(f"  Stage 1: TF-IDF过滤 ({len(papers)}篇)")
        filtered = self.stage1.filter(papers, threshold=tfidf_threshold)
        print(f"  → 通过: {len(filtered)}篇")
        
        print(f"  Stage 2: LLM评分 (Top {llm_top_n}篇)")
        top_papers = filtered[:llm_top_n]
        scored = self.stage2.score_batch(top_papers)
        
        # 按LLM分数排序
        scored.sort(key=lambda x: x.get('llm_score', 0), reverse=True)
        
        print(f"  → 完成: {len(scored)}篇")
        return scored
