"""
论文影响力预测模块
基于多特征的论文影响力评分
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta


class ImpactPredictor:
    """论文影响力预测器"""
    
    def __init__(self):
        # 特征权重
        self.weights = {
            'recency': 0.25,      # 新鲜度
            'citation': 0.20,     # 引用数
            'author_count': 0.10, # 作者数
            'abstract_quality': 0.15,  # 摘要质量
            'keyword_relevance': 0.15, # 关键词相关性
            'category_importance': 0.15 # 领域重要性
        }
        
        # 领域重要性分数
        self.category_scores = {
            'cs.AI': 0.9,
            'cs.LG': 0.85,
            'cs.CL': 0.8,
            'cs.CV': 0.75,
            'cs.SE': 0.7,
            'cs.AR': 0.65,
            'eess.SP': 0.6,
            'cs.IT': 0.55,
        }
    
    def predict_impact(self, paper: Dict) -> Dict:
        """预测论文影响力"""
        # 提取特征
        features = self._extract_features(paper)
        
        # 计算加权分数
        score = sum(features[k] * self.weights[k] for k in self.weights)
        
        # 归一化到0-100
        score = min(max(score * 100, 0), 100)
        
        # 生成预测
        prediction = {
            'impact_score': round(score, 1),
            'impact_level': self._get_impact_level(score),
            'features': features,
            'confidence': self._compute_confidence(paper)
        }
        
        return prediction
    
    def _extract_features(self, paper: Dict) -> Dict:
        """提取特征"""
        features = {}
        
        # 1. 新鲜度
        pub_date = paper.get('published_date', '')
        if pub_date:
            try:
                days_old = (datetime.now() - datetime.strptime(pub_date[:10], '%Y-%m-%d')).days
                features['recency'] = max(0, 1 - days_old / 30)  # 30天内有效
            except:
                features['recency'] = 0.5
        else:
            features['recency'] = 0.5
        
        # 2. 引用数
        citations = paper.get('citation_count', 0)
        features['citation'] = min(citations / 100, 1.0)
        
        # 3. 作者数
        n_authors = len(paper.get('authors', []))
        features['author_count'] = min(n_authors / 10, 1.0)
        
        # 4. 摘要质量
        abstract = paper.get('abstract', '')
        quality_indicators = ['novel', 'state-of-the-art', 'outperform', 'significant', 'achieve']
        quality_score = sum(1 for ind in quality_indicators if ind in abstract.lower())
        features['abstract_quality'] = min(quality_score / 3, 1.0)
        
        # 5. 关键词相关性
        tags = paper.get('llm_tags', [])
        hot_keywords = ['agent', 'llm', 'inference', 'optimization', 'multimodal']
        relevance = sum(1 for tag in tags if tag.lower() in hot_keywords)
        features['keyword_relevance'] = min(relevance / 2, 1.0)
        
        # 6. 领域重要性
        categories = paper.get('categories', [])
        if categories:
            cat = categories[0]
            features['category_importance'] = self.category_scores.get(cat, 0.5)
        else:
            features['category_importance'] = 0.5
        
        return features
    
    def _get_impact_level(self, score: float) -> str:
        """获取影响力等级"""
        if score >= 80:
            return 'HIGH'
        elif score >= 60:
            return 'MEDIUM'
        elif score >= 40:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def _compute_confidence(self, paper: Dict) -> float:
        """计算置信度"""
        confidence = 0.5
        
        if paper.get('citation_count', 0) > 0:
            confidence += 0.2
        
        if paper.get('llm_score', 0) > 7:
            confidence += 0.15
        
        if paper.get('abstract', '') and len(paper.get('abstract', '')) > 100:
            confidence += 0.15
        
        return min(confidence, 1.0)
    
    def batch_predict(self, papers: List[Dict]) -> List[Dict]:
        """批量预测"""
        return [self.predict_impact(p) for p in papers]
    
    def rank_papers(self, papers: List[Dict]) -> List[Dict]:
        """按影响力排名"""
        ranked = []
        for p in papers:
            prediction = self.predict_impact(p)
            ranked.append({**p, 'impact': prediction})
        
        ranked.sort(key=lambda x: x['impact']['impact_score'], reverse=True)
        return ranked
