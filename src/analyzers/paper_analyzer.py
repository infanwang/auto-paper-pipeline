"""论文深度分析器 - V2.0"""

import json
import re
from typing import Dict, List, Optional
from pathlib import Path


class PaperAnalyzer:
    """论文深度分析器"""
    
    def __init__(self):
        self.method_patterns = {
            'propose': 'proposed method',
            'introduce': 'introduced approach',
            'develop': 'developed framework',
            'present': 'presented system',
            'design': 'designed architecture',
        }
        
        self.result_patterns = {
            'achieve': 'achieved result',
            'outperform': 'outperformed baseline',
            'improve': 'improved performance',
            'surpass': 'surpassed previous',
            'state-of-the-art': 'state-of-the-art',
        }
    
    def analyze(self, paper: Dict) -> Dict:
        """分析单篇论文"""
        abstract = paper.get('abstract', '')
        
        analysis = {
            'paper_id': paper['id'],
            'title': paper['title'],
            'problem': self._extract_problem(abstract),
            'method': self._extract_method(abstract),
            'architecture': self._extract_architecture(abstract),
            'experiments': self._extract_experiments(abstract),
            'results': self._extract_results(abstract),
            'strengths': self._extract_strengths(abstract),
            'weaknesses': self._infer_weaknesses(abstract),
            'improvements': self._suggest_improvements(paper),
            'code_quality': self._estimate_code_quality(paper),
            'reproducibility': self._estimate_reproducibility(paper),
        }
        
        return analysis
    
    def _extract_problem(self, abstract: str) -> str:
        """提取问题"""
        sentences = abstract.split('. ')
        for s in sentences:
            if any(kw in s.lower() for kw in ['challenge', 'problem', 'limitation', 'bottleneck', 'issue']):
                return s.strip()[:200]
        return sentences[0].strip()[:200] if sentences else "N/A"
    
    def _extract_method(self, abstract: str) -> str:
        """提取方法"""
        sentences = abstract.split('. ')
        for s in sentences:
            if any(kw in s.lower() for kw in ['propose', 'introduce', 'present', 'develop', 'method', 'framework']):
                return s.strip()[:200]
        return "N/A"
    
    def _extract_architecture(self, abstract: str) -> List[str]:
        """提取架构组件"""
        components = []
        arch_keywords = ['network', 'model', 'encoder', 'decoder', 'attention', 
                        'transformer', 'layer', 'module', 'block', 'component']
        
        for kw in arch_keywords:
            if kw in abstract.lower():
                components.append(kw)
        
        return components[:5] if components else ['N/A']
    
    def _extract_experiments(self, abstract: str) -> Dict:
        """提取实验信息"""
        experiments = {
            'datasets': [],
            'baselines': [],
            'metrics': [],
        }
        
        # 数据集
        dataset_keywords = ['dataset', 'benchmark', 'corpus', 'evaluation']
        for kw in dataset_keywords:
            if kw in abstract.lower():
                experiments['datasets'].append(kw)
        
        # 指标
        metric_keywords = ['accuracy', 'f1', 'bleu', 'rouge', 'perplexity', 'latency', 'throughput']
        for kw in metric_keywords:
            if kw in abstract.lower():
                experiments['metrics'].append(kw)
        
        return experiments
    
    def _extract_results(self, abstract: str) -> Dict:
        """提取结果"""
        results = {
            'performance': [],
            'comparison': [],
        }
        
        # 性能数字
        numbers = re.findall(r'(\d+\.?\d*)\s*%', abstract)
        if numbers:
            results['performance'] = [f"{n}%" for n in numbers[:3]]
        
        # 对比
        comparison_keywords = ['compared to', 'vs', 'versus', 'against']
        for kw in comparison_keywords:
            if kw in abstract.lower():
                results['comparison'].append(kw)
        
        return results
    
    def _extract_strengths(self, abstract: str) -> List[str]:
        """提取优点"""
        strengths = []
        
        strength_keywords = {
            'novelty': ['novel', 'new', 'first', 'innovative'],
            'performance': ['state-of-the-art', 'sota', 'best', 'superior'],
            'efficiency': ['efficient', 'fast', 'scalable', 'lightweight'],
            'generalization': ['generalize', 'robust', 'adaptive'],
        }
        
        for category, keywords in strength_keywords.items():
            if any(kw in abstract.lower() for kw in keywords):
                strengths.append(category)
        
        return strengths[:3] if strengths else ['N/A']
    
    def _infer_weaknesses(self, abstract: str) -> List[str]:
        """推断缺点"""
        weaknesses = []
        
        weakness_keywords = {
            'complexity': ['complex', 'complicated', 'sophisticated'],
            'computational_cost': ['expensive', 'costly', 'computationally'],
            'data_dependency': ['large dataset', 'data hungry', 'requires'],
            'limited_scope': ['specific', 'limited', 'particular'],
        }
        
        for category, keywords in weakness_keywords.items():
            if any(kw in abstract.lower() for kw in keywords):
                weaknesses.append(category)
        
        return weaknesses[:2] if weaknesses else ['需要进一步验证']
    
    def _suggest_improvements(self, paper: Dict) -> List[str]:
        """建议改进"""
        improvements = []
        
        abstract = paper.get('abstract', '').lower()
        
        if 'small dataset' in abstract or 'limited data' in abstract:
            improvements.append('可以尝试数据增强或迁移学习')
        
        if 'computational cost' in abstract or 'expensive' in abstract:
            improvements.append('可以优化计算效率，如模型压缩或蒸馏')
        
        if 'specific domain' in abstract:
            improvements.append('可以扩展到更多领域验证泛化性')
        
        if len(improvements) < 3:
            improvements.extend([
                '可以与其他方法进行对比实验',
                '可以提供更详细的消融实验',
                '可以开源代码和数据集',
            ])
        
        return improvements[:3]
    
    def _estimate_code_quality(self, paper: Dict) -> float:
        """估算代码质量"""
        # 基于论文信息估算
        score = 5.0
        
        # 有GitHub链接加1分
        if paper.get('github_repo'):
            score += 1.0
        
        # 作者数量多通常代码质量更好
        if len(paper.get('authors', [])) > 3:
            score += 0.5
        
        # 有实验结果加1分
        if paper.get('results'):
            score += 1.0
        
        return min(score, 10.0)
    
    def _estimate_reproducibility(self, paper: Dict) -> float:
        """估算可复现性"""
        score = 5.0
        
        abstract = paper.get('abstract', '').lower()
        
        # 有代码链接
        if paper.get('github_repo'):
            score += 2.0
        
        # 有详细实验描述
        if 'dataset' in abstract and 'baseline' in abstract:
            score += 1.0
        
        # 有消融实验
        if 'ablation' in abstract:
            score += 0.5
        
        return min(score, 10.0)


class CodeAnalyzer:
    """代码质量分析器"""
    
    def analyze_github_repo(self, repo_url: str) -> Dict:
        """分析GitHub仓库"""
        # 简化实现：返回基本分析
        return {
            'url': repo_url,
            'has_readme': True,
            'has_requirements': True,
            'has_tests': False,
            'code_quality': 7.0,
            'documentation_quality': 6.0,
            'reproducibility': 7.0,
        }


class ReproducibilityAssessor:
    """可复现性评估器"""
    
    def assess(self, paper: Dict, analysis: Dict) -> Dict:
        """评估可复现性"""
        factors = {
            'code_available': bool(paper.get('github_repo')),
            'dataset_mentioned': bool(analysis.get('experiments', {}).get('datasets')),
            'baselines_mentioned': bool(analysis.get('experiments', {}).get('baselines')),
            'metrics_specified': bool(analysis.get('experiments', {}).get('metrics')),
            'detailed_method': len(analysis.get('method', '')) > 100,
        }
        
        score = sum(factors.values()) / len(factors) * 10
        
        return {
            'score': round(score, 1),
            'factors': factors,
            'verdict': '高' if score >= 7 else ('中' if score >= 4 else '低'),
        }
