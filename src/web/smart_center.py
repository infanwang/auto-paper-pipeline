"""
科研智能中枢 - Smart Research Center
AI摘要 + 关联网络 + 趋势预警
"""

import json
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict


class AISummaryGenerator:
    """AI深度摘要生成器"""
    
    SUMMARY_TEMPLATE = """【核心问题】{problem}
【创新方法】{method}
【实验结论】{conclusion}
【局限与展望】{limitations}"""
    
    def __init__(self):
        self.cache = {}
    
    def generate_summary(self, paper: Dict) -> str:
        """生成AI摘要"""
        paper_id = paper.get('id', '')
        
        # 检查缓存
        if paper_id in self.cache:
            return self.cache[paper_id]
        
        abstract = paper.get('abstract', '')
        title = paper.get('title', '')
        
        # 基于规则的摘要生成（实际应用中调用LLM）
        problem = self._extract_problem(abstract)
        method = self._extract_method(abstract)
        conclusion = self._extract_conclusion(abstract)
        limitations = self._infer_limitations(abstract, paper)
        
        summary = self.SUMMARY_TEMPLATE.format(
            problem=problem,
            method=method,
            conclusion=conclusion,
            limitations=limitations
        )
        
        # 缓存结果
        self.cache[paper_id] = summary
        return summary
    
    def _extract_problem(self, abstract: str) -> str:
        """提取核心问题（20字以内）"""
        sentences = abstract.split('. ')
        for s in sentences:
            if any(kw in s.lower() for kw in ['challenge', 'problem', 'limitation', 'bottleneck', 'issue', 'difficulty']):
                return s.strip()[:30]
        return sentences[0].strip()[:30] if sentences else "N/A"
    
    def _extract_method(self, abstract: str) -> str:
        """提取创新方法"""
        sentences = abstract.split('. ')
        for s in sentences:
            if any(kw in s.lower() for kw in ['propose', 'introduce', 'present', 'develop', 'novel']):
                return s.strip()[:100]
        return "N/A"
    
    def _extract_conclusion(self, abstract: str) -> str:
        """提取实验结论（量化结果）"""
        sentences = abstract.split('. ')
        for s in sentences:
            if any(kw in s.lower() for kw in ['achieve', 'result', 'outperform', 'improve', 'show']):
                # 提取数字
                numbers = re.findall(r'(\d+\.?\d*)\s*%', s)
                if numbers:
                    return f"性能提升{'/'.join(numbers)}%"
                return s.strip()[:100]
        return "N/A"
    
    def _infer_limitations(self, abstract: str, paper: Dict) -> str:
        """推断局限与展望"""
        limitations = []
        
        if 'dataset' in abstract.lower() and 'large' in abstract.lower():
            limitations.append("可扩展到更大规模数据集")
        
        if 'single' in abstract.lower() or 'one' in abstract.lower():
            limitations.append("可验证更多场景的泛化性")
        
        if paper.get('citation_count', 0) < 10:
            limitations.append("需更多实验验证")
        
        return '；'.join(limitations[:2]) if limitations else "需要进一步验证"


class CitationNetwork:
    """关联网络挖掘"""
    
    def __init__(self, papers: List[Dict]):
        self.papers = {p.get('id'): p for p in papers}
        self.citation_graph = self._build_graph(papers)
    
    def _build_graph(self, papers: List[Dict]) -> Dict:
        """构建引用关系图"""
        graph = defaultdict(lambda: {'cites': [], 'cited_by': []})
        
        # 基于标题和摘要的相似度构建虚拟引用关系
        for i, p1 in enumerate(papers):
            for j, p2 in enumerate(papers):
                if i >= j:
                    continue
                
                # 基于关键词相似度判断是否相关
                similarity = self._compute_similarity(p1, p2)
                if similarity > 0.3:
                    pid1, pid2 = p1.get('id'), p2.get('id')
                    graph[pid1]['cites'].append(pid2)
                    graph[pid2]['cited_by'].append(pid1)
        
        return graph
    
    def _compute_similarity(self, p1: Dict, p2: Dict) -> float:
        """计算论文相似度"""
        text1 = f"{p1.get('title', '')} {p1.get('abstract', '')}".lower()
        text2 = f"{p2.get('title', '')} {p2.get('abstract', '')}".lower()
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def get_related_papers(self, paper_id: str, max_related: int = 3) -> Dict:
        """获取相关论文"""
        refs = self.citation_graph.get(paper_id, {'cites': [], 'cited_by': []})
        
        cites = []
        for ref_id in refs['cites'][:max_related]:
            if ref_id in self.papers:
                p = self.papers[ref_id]
                cites.append({
                    'id': ref_id,
                    'title': p.get('title', 'N/A')[:50],
                    'relation': 'cites'
                })
        
        cited_by = []
        for ref_id in refs['cited_by'][:max_related]:
            if ref_id in self.papers:
                p = self.papers[ref_id]
                cited_by.append({
                    'id': ref_id,
                    'title': p.get('title', 'N/A')[:50],
                    'relation': 'cited_by'
                })
        
        return {
            'paper_id': paper_id,
            'cites': cites,
            'cited_by': cited_by,
            'total_connections': len(cites) + len(cited_by)
        }


class TrendAlertSystem:
    """趋势预警系统"""
    
    def __init__(self, papers: List[Dict]):
        self.papers = papers
        self.topic_history = self._build_history(papers)
    
    def _build_history(self, papers: List[Dict]) -> Dict:
        """构建历史数据"""
        history = defaultdict(list)
        
        for p in papers:
            topic = p.get('topic', 'Unknown')
            date = p.get('published_date', '')
            if date:
                history[topic].append(date)
        
        return history
    
    def analyze_trends(self) -> List[Dict]:
        """分析趋势并生成预警"""
        alerts = []
        
        for topic, dates in self.topic_history.items():
            if len(dates) < 3:
                continue
            
            # 计算7日增长率
            now = datetime.now()
            recent_7d = sum(1 for d in dates if self._parse_date(d) and (now - self._parse_date(d)).days <= 7)
            previous_7d = sum(1 for d in dates if self._parse_date(d) and 
                             7 < (now - self._parse_date(d)).days <= 14)
            
            if previous_7d == 0:
                growth_rate = 100.0 if recent_7d > 0 else 0.0
            else:
                growth_rate = ((recent_7d - previous_7d) / previous_7d) * 100
            
            # 计算波动系数
            daily_counts = self._daily_counts(dates)
            if daily_counts:
                mean_count = np.mean(daily_counts)
                std_count = np.std(daily_counts)
                volatility = std_count / (mean_count + 1e-8)
            else:
                volatility = 0
            
            # 判断预警级别
            alert_level = "LOW"
            if growth_rate > 30 and volatility > 0.5:
                alert_level = "HIGH"
            elif growth_rate > 20:
                alert_level = "MEDIUM"
            
            # 获取关键作者
            key_authors = self._get_key_authors(topic)
            
            alerts.append({
                'topic': topic,
                'growth_rate': f"{growth_rate:+.1f}%",
                'new_papers': recent_7d,
                'key_authors': key_authors,
                'alert_level': alert_level,
                'volatility': round(volatility, 2)
            })
        
        # 按增长率排序
        alerts.sort(key=lambda x: float(x['growth_rate'].replace('%', '').replace('+', '')), reverse=True)
        return alerts
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """解析日期"""
        try:
            return datetime.strptime(date_str[:10], '%Y-%m-%d')
        except:
            return None
    
    def _daily_counts(self, dates: List[str]) -> List[int]:
        """计算每日论文数"""
        daily = defaultdict(int)
        for d in dates:
            parsed = self._parse_date(d)
            if parsed:
                daily[parsed.date()] += 1
        
        if not daily:
            return []
        
        return list(daily.values())
    
    def _get_key_authors(self, topic: str) -> List[str]:
        """获取关键作者"""
        author_counts = defaultdict(int)
        
        for p in self.papers:
            if p.get('topic') == topic:
                for author in p.get('authors', []):
                    name = author.get('name', '') if isinstance(author, dict) else author
                    author_counts[name] += 1
        
        # 返回出现次数最多的前3个作者
        sorted_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)
        return [a[0] for a in sorted_authors[:3]]
