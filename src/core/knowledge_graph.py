"""
知识图谱构建模块
构建论文引用关系图谱
"""

import json
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict


class KnowledgeGraph:
    """论文知识图谱"""
    
    def __init__(self, papers: List[Dict] = None):
        self.papers = papers or []
        self.graph = defaultdict(lambda: {'cites': [], 'cited_by': [], 'related': []})
        self.node_types = {}  # node_id -> type
    
    def build_from_papers(self, papers: List[Dict]):
        """从论文列表构建图谱"""
        self.papers = papers
        
        # 构建引用关系
        for i, p1 in enumerate(papers):
            pid1 = p1.get('id', '')
            self.node_types[pid1] = p1.get('topic', 'paper')
            
            for j, p2 in enumerate(papers):
                if i >= j:
                    continue
                
                pid2 = p2.get('id', '')
                similarity = self._compute_similarity(p1, p2)
                
                if similarity > 0.15:
                    self.graph[pid1]['related'].append(pid2)
                    self.graph[pid2]['related'].append(pid1)
        
        return self.graph
    
    def _compute_similarity(self, p1: Dict, p2: Dict) -> float:
        """计算论文相似度"""
        text1 = f"{p1.get('title', '')} {' '.join(p1.get('llm_tags', []))}"
        text2 = f"{p2.get('title', '')} {' '.join(p2.get('llm_tags', []))}"
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def get_node(self, node_id: str) -> Dict:
        """获取节点信息"""
        return {
            'id': node_id,
            'type': self.node_types.get(node_id, 'paper'),
            'connections': self.graph.get(node_id, {'cites': [], 'cited_by': [], 'related': []})
        }
    
    def get_related_papers(self, node_id: str, max_related: int = 5) -> List[Dict]:
        """获取相关论文"""
        related = self.graph.get(node_id, {}).get('related', [])
        result = []
        
        for rid in related[:max_related]:
            for p in self.papers:
                if p.get('id') == rid:
                    result.append({
                        'id': rid,
                        'title': p.get('title', 'N/A')[:50],
                        'topic': p.get('topic', 'N/A'),
                        'score': p.get('llm_score', 0)
                    })
                    break
        
        return result
    
    def get_subgraph(self, node_id: str, depth: int = 2) -> Dict:
        """获取子图（用于可视化）"""
        nodes = []
        edges = []
        visited = set()
        
        def traverse(nid, d):
            if d > depth or nid in visited:
                return
            visited.add(nid)
            
            node = self.get_node(nid)
            nodes.append({
                'id': nid,
                'label': nid[:20],
                'type': node['type']
            })
            
            for related_id in node['connections']['related']:
                if related_id not in visited:
                    edges.append({'source': nid, 'target': related_id})
                    traverse(related_id, d + 1)
        
        traverse(node_id, 0)
        
        return {'nodes': nodes, 'edges': edges}
    
    def export_d3_json(self) -> Dict:
        """导出D3.js格式"""
        nodes = []
        edges = []
        
        for node_id in self.node_types:
            nodes.append({
                'id': node_id,
                'label': node_id[:15],
                'type': self.node_types[node_id],
                'size': len(self.graph.get(node_id, {}).get('related', []))
            })
        
        seen_edges = set()
        for node_id, connections in self.graph.items():
            for related_id in connections['related']:
                edge_key = tuple(sorted([node_id, related_id]))
                if edge_key not in seen_edges:
                    edges.append({'source': node_id, 'target': related_id})
                    seen_edges.add(edge_key)
        
        return {'nodes': nodes, 'links': edges}
    
    def save(self, filepath: str):
        """保存图谱"""
        data = {
            'nodes': list(self.node_types.keys()),
            'edges': [],
            'metadata': {
                'total_nodes': len(self.node_types),
                'total_edges': sum(len(v['related']) for v in self.graph.values()) // 2
            }
        }
        
        for node_id, connections in self.graph.items():
            for related_id in connections['related']:
                data['edges'].append({'source': node_id, 'target': related_id})
        
        Path(filepath).write_text(json.dumps(data, indent=2))
    
    def load(self, filepath: str):
        """加载图谱"""
        data = json.loads(Path(filepath).read_text())
        self.node_types = {n: 'paper' for n in data.get('nodes', [])}
        
        for edge in data.get('edges', []):
            src, tgt = edge['source'], edge['target']
            self.graph[src]['related'].append(tgt)
            self.graph[tgt]['related'].append(src)
        
        return self
