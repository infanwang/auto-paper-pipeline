"""
AI深度摘要生成器 - 精准版：基于关键词匹配的结构化摘要
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple


class AISummaryGenerator:
    """AI深度摘要生成器"""
    
    def __init__(self, cache_dir: str = "/app/data/summary_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = self._load_cache()
        
        # 关键词-问题映射
        self.problem_keywords = {
            'attention': '解决Transformer注意力机制的计算瓶颈',
            'flash': '解决注意力计算的内存和速度问题',
            'kv cache': '优化LLM推理的KV缓存内存占用',
            'rag': '提升检索增强生成的效率和准确性',
            'agent': '提升AI Agent的任务执行和推理能力',
            'moe': '解决MoE模型的负载均衡和效率问题',
            'quantiz': '降低模型量化后的精度损失',
            'video': '实现高质量视频生成或理解',
            'robot': '提升机器人embodied控制和感知能力',
            'embodied': '实现embodied AI的高效控制',
            'code': '提升代码生成质量和效率',
            'multimodal': '增强多模态跨模态理解能力',
            'vision': '提升视觉理解和表示学习',
            'language': '增强语言理解和生成能力',
            'reasoning': '提升模型推理和逻辑能力',
            'inference': '优化模型推理效率和速度',
            'training': '提升模型训练效率',
            'distill': '通过知识蒸馏压缩模型',
            'prune': '通过剪枝减少模型参数',
            'sparse': '利用稀疏性提升效率',
        }
    
    def _load_cache(self) -> Dict:
        cache_file = self.cache_dir / "summary_cache.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text())
            except:
                return {}
        return {}
    
    def _save_cache(self):
        cache_file = self.cache_dir / "summary_cache.json"
        cache_file.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2))
    
    def generate_summary(self, paper: Dict) -> Tuple[Dict, bool]:
        """生成AI摘要"""
        paper_id = paper.get('id', '')
        
        # 检查缓存
        if paper_id in self.cache:
            return self.cache[paper_id], True
        
        # 预处理
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')[:500]
        text = f"{title} {abstract}".lower()
        
        # 生成摘要
        summary = {
            "core_problem": self._match_problem(text),
            "innovation": self._match_innovation(text),
            "conclusion": self._extract_conclusion(abstract),
            "limitation": self._infer_limitation(text)
        }
        
        # 缓存
        self.cache[paper_id] = summary
        self._save_cache()
        
        return summary, True
    
    def _match_problem(self, text: str) -> str:
        """匹配核心问题"""
        for keyword, problem in self.problem_keywords.items():
            if keyword in text:
                return problem
        return '提升模型性能和效率'
    
    def _match_innovation(self, text: str) -> str:
        """匹配创新方法"""
        innovations = {
            'flash': 'Flash Attention机制优化',
            'sparse': '稀疏注意力和动态路由',
            'mixture': '混合专家动态门控',
            'distill': '知识蒸馏压缩',
            'prune': '结构化剪枝',
            'quantiz': '混合精度量化',
            'adapter': '轻量级适配器微调',
            'lora': '低秩适配高效微调',
            'rag': '检索增强生成优化',
            'agent': '多Agent协作框架',
            'video': '视频扩散模型',
            'robot': '机器人策略学习',
            'embodied': 'Embodied AI控制',
            'code': '代码生成和补全',
            'multimodal': '跨模态融合',
        }
        
        for keyword, innovation in innovations.items():
            if keyword in text:
                return innovation
        
        if 'novel' in text or 'first' in text:
            return '首次提出新方法'
        return '提出创新方案'
    
    def _extract_conclusion(self, abstract: str) -> str:
        """提取实验结论"""
        # 查找具体数字
        numbers = re.findall(r'(\d+\.?\d*)\s*(?:x|×|percent|%)', abstract)
        if numbers:
            return f"性能提升{'/'.join(numbers[:2])}"
        
        if 'outperform' in abstract.lower() or 'surpass' in abstract.lower():
            return '超越现有方法'
        elif 'state-of-the-art' in abstract.lower() or 'sota' in abstract.lower():
            return '达到SOTA水平'
        elif 'improv' in abstract.lower():
            return '显著提升性能'
        
        return '实验验证有效'
    
    def _infer_limitation(self, text: str) -> str:
        """推断局限"""
        if 'dataset' in text and 'small' in text:
            return '扩展到更大规模数据集'
        elif 'real' in text and 'world' in text:
            return '实际部署验证'
        elif 'comput' in text or 'cost' in text:
            return '降低计算成本'
        elif 'generaliz' in text:
            return '提升泛化能力'
        return '进一步实验验证'
    
    def get_summary(self, paper_id: str) -> Optional[Dict]:
        return self.cache.get(paper_id)
    
    def has_summary(self, paper_id: str) -> bool:
        return paper_id in self.cache
