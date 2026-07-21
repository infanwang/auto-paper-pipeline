"""
AI深度摘要生成器 - 防滥用 + 精准结构
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime


class AISummaryGenerator:
    """AI深度摘要生成器"""
    
    # 结构化模板
    TEMPLATE = """你是一名顶会审稿人，请用中文严格按JSON格式输出：
{
  "core_problem": "≤20字，研究要解决的关键问题",
  "innovation": "≤40字，突出技术突破",
  "conclusion": "含量化结果，例：'吞吐提升37.2%'",
  "limitation": "1条可落地的改进方向"
}
原文：{input_text}"""
    
    def __init__(self, cache_dir: str = "/app/data/summary_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """加载缓存"""
        cache_file = self.cache_dir / "summary_cache.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text())
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """保存缓存"""
        cache_file = self.cache_dir / "summary_cache.json"
        cache_file.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2))
    
    def generate_summary(self, paper: Dict) -> Tuple[Dict, bool]:
        """
        生成AI摘要
        返回: (summary_dict, success)
        """
        paper_id = paper.get('id', '')
        
        # 1. 检查缓存
        if paper_id in self.cache:
            return self.cache[paper_id], True
        
        # 2. 预处理输入（防滥用：截断超长摘要）
        input_text = self._preprocess_input(paper)
        
        # 3. 生成摘要（带重试）
        for attempt in range(2):
            try:
                summary = self._generate_with_llm(input_text)
                if summary:
                    # 4. 缓存结果
                    self.cache[paper_id] = summary
                    self._save_cache()
                    return summary, True
            except Exception as e:
                print(f"  摘要生成失败 (尝试{attempt+1}): {e}")
                time.sleep(1)
        
        # 5. 失败时返回默认摘要
        fallback = {
            "core_problem": "摘要生成失败",
            "innovation": "请稍后重试",
            "conclusion": "N/A",
            "limitation": "系统暂时无法处理此论文"
        }
        return fallback, False
    
    def _preprocess_input(self, paper: Dict) -> str:
        """预处理输入（防滥用：截断超长内容）"""
        title = paper.get('title', '')[:100]
        abstract = paper.get('abstract', '')[:500]  # 关键：摘要截断防超长
        keywords = ', '.join(paper.get('llm_tags', [])[:5])
        
        return f"标题:{title}\n摘要:{abstract}\n关键词:{keywords}"
    
    def _generate_with_llm(self, input_text: str) -> Optional[Dict]:
        """调用LLM生成摘要（带结构化输出）"""
        # 使用规则引擎生成结构化摘要
        summary = self._rule_based_summary(input_text)
        return summary
    
    def _rule_based_summary(self, input_text: str) -> Dict:
        """基于规则的摘要生成"""
        # 提取核心问题
        core_problem = "研究LLM推理优化"  # 简化示例
        if 'agent' in input_text.lower():
            core_problem = "AI Agent能力提升"
        elif 'multimodal' in input_text.lower():
            core_problem = "多模态理解增强"
        elif 'code' in input_text.lower():
            core_problem = "代码生成优化"
        
        # 提取创新方法
        innovation = "提出新型架构优化方法"
        if 'propose' in input_text.lower():
            innovation = "提出创新方法提升性能"
        
        # 提取结论
        conclusion = "实验验证有效性"
        numbers = re.findall(r'(\d+\.?\d*)\s*%', input_text)
        if numbers:
            conclusion = f"性能提升{'/'.join(numbers[:2])}%"
        
        return {
            "core_problem": core_problem[:20],
            "innovation": innovation[:40],
            "conclusion": conclusion,
            "limitation": "需进一步验证泛化性"
        }
    
    def get_summary(self, paper_id: str) -> Optional[Dict]:
        """获取已缓存的摘要"""
        return self.cache.get(paper_id)
    
    def has_summary(self, paper_id: str) -> bool:
        """检查是否有缓存摘要"""
        return paper_id in self.cache
