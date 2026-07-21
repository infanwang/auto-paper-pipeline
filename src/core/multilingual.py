"""
多语言支持模块
支持中英文混合搜索
"""

import re
from typing import Dict, List


class MultilingualSupport:
    """多语言支持"""
    
    def __init__(self):
        # 中英文关键词映射
        self.keyword_map = {
            # AI相关
            '大模型': ['LLM', 'large language model', 'language model'],
            '智能体': ['agent', 'agentic', 'multi-agent'],
            '推理优化': ['inference', 'optimization', 'efficient'],
            '多模态': ['multimodal', 'vision', 'language'],
            '代码生成': ['code generation', 'programming', 'software'],
            '芯片验证': ['verification', 'formal', 'hardware'],
            '5G通信': ['5G', 'wireless', 'MIMO'],
            
            # 技术术语
            '注意力机制': ['attention', 'transformer', 'self-attention'],
            '知识蒸馏': ['distillation', 'knowledge distillation'],
            '量化': ['quantization', 'quantize', 'int4', 'int8'],
            '剪枝': ['pruning', 'prune', 'sparse'],
            '微调': ['fine-tuning', 'finetune', 'adaptation'],
            '强化学习': ['reinforcement learning', 'RL', 'policy gradient'],
            
            # 应用场景
            '自然语言处理': ['NLP', 'natural language', 'text'],
            '计算机视觉': ['CV', 'computer vision', 'image'],
            '语音识别': ['ASR', 'speech recognition', 'audio'],
            '机器人': ['robot', 'robotics', 'embodied'],
        }
        
        # 反向映射（英文→中文）
        self.reverse_map = {}
        for cn, en_list in self.keyword_map.items():
            for en in en_list:
                self.reverse_map[en.lower()] = cn
    
    def extract_keywords(self, query: str) -> List[str]:
        """从查询中提取关键词"""
        keywords = []
        
        # 检查中文关键词
        for cn, en_keywords in self.keyword_map.items():
            if cn in query:
                keywords.extend(en_keywords)
        
        # 如果没有匹配到中文，提取英文关键词
        if not keywords:
            # 提取英文单词
            words = re.findall(r'[a-zA-Z]+', query)
            keywords = [w for w in words if len(w) > 2]
        
        return list(set(keywords))
    
    def translate_query(self, query: str) -> str:
        """翻译查询"""
        keywords = self.extract_keywords(query)
        return ' '.join(keywords)
    
    def get_cn_label(self, en_keyword: str) -> str:
        """获取中文标签"""
        return self.reverse_map.get(en_keyword.lower(), en_keyword)
    
    def detect_language(self, text: str) -> str:
        """检测语言"""
        # 简单检测：包含中文字符就是中文
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh'
        return 'en'
    
    def normalize_query(self, query: str) -> List[str]:
        """标准化查询"""
        keywords = self.extract_keywords(query)
        
        # 添加同义词
        expanded = []
        for kw in keywords:
            expanded.append(kw)
            if kw.lower() in self.reverse_map:
                expanded.append(self.reverse_map[kw.lower()])
        
        return list(set(expanded))
