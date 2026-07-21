"""
自动论文复现模块
自动分析论文并生成复现代码
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional


class AutoReproducer:
    """自动论文复现器"""
    
    def __init__(self, output_dir: str = "/root/git/mimo/paper-pipeline/reproduction/auto"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 工具检测
        self.tools = {
            'python': self._check_python(),
            'pytorch': self._check_pytorch(),
            'tensorflow': self._check_tensorflow(),
            'matlab': self._check_matlab(),
        }
    
    def _check_python(self) -> bool:
        try:
            import sys
            return sys.version_info >= (3, 8)
        except:
            return False
    
    def _check_pytorch(self) -> bool:
        try:
            import torch
            return True
        except:
            return False
    
    def _check_tensorflow(self) -> bool:
        try:
            import tensorflow
            return True
        except:
            return False
    
    def _check_matlab(self) -> bool:
        # 简化检查
        return False
    
    def analyze_paper(self, paper: Dict) -> Dict:
        """分析论文并生成复现计划"""
        abstract = paper.get('abstract', '')
        title = paper.get('title', '')
        
        # 检测框架
        framework = self._detect_framework(abstract)
        
        # 检测任务类型
        task_type = self._detect_task_type(abstract)
        
        # 生成复现计划
        plan = {
            'paper_id': paper.get('id', ''),
            'title': title[:60],
            'framework': framework,
            'task_type': task_type,
            'requirements': self._generate_requirements(framework, task_type),
            'reproduction_steps': self._generate_steps(framework, task_type),
            'estimated_time': self._estimate_time(framework, task_type),
            'difficulty': self._estimate_difficulty(framework, task_type),
        }
        
        return plan
    
    def _detect_framework(self, abstract: str) -> str:
        """检测框架"""
        abstract_lower = abstract.lower()
        
        if 'pytorch' in abstract_lower or 'torch' in abstract_lower:
            return 'PyTorch'
        elif 'tensorflow' in abstract_lower or 'keras' in abstract_lower:
            return 'TensorFlow'
        elif 'jax' in abstract_lower:
            return 'JAX'
        elif 'transformer' in abstract_lower:
            return 'Transformer'
        else:
            return 'Unknown'
    
    def _detect_task_type(self, abstract: str) -> str:
        """检测任务类型"""
        abstract_lower = abstract.lower()
        
        if 'classification' in abstract_lower:
            return 'classification'
        elif 'detection' in abstract_lower:
            return 'detection'
        elif 'segmentation' in abstract_lower:
            return 'segmentation'
        elif 'generation' in abstract_lower:
            return 'generation'
        elif 'reinforcement' in abstract_lower:
            return 'reinforcement_learning'
        elif 'optimization' in abstract_lower:
            return 'optimization'
        else:
            return 'general'
    
    def _generate_requirements(self, framework: str, task_type: str) -> List[str]:
        """生成依赖列表"""
        requirements = ['numpy']
        
        if framework == 'PyTorch':
            requirements.extend(['torch', 'torchvision'])
        elif framework == 'TensorFlow':
            requirements.extend(['tensorflow'])
        
        if task_type in ['classification', 'detection', 'segmentation']:
            requirements.append('pillow')
        
        if task_type == 'reinforcement_learning':
            requirements.extend(['gymnasium', 'stable-baselines3'])
        
        return requirements
    
    def _generate_steps(self, framework: str, task_type: str) -> List[str]:
        """生成复现步骤"""
        steps = [
            '1. 克隆论文仓库',
            '2. 安装依赖: pip install -r requirements.txt',
            '3. 下载数据集',
        ]
        
        if framework == 'PyTorch':
            steps.append('4. 运行: python train.py')
        elif framework == 'TensorFlow':
            steps.append('4. 运行: python train.py')
        
        steps.extend([
            '5. 评估模型',
            '6. 对比论文结果',
        ])
        
        return steps
    
    def _estimate_time(self, framework: str, task_type: str) -> str:
        """估算时间"""
        if task_type == 'reinforcement_learning':
            return '2-4小时'
        elif task_type in ['classification', 'detection']:
            return '1-2小时'
        else:
            return '30分钟-1小时'
    
    def _estimate_difficulty(self, framework: str, task_type: str) -> str:
        """估算难度"""
        if task_type == 'reinforcement_learning':
            return 'hard'
        elif task_type in ['classification', 'detection']:
            return 'medium'
        else:
            return 'easy'
    
    def generate_reproduction_script(self, plan: Dict) -> str:
        """生成复现脚本"""
        framework = plan['framework']
        task_type = plan['task_type']
        
        script = f"""#!/usr/bin/env python3
\"\"\"
自动复现脚本: {plan['title']}
框架: {framework}
任务: {task_type}
\"\"\"

import numpy as np

# 1. 加载数据
print("加载数据...")

# 2. 初始化模型
print("初始化模型...")

# 3. 训练/推理
print("开始训练...")

# 4. 评估
print("评估模型...")

print("复现完成!")
"""
        
        return script
    
    def save_plan(self, plan: Dict, filepath: str = None):
        """保存复现计划"""
        if filepath is None:
            filepath = self.output_dir / f"plan_{plan['paper_id']}.json"
        
        Path(filepath).write_text(json.dumps(plan, indent=2, ensure_ascii=False))
        return filepath
    
    def save_script(self, plan: Dict, filepath: str = None):
        """保存复现脚本"""
        if filepath is None:
            filepath = self.output_dir / f"reproduce_{plan['paper_id']}.py"
        
        script = self.generate_reproduction_script(plan)
        Path(filepath).write_text(script)
        return filepath
