"""测试配置 - 共享fixtures"""

import pytest
import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture
def sample_paper():
    """示例论文数据"""
    return {
        'id': '2607.16193',
        'title': 'Knowing the Self, Understanding the World: A Dual-Cognition Benchmark',
        'abstract': 'We propose a novel benchmark for aerial multiview spatiotemporal reasoning...',
        'authors': [{'name': 'Like Liu'}, {'name': 'Zhengzheng Xu'}],
        'categories': ['cs.CV', 'cs.CL'],
        'published_date': '2026-07-17',
        'topic': 'AI_Agent',
        'citation_count': 0,
        'llm_score': 7.5,
        'llm_tags': ['Agent', '多模态'],
    }

@pytest.fixture
def sample_papers():
    """示例论文列表"""
    return [
        {
            'id': '2607.16193',
            'title': 'UAV-DualCog: Dual-Cognition Benchmark',
            'abstract': 'Multimodal large language models have achieved strong performance...',
            'authors': [{'name': 'Like Liu'}],
            'published_date': '2026-07-17',
            'topic': 'AI_Agent',
            'citation_count': 0,
            'llm_score': 7.5,
        },
        {
            'id': '2607.16189',
            'title': 'VideoTreeSearch: Self-Correcting Agents',
            'abstract': 'Grounded long-video question answering requires answering...',
            'authors': [{'name': 'Ce Zhang'}],
            'published_date': '2026-07-17',
            'topic': 'AI_Agent',
            'citation_count': 5,
            'llm_score': 8.0,
        },
        {
            'id': '2607.16190',
            'title': 'FVAttn: Adaptive Sparse Attention',
            'abstract': 'Video Diffusion Transformers process long spatio-temporal sequences...',
            'authors': [{'name': 'Hao Liu'}],
            'published_date': '2026-07-17',
            'topic': 'LLM推理优化',
            'citation_count': 10,
            'llm_score': 8.5,
        },
        {
            'id': '2607.16154',
            'title': 'CLIFE: Camera-LiDAR Fusion Framework',
            'abstract': 'Camera-LiDAR fusion for edge-deployable roadside VRU perception...',
            'authors': [{'name': 'Yuhong Li'}],
            'published_date': '2026-07-17',
            'topic': '多模态大模型',
            'citation_count': 3,
            'llm_score': 7.0,
        },
        {
            'id': '2607.16161',
            'title': 'ADA-ST: Adaptive Fault Injection',
            'abstract': 'Adaptive fault injection planning for multi-layer self-healing...',
            'authors': [{'name': 'John Smith'}],
            'published_date': '2026-07-17',
            'topic': '代码生成',
            'citation_count': 8,
            'llm_score': 7.8,
        },
    ]

@pytest.fixture
def domain_keywords():
    """领域关键词"""
    return [
        "LLM", "agent", "inference", "optimization", "multimodal",
        "code generation", "verification", "5G", "MIMO", "quantization",
        "transformer", "attention", "neural network", "deep learning"
    ]
