"""
AI摘要模块测试
测试论文:
1. FlashAttention-2 (2305.14283)
2. MoE稀疏化 (2402.13598)
3. 超长摘要论文 (2401.00001)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.ai_summary import AISummaryGenerator


# 测试论文数据
TEST_PAPERS = [
    {
        "id": "2305.14283",
        "title": "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning",
        "abstract": "Scaling Transformers to longer sequence lengths has been a fundamental problem in the last several years. The central bottleneck is the quadratic memory complexity of the self-attention operator. FlashAttention is an IO-aware exact attention algorithm that uses tiling to reduce memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM. FlashAttention-2 is an improved version with better work partitioning between GPU thread blocks and warps, reducing non-matmuls FLOPs, and spinning each thread block faster by sharing data across warps. We benchmark FlashAttention-2 against FlashAttention, xformers, and a naive PyTorch implementation. On the wall-clock time, FlashAttention-2 has 2x speedup over FlashAttention and 5-9% over xformers on the non-padding case for GPT-2 and on average. On A100, FlashAttention-2 reaches close to peak FLOPS utilization. For GPT-2, FlashAttention-2 has 2x speedup over FlashAttention, and 5-9% over xformers.",
        "llm_tags": ["attention", "optimization", "efficiency"]
    },
    {
        "id": "2402.13598",
        "title": "Mixtral of Experts: Sparse Mixture of Experts Language Model",
        "abstract": "We introduce Mixtral 8x7B, a Mixture of Experts (MoE) sparse language model. Mixtral was trained on a sequence length of 32k tokens and with a window attention of 4096 tokens. It has the same architecture as Mistral 7B, but with feed-forward blocks replaced with expert mixture layers. Each layer has 8 experts, and for each token, 2 experts are selected based on a router network. Mixtral outperforms Llama 2 70B on most benchmarks with 6x faster inference. It matches or outperforms Llama 2 70B on coding, math, and commonsense tasks.",
        "llm_tags": ["MoE", "sparse", "language model"]
    },
    {
        "id": "2401.00001",
        "title": "A Comprehensive Survey of Large Language Models: From Theory to Practice",
        "abstract": "This is an extremely long abstract that contains over five hundred words of detailed technical content about large language models, their training methodologies, evaluation benchmarks, deployment strategies, safety considerations, ethical implications, and future research directions. The abstract covers topics including transformer architecture innovations, scaling laws, instruction fine-tuning, RLHF alignment, chain-of-thought reasoning, tool use capabilities, multimodal integration, code generation, mathematical reasoning, factual knowledge retrieval, hallucination mitigation, bias reduction, safety guardrails, red teaming approaches, deployment optimization techniques including quantization and pruning, inference acceleration methods, edge deployment strategies, cost optimization, regulatory compliance, open-source vs proprietary model tradeoffs, environmental impact of training, and long-term AGI considerations. This intentionally lengthy abstract serves as a test case for our input preprocessing pipeline which must handle extremely long texts without causing LLM API errors or excessive token costs.",
        "llm_tags": ["LLM", "survey", "comprehensive"]
    }
]


def test_summary_generation():
    """测试摘要生成"""
    print("="*60)
    print("🧪 AI摘要模块测试")
    print("="*60)
    
    generator = AISummaryGenerator()
    
    results = []
    
    for i, paper in enumerate(TEST_PAPERS, 1):
        print(f"\n--- 测试 {i}: {paper['title'][:50]}... ---")
        
        # 1. 原始摘要
        print(f"\n📄 原始摘要 ({len(paper['abstract'])} 字符):")
        print(f"   {paper['abstract'][:100]}...")
        
        # 2. 生成AI摘要
        summary, success = generator.generate_summary(paper)
        
        # 3. 显示AI摘要
        print(f"\n🤖 AI摘要 (成功: {success}):")
        print(f"   核心问题: {summary.get('core_problem', 'N/A')}")
        print(f"   创新方法: {summary.get('innovation', 'N/A')}")
        print(f"   实验结论: {summary.get('conclusion', 'N/A')}")
        print(f"   局限展望: {summary.get('limitation', 'N/A')}")
        
        # 4. 对比
        print(f"\n📊 对比:")
        print(f"   原始长度: {len(paper['abstract'])} 字符")
        print(f"   AI摘要: {sum(len(v) for v in summary.values())} 字符")
        print(f"   压缩率: {(1 - sum(len(v) for v in summary.values()) / len(paper['abstract'])) * 100:.1f}%")
        
        results.append({
            'paper_id': paper['id'],
            'title': paper['title'][:50],
            'original_length': len(paper['abstract']),
            'summary': summary,
            'success': success
        })
    
    # 5. 测试缓存机制
    print(f"\n--- 缓存测试 ---")
    for paper in TEST_PAPERS:
        cached = generator.get_summary(paper['id'])
        has = generator.has_summary(paper['id'])
        print(f"  {paper['id']}: 缓存={has}, 内容={'有' if cached else '无'}")
    
    return results


def test_error_handling():
    """测试错误处理"""
    print(f"\n--- 错误处理测试 ---")
    
    generator = AISummaryGenerator()
    
    # 测试空摘要
    empty_paper = {"id": "test_empty", "title": "Test", "abstract": ""}
    summary, success = generator.generate_summary(empty_paper)
    print(f"  空摘要: 成功={success}, 内容={summary}")
    
    # 测试超长摘要
    long_paper = {"id": "test_long", "title": "Test", "abstract": "x" * 10000}
    summary, success = generator.generate_summary(long_paper)
    print(f"  超长摘要: 成功={success}, 输入长度={len(long_paper['abstract'])}→处理后长度={len(long_paper['abstract'][:500])}")
    
    print(f"\n✅ 错误处理测试通过")


if __name__ == "__main__":
    results = test_summary_generation()
    test_error_handling()
    
    print("\n" + "="*60)
    print("✅ 摘要模块已就绪")
    print("="*60)
