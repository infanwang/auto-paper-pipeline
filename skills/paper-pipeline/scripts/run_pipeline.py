#!/usr/bin/env python3
"""
Auto Paper Pipeline V2.0 - 主调度脚本
用法: python scripts/run_pipeline.py [--mode daily|weekly|full]
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from collectors.arxiv import ArxivCollector, SemanticScholarCollector
from pipeline.funnel import FunnelPipeline
from analyzers.paper_analyzer import PaperAnalyzer, ReproducibilityAssessor
from analyzers.doc_generator import ReportGenerator
from notifiers.email import EmailNotifier


# 研究领域配置
TOPICS = {
    "AI_Agent": {
        "queries": ["LLM agent tool use", "multi-agent collaboration", "agentic reasoning"],
        "categories": ["cs.AI", "cs.CL", "cs.MA"]
    },
    "LLM推理优化": {
        "queries": ["LLM inference optimization", "KV cache efficient", "model quantization"],
        "categories": ["cs.CL", "cs.LG"]
    },
    "多模态大模型": {
        "queries": ["multimodal large language model", "vision language model"],
        "categories": ["cs.CV", "cs.CL"]
    },
    "代码生成": {
        "queries": ["code generation LLM", "AI programming assistant"],
        "categories": ["cs.SE", "cs.CL"]
    },
    "芯片验证": {
        "queries": ["chip verification formal", "hardware verification"],
        "categories": ["cs.AR", "cs.CV"]
    },
    "5G移动通信": {
        "queries": ["5G NR resource allocation", "massive MIMO beamforming"],
        "categories": ["eess.SP", "cs.IT"]
    }
}

# 领域关键词
DOMAIN_KEYWORDS = [
    "LLM", "agent", "inference", "optimization", "multimodal",
    "code generation", "verification", "5G", "MIMO", "quantization",
    "transformer", "attention", "neural network", "deep learning",
    "reinforcement learning", "language model", "benchmark"
]


def run_pipeline(mode: str = "daily"):
    """运行主流程"""
    print("="*60)
    print(f"🚀 Auto Paper Pipeline V2.0 - {mode.upper()} 模式")
    print("="*60)
    
    # Step 1: 论文采集
    print("\n[1/5] 📥 论文采集...")
    arxiv = ArxivCollector()
    papers = arxiv.search_multi_topic(TOPICS, days_back=7)
    print(f"  找到 {len(papers)} 篇论文")
    
    # Step 2: 多源聚合
    print("\n[2/5] 🔗 Semantic Scholar 数据增强...")
    ss = SemanticScholarCollector()
    papers = ss.batch_enrich(papers, limit=30)
    print(f"  增强了 {min(30, len(papers))} 篇论文")
    
    # Step 3: 多级过滤漏斗
    print("\n[3/5] 🔍 多级过滤漏斗...")
    funnel = FunnelPipeline(DOMAIN_KEYWORDS)
    papers = funnel.run(papers, tfidf_threshold=0.2, llm_top_n=20)
    print(f"  最终筛选: {len(papers)} 篇")
    
    # Step 4: 深度分析
    print("\n[4/5] 📊 论文深度分析...")
    analyzer = PaperAnalyzer()
    assessor = ReproducibilityAssessor()
    
    for p in papers[:10]:
        analysis = analyzer.analyze(p)
        reproducibility = assessor.assess(p, analysis)
        p['analysis'] = analysis
        p['reproducibility'] = reproducibility
    
    # Step 5: 生成报告
    print("\n[5/5] 📝 生成报告...")
    today = datetime.now().strftime("%Y-%m-%d")
    
    # 保存数据
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    output_data = {
        'date': today,
        'total_papers': len(papers),
        'by_topic': {},
        'papers': papers
    }
    
    for p in papers:
        topic = p.get('topic', 'Unknown')
        output_data['by_topic'][topic] = output_data['by_topic'].get(topic, 0) + 1
    
    output_path = output_dir / f"pipeline_{today}.json"
    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
    
    # 生成静态站点
    print("\n🌐 生成静态站点...")
    from web.static_generator import StaticSiteGenerator
    site_gen = StaticSiteGenerator("docs")
    site_gen.generate(papers, today)
    
    # 生成报告
    print("\n📄 生成报告...")
    report_gen = ReportGenerator("reports")
    report_gen.generate_full_report(papers, today)
    
    # 发送邮件
    print("\n📧 发送邮件简报...")
    notifier = EmailNotifier()
    notifier.send_daily_briefing(papers, today)
    
    print("\n" + "="*60)
    print(f"✅ 完成! 共处理 {len(papers)} 篇论文")
    print(f"📁 数据: {output_path}")
    print(f"🌐 静态站点: docs/")
    print(f"📄 报告: reports/")
    print("="*60)
    
    return papers


def main():
    parser = argparse.ArgumentParser(description="Auto Paper Pipeline V2.0")
    parser.add_argument("--mode", choices=["daily", "weekly", "full"], 
                       default="daily", help="运行模式")
    args = parser.parse_args()
    
    run_pipeline(args.mode)


if __name__ == "__main__":
    main()
