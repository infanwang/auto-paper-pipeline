"""静态站点生成器 - V2.0"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict


class StaticSiteGenerator:
    """生成静态HTML站点"""
    
    def __init__(self, output_dir: str = "docs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(self, papers: List[Dict], date: str):
        """生成完整静态站点"""
        # 生成主页
        self._generate_index(papers, date)
        
        # 按领域生成分类页面
        by_topic = {}
        for p in papers:
            topic = p.get('topic', 'Unknown')
            by_topic.setdefault(topic, []).append(p)
        
        for topic, topic_papers in by_topic.items():
            self._generate_topic_page(topic, topic_papers, date)
        
        # 生成论文详情页
        for p in papers:
            self._generate_paper_page(p)
        
        print(f"  静态站点已生成: {self.output_dir}")
    
    def _generate_index(self, papers: List[Dict], date: str):
        """生成主页"""
        html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Paper Pipeline - {date}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px 20px; text-align: center; }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header p {{ opacity: 0.9; font-size: 1.1em; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }}
        .stat-card {{ background: white; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .stat-number {{ font-size: 2.5em; font-weight: bold; color: #667eea; }}
        .stat-label {{ color: #666; margin-top: 5px; }}
        .papers-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 20px; }}
        .paper-card {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); transition: transform 0.2s; }}
        .paper-card:hover {{ transform: translateY(-5px); }}
        .paper-title {{ font-size: 1.1em; font-weight: 600; color: #1a1a2e; margin-bottom: 10px; }}
        .paper-meta {{ color: #666; font-size: 0.9em; margin-bottom: 10px; }}
        .paper-abstract {{ color: #444; font-size: 0.95em; line-height: 1.5; }}
        .paper-tags {{ margin-top: 10px; }}
        .tag {{ display: inline-block; background: #e8eaf6; color: #3f51b5; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; margin-right: 5px; }}
        .topic-filter {{ margin: 20px 0; }}
        .topic-btn {{ display: inline-block; padding: 8px 16px; margin: 5px; border: 2px solid #667eea; border-radius: 20px; cursor: pointer; transition: all 0.2s; }}
        .topic-btn:hover, .topic-btn.active {{ background: #667eea; color: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📚 AI Paper Pipeline</h1>
        <p>AI驱动的论文研究自动化平台 | {date}</p>
    </div>
    
    <div class="container">
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{len(papers)}</div>
                <div class="stat-label">论文总数</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(set(p.get('topic') for p in papers))}</div>
                <div class="stat-label">研究领域</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{sum(1 for p in papers if p.get('llm_score', 0) >= 7)}</div>
                <div class="stat-label">高分论文</div>
            </div>
        </div>
        
        <div class="topic-filter">
            <span class="topic-btn active" onclick="filterPapers('all')">全部</span>
"""
        
        # 添加领域按钮
        topics = set(p.get('topic') for p in papers)
        for topic in sorted(topics):
            html += f'            <span class="topic-btn" onclick="filterPapers(\'{topic}\')">{topic}</span>\n'
        
        html += """        </div>
        
        <div class="papers-grid" id="papers-grid">
"""
        
        # 添加论文卡片
        for i, p in enumerate(papers[:30]):  # 只显示前30篇
            title = p.get('title', 'N/A')[:80]
            topic = p.get('topic', 'Unknown')
            date_pub = p.get('published_date', 'N/A')
            abstract = p.get('abstract', '')[:150]
            score = p.get('llm_score', 0)
            tags = p.get('llm_tags', [])
            
            html += f"""
            <div class="paper-card" data-topic="{topic}">
                <div class="paper-title">{title}</div>
                <div class="paper-meta">
                    📁 {topic} | 📅 {date_pub} | ⭐ {score:.1f}/10
                </div>
                <div class="paper-abstract">{abstract}...</div>
                <div class="paper-tags">
"""
            for tag in tags[:3]:
                html += f'                    <span class="tag">{tag}</span>\n'
            
            html += """                </div>
            </div>
"""
        
        html += """        </div>
    </div>
    
    <script>
    function filterPapers(topic) {
        const cards = document.querySelectorAll('.paper-card');
        const btns = document.querySelectorAll('.topic-btn');
        
        btns.forEach(btn => btn.classList.remove('active'));
        event.target.classList.add('active');
        
        cards.forEach(card => {
            if (topic === 'all' || card.dataset.topic === topic) {
                card.style.display = 'block';
            } else {
                card.style.display = 'none';
            }
        });
    }
    </script>
</body>
</html>"""
        
        (self.output_dir / "index.html").write_text(html, encoding='utf-8')
    
    def _generate_topic_page(self, topic: str, papers: List[Dict], date: str):
        """生成领域页面"""
        topic_dir = self.output_dir / topic
        topic_dir.mkdir(exist_ok=True)
        
        html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="utf-8">
    <title>{topic} - AI Paper Pipeline</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #1a1a2e; border-bottom: 2px solid #667eea; }}
        .paper {{ background: #f9f9f9; border-radius: 8px; padding: 15px; margin: 10px 0; }}
        .paper h3 {{ color: #333; }}
        .meta {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>📁 {topic} ({len(papers)}篇)</h1>
    <a href="../index.html">← 返回首页</a>
"""
        
        for i, p in enumerate(papers, 1):
            html += f"""
    <div class="paper">
        <h3>{i}. {p.get('title', 'N/A')}</h3>
        <div class="meta">
            ArXiv: {p.get('id', 'N/A')} | 日期: {p.get('published_date', 'N/A')} | 评分: {p.get('llm_score', 0):.1f}
        </div>
        <p>{p.get('abstract', '')[:200]}...</p>
    </div>
"""
        
        html += "</body></html>"
        (topic_dir / "index.html").write_text(html, encoding='utf-8')
    
    def _generate_paper_page(self, paper: Dict):
        """生成论文详情页"""
        paper_dir = self.output_dir / "papers"
        paper_dir.mkdir(exist_ok=True)
        
        paper_id = paper.get('id', 'unknown')
        html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="utf-8">
    <title>{paper.get('title', 'N/A')} - AI Paper Pipeline</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #1a1a2e; }}
        .meta {{ color: #666; margin: 10px 0; }}
        .abstract {{ background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .analysis {{ background: #e8f5e9; padding: 20px; border-radius: 8px; margin: 20px 0; }}
    </style>
</head>
<body>
    <a href="../index.html">← 返回首页</a>
    <h1>{paper.get('title', 'N/A')}</h1>
    <div class="meta">
        ArXiv: {paper.get('id', 'N/A')} | 日期: {paper.get('published_date', 'N/A')}
    </div>
    <div class="abstract">
        <h3>摘要</h3>
        <p>{paper.get('abstract', 'N/A')}</p>
    </div>
"""
        
        # 添加分析结果
        analysis = paper.get('analysis', {})
        if analysis:
            html += """
    <div class="analysis">
        <h3>深度分析</h3>
"""
            if analysis.get('problem'):
                html += f"<p><strong>问题:</strong> {analysis['problem']}</p>\n"
            if analysis.get('method'):
                html += f"<p><strong>方法:</strong> {analysis['method']}</p>\n"
            if analysis.get('strengths'):
                html += f"<p><strong>优点:</strong> {', '.join(analysis['strengths'])}</p>\n"
            if analysis.get('improvements'):
                html += f"<p><strong>改进建议:</strong> {', '.join(analysis['improvements'])}</p>\n"
            html += "    </div>\n"
        
        html += "</body></html>"
        (paper_dir / f"{paper_id}.html").write_text(html, encoding='utf-8')
