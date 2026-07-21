"""静态站点生成器 - V2.0 美化版"""

import json
from pathlib import Path
from typing import List, Dict


class StaticSiteGenerator:
    """生成静态HTML站点"""
    
    def __init__(self, output_dir: str = "docs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(self, papers: List[Dict], date: str):
        """生成完整静态站点"""
        self._generate_index(papers, date)
        
        by_topic = {}
        for p in papers:
            topic = p.get('topic', 'Unknown')
            by_topic.setdefault(topic, []).append(p)
        
        for topic, topic_papers in by_topic.items():
            self._generate_topic_page(topic, topic_papers, date)
        
        for p in papers:
            self._generate_paper_page(p)
        
        print(f"  静态站点已生成: {self.output_dir}")
    
    def _generate_index(self, papers: List[Dict], date: str):
        """生成主页"""
        topics = {}
        for p in papers:
            topic = p.get('topic', 'Unknown')
            topics.setdefault(topic, []).append(p)
        
        topic_stats = ""
        for topic, tp in topics.items():
            topic_stats += f'<div class="stat-card"><div class="stat-number">{len(tp)}</div><div class="stat-label">{topic}</div></div>\n'
        
        topic_buttons = ""
        for topic in sorted(topics.keys()):
            topic_buttons += f'<button class="topic-btn" onclick="filterPapers(\'{topic}\')">{topic}</button>\n'
        
        paper_cards = ""
        for p in papers[:30]:
            title = p.get('title', 'N/A')[:80]
            topic = p.get('topic', 'Unknown')
            date_pub = p.get('published_date', 'N/A')
            abstract = p.get('abstract', '')[:120]
            score = p.get('llm_score', 0)
            tags = p.get('llm_tags', [])
            arxiv_id = p.get('id', '')
            
            score_color = '#4CAF50' if score >= 7 else ('#FFC107' if score >= 5 else '#f44336')
            
            tags_html = ''.join(f'<span class="tag">{t}</span>' for t in tags[:3])
            
            paper_cards += f'''
            <div class="paper-card" data-topic="{topic}">
                <div class="paper-header">
                    <span class="paper-score" style="background:{score_color}">{score:.1f}</span>
                    <span class="paper-topic">{topic}</span>
                </div>
                <div class="paper-title"><a href="papers/{arxiv_id}.html">{title}</a></div>
                <div class="paper-date">📅 {date_pub}</div>
                <div class="paper-abstract">{abstract}...</div>
                <div class="paper-tags">{tags_html}</div>
            </div>'''
        
        html = f'''<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Paper Pipeline - {date}</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📚</text></svg>">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); min-height: 100vh; }}
        
        .header {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; padding: 50px 20px; text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }}
        .header h1 {{ font-size: 2.8em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }}
        .header p {{ opacity: 0.95; font-size: 1.2em; }}
        .header .subtitle {{ font-size: 0.9em; opacity: 0.8; margin-top: 10px; }}
        
        .container {{ max-width: 1400px; margin: 0 auto; padding: 30px 20px; }}
        
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 20px; margin: -40px 0 30px; position: relative; z-index: 10; }}
        .stat-card {{ 
            background: white; border-radius: 16px; padding: 25px 15px; text-align: center; 
            box-shadow: 0 8px 30px rgba(0,0,0,0.12); transition: transform 0.3s;
        }}
        .stat-card:hover {{ transform: translateY(-5px); box-shadow: 0 12px 40px rgba(0,0,0,0.15); }}
        .stat-number {{ font-size: 2.8em; font-weight: 800; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .stat-label {{ color: #666; margin-top: 8px; font-size: 0.95em; }}
        
        .section {{ background: white; border-radius: 16px; padding: 25px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); }}
        .section-title {{ font-size: 1.4em; color: #1a1a2e; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #667eea; }}
        
        .filter-bar {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; }}
        .topic-btn {{ 
            padding: 10px 20px; border: 2px solid #667eea; border-radius: 25px; 
            cursor: pointer; transition: all 0.3s; background: white; font-weight: 500;
        }}
        .topic-btn:hover {{ background: #667eea; color: white; transform: scale(1.05); }}
        .topic-btn.active {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; border-color: transparent; }}
        
        .papers-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 20px; }}
        
        .paper-card {{ 
            background: white; border-radius: 12px; padding: 20px; 
            box-shadow: 0 2px 12px rgba(0,0,0,0.08); transition: all 0.3s;
            border-left: 4px solid #667eea;
        }}
        .paper-card:hover {{ transform: translateY(-3px); box-shadow: 0 8px 25px rgba(0,0,0,0.12); }}
        
        .paper-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
        .paper-score {{ padding: 4px 12px; border-radius: 20px; color: white; font-weight: bold; font-size: 0.9em; }}
        .paper-topic {{ color: #666; font-size: 0.85em; }}
        .paper-title {{ font-size: 1.05em; font-weight: 600; color: #1a1a2e; margin-bottom: 8px; line-height: 1.4; }}
        .paper-title a {{ color: inherit; text-decoration: none; }}
        .paper-title a:hover {{ color: #667eea; }}
        .paper-date {{ color: #888; font-size: 0.85em; margin-bottom: 8px; }}
        .paper-abstract {{ color: #555; font-size: 0.9em; line-height: 1.6; }}
        .paper-tags {{ margin-top: 12px; }}
        .tag {{ display: inline-block; background: linear-gradient(135deg, #e8eaf6, #f3e5f5); color: #5c6bc0; padding: 4px 10px; border-radius: 12px; font-size: 0.8em; margin-right: 6px; }}
        
        .footer {{ text-align: center; padding: 30px; color: #888; font-size: 0.9em; }}
        .footer a {{ color: #667eea; }}
        
        @media (max-width: 768px) {{
            .papers-grid {{ grid-template-columns: 1fr; }}
            .stats {{ grid-template-columns: repeat(2, 1fr); }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📚 AI Paper Pipeline</h1>
        <p>AI驱动的论文研究自动化平台</p>
        <div class="subtitle">📅 {date} | 🔍 {len(papers)}篇论文 | 📊 {len(topics)}个领域</div>
    </div>
    
    <div class="container">
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{len(papers)}</div>
                <div class="stat-label">论文总数</div>
            </div>
            {topic_stats}
            <div class="stat-card">
                <div class="stat-number">{sum(1 for p in papers if p.get('llm_score', 0) >= 7)}</div>
                <div class="stat-label">高分论文</div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">🔍 筛选领域</div>
            <div class="filter-bar">
                <button class="topic-btn active" onclick="filterPapers('all')">全部</button>
                {topic_buttons}
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">📄 论文列表</div>
            <div class="papers-grid" id="papers-grid">
                {paper_cards}
            </div>
        </div>
    </div>
    
    <div class="footer">
        <p>Powered by <a href="https://github.com/infanwang/auto-paper-pipeline">Auto Paper Pipeline V2.0</a></p>
        <p>数据来源: ArXiv + Semantic Scholar</p>
    </div>
    
    <script>
    function filterPapers(topic) {{
        const cards = document.querySelectorAll('.paper-card');
        const btns = document.querySelectorAll('.topic-btn');
        btns.forEach(btn => btn.classList.remove('active'));
        event.target.classList.add('active');
        cards.forEach(card => {{
            card.style.display = (topic === 'all' || card.dataset.topic === topic) ? 'block' : 'none';
        }});
    }}
    </script>
</body>
</html>'''
        
        (self.output_dir / "index.html").write_text(html, encoding='utf-8')
    
    def _generate_topic_page(self, topic: str, papers: List[Dict], date: str):
        """生成领域页面"""
        topic_dir = self.output_dir / topic
        topic_dir.mkdir(exist_ok=True)
        
        paper_rows = ""
        for i, p in enumerate(papers, 1):
            score = p.get('llm_score', 0)
            score_color = '#4CAF50' if score >= 7 else ('#FFC107' if score >= 5 else '#f44336')
            paper_rows += f'''
            <tr>
                <td>{i}</td>
                <td><a href="../papers/{p.get('id', '')}.html">{p.get('title', 'N/A')[:60]}</a></td>
                <td>{p.get('published_date', 'N/A')}</td>
                <td><span style="background:{score_color};color:white;padding:3px 8px;border-radius:10px;">{score:.1f}</span></td>
            </tr>'''
        
        html = f'''<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="utf-8">
    <title>{topic} - AI Paper Pipeline</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 30px; border-radius: 12px; margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        th {{ background: #667eea; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 12px; border-bottom: 1px solid #eee; }}
        tr:hover {{ background: #f5f5f5; }}
        a {{ color: #667eea; text-decoration: none; }}
        .back {{ display: inline-block; margin-bottom: 20px; color: #667eea; }}
    </style>
</head>
<body>
    <a class="back" href="../index.html">← 返回首页</a>
    <div class="header">
        <h1>📁 {topic}</h1>
        <p>{len(papers)}篇论文</p>
    </div>
    <table>
        <tr><th>#</th><th>标题</th><th>日期</th><th>评分</th></tr>
        {paper_rows}
    </table>
</body>
</html>'''
        (topic_dir / "index.html").write_text(html, encoding='utf-8')
    
    def _generate_paper_page(self, paper: Dict):
        """生成论文详情页"""
        paper_dir = self.output_dir / "papers"
        paper_dir.mkdir(exist_ok=True)
        
        paper_id = paper.get('id', 'unknown')
        analysis = paper.get('analysis', {})
        
        analysis_html = ""
        if analysis:
            analysis_html = '<div class="section"><h2>📊 深度分析</h2>'
            for key, label in [('problem', '问题'), ('method', '方法'), ('strengths', '优点'), ('improvements', '改进建议')]:
                val = analysis.get(key)
                if val:
                    if isinstance(val, list):
                        val = ', '.join(val)
                    analysis_html += f'<p><strong>{label}:</strong> {val}</p>'
            analysis_html += '</div>'
        
        html = f'''<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="utf-8">
    <title>{paper.get('title', 'N/A')}</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .paper {{ background: white; border-radius: 12px; padding: 30px; box-shadow: 0 2px 15px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a1a2e; line-height: 1.4; }}
        .meta {{ color: #666; margin: 15px 0; padding: 10px; background: #f8f9fa; border-radius: 8px; }}
        .abstract {{ background: #f0f4ff; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #667eea; }}
        .section {{ margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
        .back {{ display: inline-block; margin-bottom: 20px; color: #667eea; text-decoration: none; }}
    </style>
</head>
<body>
    <a class="back" href="../index.html">← 返回首页</a>
    <div class="paper">
        <h1>{paper.get('title', 'N/A')}</h1>
        <div class="meta">
            <strong>ArXiv:</strong> {paper.get('id', 'N/A')} | 
            <strong>日期:</strong> {paper.get('published_date', 'N/A')} | 
            <strong>评分:</strong> {paper.get('llm_score', 0):.1f}/10
        </div>
        <div class="abstract">
            <h3>摘要</h3>
            <p>{paper.get('abstract', 'N/A')}</p>
        </div>
        {analysis_html}
    </div>
</body>
</html>'''
        (paper_dir / f"{paper_id}.html").write_text(html, encoding='utf-8')
