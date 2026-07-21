"""文档生成器 - PDF/Word - V2.0"""

from pathlib import Path
from typing import List, Dict
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


class PDFGenerator:
    """PDF文档生成器"""
    
    FONT_PATH = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
    
    def __init__(self):
        if HAS_REPORTLAB:
            try:
                pdfmetrics.registerFont(TTFont('Chinese', self.FONT_PATH))
            except:
                pass
    
    def generate(self, papers: List[Dict], output_path: str, title: str = "AI论文研究报告"):
        """生成PDF报告"""
        if not HAS_REPORTLAB:
            print("reportlab未安装，跳过PDF生成")
            return
        
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=1.5*cm,
            leftMargin=1.5*cm,
            topMargin=1.5*cm,
            bottomMargin=1.5*cm
        )
        
        styles = getSampleStyleSheet()
        
        # 自定义样式
        title_style = ParagraphStyle('Title2', fontName='Chinese', fontSize=18,
                                     textColor=HexColor('#1a1a2e'), spaceAfter=12, leading=24)
        h2_style = ParagraphStyle('H2', fontName='Chinese', fontSize=14,
                                  textColor=HexColor('#16213e'), spaceAfter=8, spaceBefore=14, leading=18)
        body_style = ParagraphStyle('Body', fontName='Chinese', fontSize=9, leading=13, spaceAfter=4)
        small_style = ParagraphStyle('Small', fontName='Chinese', fontSize=8, leading=11, textColor=HexColor('#666666'))
        
        story = []
        
        # 封面
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"生成日期: {datetime.now().strftime('%Y-%m-%d')}", body_style))
        story.append(Paragraph(f"论文数量: {len(papers)}", body_style))
        story.append(Spacer(1, 20))
        story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#cccccc')))
        story.append(Spacer(1, 20))
        
        # 摘要
        topics = {}
        for p in papers:
            topic = p.get('topic', 'Unknown')
            topics.setdefault(topic, []).append(p)
        
        story.append(Paragraph("执行摘要", h2_style))
        story.append(Paragraph(f"本报告包含{len(papers)}篇AI领域最新论文，覆盖{len(topics)}个研究方向。", body_style))
        story.append(Spacer(1, 10))
        
        # 按领域分组
        for topic, topic_papers in topics.items():
            story.append(Paragraph(f"{topic} ({len(topic_papers)}篇)", h2_style))
            
            for i, p in enumerate(topic_papers[:5], 1):
                title = p.get('title', 'N/A')[:60]
                score = p.get('llm_score', 0)
                story.append(Paragraph(f"<b>{i}. {title}</b>", body_style))
                story.append(Paragraph(f"ArXiv: {p.get('id', 'N/A')} | 评分: {score:.1f}/10", small_style))
                
                abstract = p.get('abstract', '')[:200]
                story.append(Paragraph(f"{abstract}...", small_style))
                story.append(Spacer(1, 6))
        
        doc.build(story)
        print(f"  PDF已生成: {output_path}")


class ReportGenerator:
    """综合报告生成器"""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_gen = PDFGenerator()
    
    def generate_full_report(self, papers: List[Dict], date: str):
        """生成完整报告（PDF + Markdown）"""
        # Markdown报告
        md_path = self.output_dir / f"report_{date}.md"
        self._generate_markdown(papers, date, md_path)
        
        # PDF报告
        pdf_path = self.output_dir / f"report_{date}.pdf"
        self.pdf_gen.generate(papers, str(pdf_path), f"AI论文研究报告 {date}")
        
        print(f"  报告已生成:")
        print(f"    Markdown: {md_path}")
        print(f"    PDF: {pdf_path}")
    
    def _generate_markdown(self, papers: List[Dict], date: str, output_path: Path):
        """生成Markdown报告"""
        topics = {}
        for p in papers:
            topic = p.get('topic', 'Unknown')
            topics.setdefault(topic, []).append(p)
        
        report = f"""# AI论文研究报告

> {date} | {len(papers)}篇论文 | {len(topics)}个领域

---

## 执行摘要

本报告汇总了{date}抓取的{len(papers)}篇AI领域最新论文，按研究方向分类整理。

---

"""
        
        for topic, topic_papers in topics.items():
            report += f"## {topic} ({len(topic_papers)}篇)\n\n"
            
            for i, p in enumerate(topic_papers, 1):
                report += f"### {i}. {p.get('title', 'N/A')}\n"
                report += f"- **ArXiv**: {p.get('id', 'N/A')}\n"
                report += f"- **日期**: {p.get('published_date', 'N/A')}\n"
                report += f"- **评分**: {p.get('llm_score', 0):.1f}/10\n"
                report += f"- **摘要**: {p.get('abstract', 'N/A')[:200]}...\n"
                
                # 分析结果
                analysis = p.get('analysis', {})
                if analysis:
                    if analysis.get('strengths'):
                        report += f"- **优点**: {', '.join(analysis['strengths'])}\n"
                    if analysis.get('improvements'):
                        report += f"- **改进建议**: {', '.join(analysis['improvements'])}\n"
                
                report += "\n"
        
        report += f"\n---\n*Generated by Auto Paper Pipeline V2.0 on {date}*\n"
        
        output_path.write_text(report, encoding='utf-8')
