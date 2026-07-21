"""文档生成器 - PDF/Word - V2.0 美化版"""

from pathlib import Path
from typing import List, Dict
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.colors import HexColor, black, white
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


class PDFGenerator:
    """PDF文档生成器 - 美化版"""
    
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
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        story = []
        
        # 样式定义
        title_style = ParagraphStyle('Title2', fontName='Chinese', fontSize=22,
                                     textColor=HexColor('#1a1a2e'), spaceAfter=15, leading=28,
                                     alignment=1)  # 居中
        subtitle_style = ParagraphStyle('Subtitle', fontName='Chinese', fontSize=12,
                                        textColor=HexColor('#666666'), spaceAfter=20, leading=16,
                                        alignment=1)
        h1_style = ParagraphStyle('H1', fontName='Chinese', fontSize=16,
                                  textColor=HexColor('#1a1a2e'), spaceAfter=10, spaceBefore=20, leading=22,
                                  borderWidth=0, borderPadding=0)
        h2_style = ParagraphStyle('H2', fontName='Chinese', fontSize=13,
                                  textColor=HexColor('#16213e'), spaceAfter=8, spaceBefore=14, leading=18)
        body_style = ParagraphStyle('Body', fontName='Chinese', fontSize=9.5, leading=14, spaceAfter=4)
        small_style = ParagraphStyle('Small', fontName='Chinese', fontSize=8.5, leading=12, textColor=HexColor('#555555'))
        score_style = ParagraphStyle('Score', fontName='Chinese', fontSize=9, leading=12, textColor=HexColor('#667eea'))
        
        # ===== 封面 =====
        story.append(Spacer(1, 80))
        story.append(Paragraph("📚 AI论文研究报告", title_style))
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"Auto Paper Pipeline V2.0", subtitle_style))
        story.append(Spacer(1, 30))
        
        # 统计信息表格
        topics = {}
        for p in papers:
            topic = p.get('topic', 'Unknown')
            topics.setdefault(topic, []).append(p)
        
        stats_data = [['领域', '论文数', '高分论文(≥7分)']]
        for topic, tp in topics.items():
            high_score = sum(1 for p in tp if p.get('llm_score', 0) >= 7)
            stats_data.append([topic, str(len(tp)), str(high_score)])
        stats_data.append(['总计', str(len(papers)), str(sum(1 for p in papers if p.get('llm_score', 0) >= 7))])
        
        stats_table = Table(stats_data, colWidths=[5*cm, 3*cm, 3*cm])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Chinese'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#dddddd')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f8f9fa')]),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(stats_table)
        
        story.append(Spacer(1, 30))
        story.append(Paragraph(f"生成日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}", subtitle_style))
        
        story.append(PageBreak())
        
        # ===== 目录 =====
        story.append(Paragraph("目录", h1_style))
        story.append(Spacer(1, 10))
        
        for i, topic in enumerate(topics.keys(), 1):
            count = len(topics[topic])
            story.append(Paragraph(f"{i}. {topic} ({count}篇)", body_style))
        
        story.append(PageBreak())
        
        # ===== 按领域分组 =====
        for topic_idx, (topic, topic_papers) in enumerate(topics.items(), 1):
            story.append(Paragraph(f"{topic_idx}. {topic} ({len(topic_papers)}篇)", h1_style))
            story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#667eea')))
            story.append(Spacer(1, 10))
            
            for i, p in enumerate(topic_papers, 1):
                title_text = p.get('title', 'N/A')
                score = p.get('llm_score', 0)
                
                story.append(Paragraph(f"<b>{i}. {title_text}</b>", h2_style))
                story.append(Paragraph(f"ArXiv: {p.get('id', 'N/A')} | 日期: {p.get('published_date', 'N/A')} | 评分: {score:.1f}/10", score_style))
                
                abstract = p.get('abstract', '')[:300]
                story.append(Paragraph(f"摘要: {abstract}...", small_style))
                
                # 分析结果
                analysis = p.get('analysis', {})
                if analysis:
                    if analysis.get('strengths'):
                        story.append(Paragraph(f"<b>优点:</b> {', '.join(analysis['strengths'])}", small_style))
                    if analysis.get('improvements'):
                        story.append(Paragraph(f"<b>改进建议:</b> {', '.join(analysis['improvements'])}", small_style))
                
                story.append(Spacer(1, 8))
                story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor('#eeeeee')))
                story.append(Spacer(1, 8))
        
        # ===== 页脚 =====
        story.append(Spacer(1, 30))
        story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#cccccc')))
        story.append(Spacer(1, 10))
        story.append(Paragraph("Generated by Auto Paper Pipeline V2.0", subtitle_style))
        
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
        md_path = self.output_dir / f"report_{date}.md"
        self._generate_markdown(papers, date, md_path)
        
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
                
                analysis = p.get('analysis', {})
                if analysis:
                    if analysis.get('strengths'):
                        report += f"- **优点**: {', '.join(analysis['strengths'])}\n"
                    if analysis.get('improvements'):
                        report += f"- **改进建议**: {', '.join(analysis['improvements'])}\n"
                
                report += "\n"
        
        report += f"\n---\n*Generated by Auto Paper Pipeline V2.0 on {date}*\n"
        
        output_path.write_text(report, encoding='utf-8')
