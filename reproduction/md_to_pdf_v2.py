#!/usr/bin/env python3
"""将Markdown报告转换为PDF - 支持中文"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pathlib import Path
import re

# 注册中文字体
FONT_PATH = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
pdfmetrics.registerFont(TTFont('Chinese', FONT_PATH))

def parse_markdown(md_content):
    """解析Markdown为结构化内容"""
    elements = []
    lines = md_content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('# ') and not line.startswith('## '):
            elements.append(('title', line[2:]))
        elif line.startswith('## '):
            elements.append(('h2', line[3:]))
        elif line.startswith('### '):
            elements.append(('h3', line[4:]))
        elif line.startswith('|') and i+1 < len(lines) and '---' in lines[i+1]:
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i].strip())
                i += 1
            elements.append(('table', table_lines))
            continue
        elif line == '---':
            elements.append(('hr', ''))
        elif line.startswith('> '):
            elements.append(('quote', line[2:]))
        elif line.startswith('- '):
            elements.append(('bullet', line[2:]))
        elif line.startswith('**') and line.endswith('**'):
            elements.append(('bold_line', line[2:-2]))
        elif line:
            elements.append(('text', line))
        
        i += 1
    
    return elements

def clean_md(text):
    """清理Markdown格式"""
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    text = re.sub(r'`(.+?)`', r'<font color="#e94560">\1</font>', text)
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    return text

def create_pdf(md_file, pdf_file):
    """创建PDF"""
    md_content = Path(md_file).read_text('utf-8')
    elements = parse_markdown(md_content)
    
    doc = SimpleDocTemplate(
        str(pdf_file),
        pagesize=A4,
        rightMargin=1.8*cm,
        leftMargin=1.8*cm,
        topMargin=1.5*cm,
        bottomMargin=1.5*cm
    )
    
    # 样式
    title_style = ParagraphStyle('Title', fontName='Chinese', fontSize=18, 
                                  textColor=HexColor('#1a1a2e'), spaceAfter=12, spaceBefore=8,
                                  leading=24)
    h2_style = ParagraphStyle('H2', fontName='Chinese', fontSize=14,
                               textColor=HexColor('#16213e'), spaceAfter=8, spaceBefore=14,
                               leading=18)
    h3_style = ParagraphStyle('H3', fontName='Chinese', fontSize=11,
                               textColor=HexColor('#0f3460'), spaceAfter=6, spaceBefore=10,
                               leading=15)
    body_style = ParagraphStyle('Body', fontName='Chinese', fontSize=9,
                                 leading=13, spaceAfter=3)
    bullet_style = ParagraphStyle('Bullet', fontName='Chinese', fontSize=9,
                                   leading=13, leftIndent=15, spaceAfter=2)
    quote_style = ParagraphStyle('Quote', fontName='Chinese', fontSize=9,
                                  leading=13, leftIndent=15, textColor=HexColor('#666666'),
                                  spaceAfter=3)
    bold_style = ParagraphStyle('Bold', fontName='Chinese', fontSize=9,
                                 leading=13, spaceAfter=2)
    
    story = []
    
    for elem_type, content in elements:
        if elem_type == 'title':
            story.append(Paragraph(clean_md(content), title_style))
            story.append(Spacer(1, 8))
        elif elem_type == 'h2':
            story.append(Spacer(1, 6))
            story.append(Paragraph(clean_md(content), h2_style))
        elif elem_type == 'h3':
            story.append(Paragraph(clean_md(content), h3_style))
        elif elem_type == 'text':
            story.append(Paragraph(clean_md(content), body_style))
        elif elem_type == 'bullet':
            story.append(Paragraph(f"• {clean_md(content)}", bullet_style))
        elif elem_type == 'quote':
            story.append(Paragraph(f"<i>{clean_md(content)}</i>", quote_style))
        elif elem_type == 'bold_line':
            story.append(Paragraph(f"<b>{content}</b>", bold_style))
        elif elem_type == 'hr':
            story.append(Spacer(1, 6))
            story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor('#cccccc')))
            story.append(Spacer(1, 6))
        elif elem_type == 'table':
            table_data = []
            for row in content:
                if '---' in row:
                    continue
                cells = [c.strip() for c in row.split('|')[1:-1]]
                cleaned = [clean_md(c) for c in cells]
                table_data.append(cleaned)
            
            if table_data:
                col_count = max(len(row) for row in table_data)
                for row in table_data:
                    while len(row) < col_count:
                        row.append('')
                
                table_paras = [[Paragraph(cell, body_style) for cell in row] for row in table_data]
                
                table = Table(table_paras, repeatRows=1)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#e8eaf6')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#1a1a2e')),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Chinese'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#dddddd')),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f5f5f5')]),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 6),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
                story.append(Spacer(1, 6))
                story.append(table)
                story.append(Spacer(1, 6))
    
    doc.build(story)
    print(f"PDF已生成: {pdf_file}")

if __name__ == "__main__":
    import sys
    md_file = sys.argv[1] if len(sys.argv) > 1 else '/root/git/mimo/paper-pipeline/reproduction/COMPLETE_REPORT.md'
    pdf_file = sys.argv[2] if len(sys.argv) > 2 else '/root/git/mimo/paper-pipeline/reproduction/COMPLETE_REPORT.pdf'
    create_pdf(md_file, pdf_file)
