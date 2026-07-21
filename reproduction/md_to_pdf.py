#!/usr/bin/env python3
"""将Markdown报告转换为PDF"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from pathlib import Path
import re

def parse_markdown(md_content):
    """解析Markdown为结构化内容"""
    elements = []
    lines = md_content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # 标题
        if line.startswith('# '):
            elements.append(('title', line[2:]))
        elif line.startswith('## '):
            elements.append(('h2', line[3:]))
        elif line.startswith('### '):
            elements.append(('h3', line[4:]))
        # 表格
        elif line.startswith('|') and i+1 < len(lines) and '---' in lines[i+1]:
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i].strip())
                i += 1
            elements.append(('table', table_lines))
            continue
        # 分隔线
        elif line == '---':
            elements.append(('hr', ''))
        # 引用
        elif line.startswith('> '):
            elements.append(('quote', line[2:]))
        # 列表
        elif line.startswith('- '):
            elements.append(('bullet', line[2:]))
        # 普通文本
        elif line:
            elements.append(('text', line))
        
        i += 1
    
    return elements

def create_pdf(md_file, pdf_file):
    """创建PDF"""
    # 读取Markdown
    md_content = Path(md_file).read_text('utf-8')
    elements = parse_markdown(md_content)
    
    # 创建PDF
    doc = SimpleDocTemplate(
        str(pdf_file),
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )
    
    styles = getSampleStyleSheet()
    
    # 自定义样式
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=HexColor('#1a1a2e'),
        spaceAfter=12,
        spaceBefore=20
    )
    
    h2_style = ParagraphStyle(
        'CustomH2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=HexColor('#16213e'),
        spaceAfter=8,
        spaceBefore=16
    )
    
    h3_style = ParagraphStyle(
        'CustomH3',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=HexColor('#0f3460'),
        spaceAfter=6,
        spaceBefore=12
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        spaceAfter=4
    )
    
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        leftIndent=20,
        spaceAfter=2
    )
    
    quote_style = ParagraphStyle(
        'CustomQuote',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        leftIndent=20,
        textColor=HexColor('#555555'),
        spaceAfter=4
    )
    
    # 构建PDF内容
    story = []
    
    for elem_type, content in elements:
        if elem_type == 'title':
            story.append(Paragraph(content, title_style))
            story.append(Spacer(1, 12))
        elif elem_type == 'h2':
            story.append(Spacer(1, 8))
            story.append(Paragraph(content, h2_style))
        elif elem_type == 'h3':
            story.append(Paragraph(content, h3_style))
        elif elem_type == 'text':
            # 清理Markdown格式
            text = content
            text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # 移除加粗
            text = re.sub(r'\*(.+?)\*', r'\1', text)  # 移除斜体
            text = re.sub(r'`(.+?)`', r'\1', text)  # 移除代码
            text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)  # 移除链接
            story.append(Paragraph(text, body_style))
        elif elem_type == 'bullet':
            text = content
            text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
            story.append(Paragraph(f"• {text}", bullet_style))
        elif elem_type == 'quote':
            story.append(Paragraph(f"<i>{content}</i>", quote_style))
        elif elem_type == 'hr':
            story.append(Spacer(1, 8))
            # 添加一条线
            from reportlab.platypus import HRFlowable
            story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#cccccc')))
            story.append(Spacer(1, 8))
        elif elem_type == 'table':
            # 解析表格
            table_data = []
            for row in content:
                if '---' in row:
                    continue
                cells = [c.strip() for c in row.split('|')[1:-1]]
                # 清理单元格文本
                cleaned = []
                for cell in cells:
                    cell = re.sub(r'\*\*(.+?)\*\*', r'\1', cell)
                    cell = re.sub(r'`(.+?)`', r'\1', cell)
                    cleaned.append(cell)
                table_data.append(cleaned)
            
            if table_data:
                # 创建表格
                col_count = max(len(row) for row in table_data)
                for row in table_data:
                    while len(row) < col_count:
                        row.append('')
                
                # 转换为Paragraph以支持换行
                table_paras = []
                for row in table_data:
                    table_paras.append([Paragraph(cell, body_style) for cell in row])
                
                table = Table(table_paras, repeatRows=1)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#f8f9fa')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#1a1a2e')),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('BACKGROUND', (0, 1), (-1, -1), HexColor('#ffffff')),
                    ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#dddddd')),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f8f9fa')]),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 6),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
                story.append(Spacer(1, 8))
                story.append(table)
                story.append(Spacer(1, 8))
    
    # 生成PDF
    doc.build(story)
    print(f"PDF已生成: {pdf_file}")

if __name__ == "__main__":
    import sys
    md_file = sys.argv[1] if len(sys.argv) > 1 else '/root/git/mimo/paper-pipeline/reproduction/COMPLETE_REPORT.md'
    pdf_file = sys.argv[2] if len(sys.argv) > 2 else '/root/git/mimo/paper-pipeline/reproduction/COMPLETE_REPORT.pdf'
    create_pdf(md_file, pdf_file)
