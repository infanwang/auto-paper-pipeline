#!/usr/bin/env python3
"""
论文列表导出Excel + PDF下载
"""

import json
import csv
from pathlib import Path
from datetime import datetime


def export_to_csv(papers, output_path):
    """导出为CSV（Excel兼容）"""
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        
        # 表头
        writer.writerow([
            '序号', 'ArXiv ID', '标题', '作者', '领域', 
            '评分', '发布日期', '标签', '摘要'
        ])
        
        # 数据
        for i, p in enumerate(papers, 1):
            authors = ', '.join([a.get('name', '') for a in p.get('authors', [])[:3]])
            tags = ', '.join(p.get('llm_tags', []))
            abstract = p.get('abstract', '')[:100]
            
            writer.writerow([
                i,
                p.get('id', ''),
                p.get('title', ''),
                authors,
                p.get('topic', ''),
                p.get('llm_score', 0),
                p.get('published_date', ''),
                tags,
                abstract
            ])
    
    return output_path


def download_pdfs(papers, output_dir):
    """下载PDF文件"""
    import urllib.request
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = []
    for i, p in enumerate(papers[:10], 1):  # 限制下载前10篇
        paper_id = p.get('id', '')
        if not paper_id:
            continue
        
        pdf_url = f"https://arxiv.org/pdf/{paper_id}"
        pdf_path = output_dir / f"{paper_id}.pdf"
        
        if pdf_path.exists():
            downloaded.append(str(pdf_path))
            continue
        
        try:
            print(f"  下载 {i}/10: {p.get('title', '')[:40]}...")
            req = urllib.request.Request(pdf_url, headers={'User-Agent': 'Mozilla/5.0'})
            data = urllib.request.urlopen(req, timeout=30).read()
            pdf_path.write_bytes(data)
            downloaded.append(str(pdf_path))
        except Exception as e:
            print(f"  下载失败: {e}")
    
    return downloaded


def main():
    print("="*60)
    print("📊 论文导出工具")
    print("="*60)
    
    # 加载数据
    data = json.loads(Path('data/pipeline_2026-07-21.json').read_text())
    papers = data.get('papers', [])
    print(f"\n论文总数: {len(papers)}篇")
    
    # 1. 导出CSV
    print("\n=== 1. 导出CSV ===")
    csv_path = f"reports/papers_{datetime.now().strftime('%Y%m%d')}.csv"
    export_to_csv(papers, csv_path)
    print(f"✅ CSV已导出: {csv_path}")
    
    # 2. 下载PDF
    print("\n=== 2. 下载PDF (前10篇) ===")
    pdf_dir = f"pdfs/download_{datetime.now().strftime('%Y%m%d')}"
    downloaded = download_pdfs(papers, pdf_dir)
    print(f"✅ 已下载: {len(downloaded)}篇PDF")
    
    # 3. 统计
    print("\n=== 3. 统计 ===")
    print(f"CSV文件: {csv_path}")
    print(f"PDF目录: {pdf_dir}")
    print(f"论文总数: {len(papers)}")
    print(f"PDF下载: {len(downloaded)}")
    
    return csv_path, downloaded


if __name__ == "__main__":
    main()
