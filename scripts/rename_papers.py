#!/usr/bin/env python3
"""Rename downloaded papers to format: 作者_年份_标题.pdf"""

import json
import os
import re
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("/root/git/mimo/paper-pipeline/data")
PDF_DIR = Path("/root/git/mimo/paper-pipeline/pdfs")


def sanitize_filename(name: str, max_length: int = 100) -> str:
    """Sanitize filename by removing special characters."""
    # Remove or replace special characters
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    # Truncate if too long
    if len(name) > max_length:
        name = name[:max_length]
    return name


def get_paper_metadata():
    """Load all paper metadata from all JSON files."""
    paper_map = {}
    
    # Read all JSON files in data directory
    json_files = list(DATA_DIR.glob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for paper in data.get('papers', []):
                # Get paper ID (could be 'id' or 'arxiv_id')
                paper_id = paper.get('id', '') or paper.get('arxiv_id', '')
                if not paper_id:
                    continue
                
                # Extract year from published_date or published
                pub_date = paper.get('published_date', '') or paper.get('published', '')
                year = pub_date[:4] if pub_date else 'unknown'
                
                # Get first author
                authors = paper.get('authors', [])
                if authors:
                    if isinstance(authors[0], dict):
                        first_author = authors[0].get('name', 'Unknown')
                    else:
                        first_author = str(authors[0])
                else:
                    first_author = 'Unknown'
                
                # Clean up author name (take last name)
                author_parts = first_author.split()
                if len(author_parts) > 1:
                    last_name = author_parts[-1]
                else:
                    last_name = first_author
                
                # Get title
                title = paper.get('title', 'Untitled')
                
                # Only store if not already exists (prefer pipeline data)
                if paper_id not in paper_map:
                    paper_map[paper_id] = {
                        'id': paper_id,
                        'author': last_name,
                        'year': year,
                        'title': title,
                        'first_author_full': first_author
                    }
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    
    return paper_map


def rename_papers(dry_run: bool = True):
    """Rename all PDFs according to the new format."""
    paper_map = get_paper_metadata()
    
    if not paper_map:
        print("No paper metadata found!")
        return
    
    print(f"Loaded metadata for {len(paper_map)} papers\n")
    
    renamed_count = 0
    skipped_count = 0
    error_count = 0
    
    # Process all PDF directories
    for pdf_file in PDF_DIR.rglob("*.pdf"):
        filename = pdf_file.stem
        
        # Extract arxiv_id from filename
        # Handle formats: "2607.16133" or "2607.16133_When_Do_Multi..."
        if '_' in filename:
            arxiv_id = filename.split('_')[0]
        else:
            arxiv_id = filename
        
        if arxiv_id not in paper_map:
            skipped_count += 1
            continue
        
        paper = paper_map[arxiv_id]
        
        # Build new filename: 作者_年份_标题.pdf
        author = sanitize_filename(paper['author'])
        year = paper['year']
        title = sanitize_filename(paper['title'], max_length=80)
        
        new_filename = f"{author}_{year}_{title}.pdf"
        new_path = pdf_file.parent / new_filename
        
        if dry_run:
            print(f"  [DRY] {pdf_file.name}")
            print(f"    → {new_filename}")
        else:
            try:
                if pdf_file != new_path:
                    # Avoid overwriting existing files
                    if new_path.exists():
                        print(f"  [SKIP] Already exists: {new_filename}")
                        skipped_count += 1
                        continue
                    
                    pdf_file.rename(new_path)
                    print(f"  [OK] {pdf_file.name}")
                    print(f"    → {new_filename}")
                    renamed_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                print(f"  [ERROR] {pdf_file.name}: {e}")
                error_count += 1
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Renamed: {renamed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors: {error_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rename papers to 作者_年份_标题.pdf format")
    parser.add_argument("--execute", action="store_true", help="Actually rename files (default: dry run)")
    args = parser.parse_args()
    
    dry_run = not args.execute
    
    if dry_run:
        print("="*60)
        print("DRY RUN - No files will be renamed")
        print("Use --execute to actually rename files")
        print("="*60 + "\n")
    
    rename_papers(dry_run=dry_run)
