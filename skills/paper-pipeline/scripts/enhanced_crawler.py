#!/usr/bin/env python3
"""Enhanced paper crawler with anti-crawling and deduplication."""

import subprocess
import json
import re
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

# Import anti-crawl and dedup modules
from anti_crawl import AntiCrawl
from dedup import Deduplicator

SCRIPTS_DIR = Path("/root/git/mimo/paper-pipeline/scripts")
DATA_DIR = Path("/root/git/mimo/paper-pipeline/data")
PDF_DIR = Path("/root/git/mimo/paper-pipeline/pdfs")

# Initialize anti-crawl and dedup
anti_crawl = AntiCrawl(min_delay=3, max_delay=8, max_retries=3)
dedup = Deduplicator()

TOPICS = {
    "AI Agent": {
        "queries": ["LLM agent tool use", "multi-agent collaboration", "agentic reasoning planning"],
        "categories": ["cs.AI", "cs.CL", "cs.MA"],
    },
    "LLM推理优化": {
        "queries": ["LLM inference optimization", "KV cache efficient inference", "model quantization inference"],
        "categories": ["cs.CL", "cs.LG"],
    },
    "多模态大模型": {
        "queries": ["multimodal large language model", "vision language model", "cross-modal alignment"],
        "categories": ["cs.CV", "cs.CL"],
    },
    "代码生成": {
        "queries": ["code generation LLM", "AI programming assistant", "automated program repair"],
        "categories": ["cs.SE", "cs.CL"],
    },
    "芯片验证": {
        "queries": ["chip verification formal", "hardware verification LLM", "RTL verification automation", "digital circuit testing"],
        "categories": ["cs.AR", "cs.CV", "eess.SP"],
    },
    "5G移动通信": {
        "queries": ["5G NR resource allocation", "massive MIMO beamforming", "wireless communication deep learning", "6G channel estimation"],
        "categories": ["eess.SP", "cs.IT", "cs.NI"],
    },
}


def sanitize_title(title: str, max_length: int = 60) -> str:
    """Sanitize title for filename."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in title[:max_length])
    return re.sub(r'_+', '_', safe).strip('_')


def download_pdf(arxiv_id: str, topic: str, title: str = "") -> Optional[Path]:
    """
    Download paper PDF with anti-crawling and deduplication.
    
    Args:
        arxiv_id: arXiv paper ID
        topic: Paper topic/category
        title: Paper title
        
    Returns:
        Path to downloaded PDF or None
    """
    # Check dedup first
    if dedup.is_duplicate(arxiv_id, title):
        existing = dedup.get_paper(arxiv_id)
        if existing and existing.get("pdf_path"):
            print(f"    [SKIP] Already downloaded: {arxiv_id}")
            return Path(existing["pdf_path"])
        # ID exists but no path - re-download
    
    # Setup directories
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    topic_dir = PDF_DIR / topic.replace(" ", "_")
    topic_dir.mkdir(exist_ok=True)
    
    # Build filename
    safe_title = sanitize_title(title) if title else "Untitled"
    pdf_path = topic_dir / f"{arxiv_id.replace('/', '_')}_{safe_title}.pdf"
    
    # Check if file already exists
    if pdf_path.exists() and pdf_path.stat().st_size > 1000:
        # File exists, register in dedup
        dedup.register(arxiv_id, title, str(pdf_path))
        return pdf_path
    
    # Download with anti-crawl
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    referer = f"https://arxiv.org/abs/{arxiv_id}"
    
    print(f"    Downloading: {arxiv_id}")
    success = anti_crawl.download_pdf(url, str(pdf_path), timeout=120, referer=referer)
    
    if success:
        # Verify file
        if pdf_path.exists() and pdf_path.stat().st_size > 1000:
            dedup.register(arxiv_id, title, str(pdf_path))
            print(f"    PDF: {pdf_path.name} ({pdf_path.stat().st_size / 1024:.1f} KB)")
            return pdf_path
        else:
            print(f"    [!] Invalid PDF file")
            pdf_path.unlink(missing_ok=True)
            return None
    else:
        print(f"    [!] PDF download failed")
        return None


def search_arxiv_advanced(query: str, category: str = None, max_results: int = 20) -> list:
    """
    Search arXiv using arxiv.py script with anti-crawl measures.
    
    Args:
        query: Search query
        category: Optional category filter
        max_results: Maximum results
        
    Returns:
        List of paper dicts
    """
    cmd = [
        "python3", str(SCRIPTS_DIR / "arxiv.py"),
        "search", query,
        "--max", str(max_results),
        "--sort", "date",
        "--json"
    ]
    if category:
        cmd.extend(["--category", category])
    
    try:
        # Use anti-crawl for subprocess
        anti_crawl.wait()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if isinstance(data, dict) and "results" in data:
                return data["results"]
            elif isinstance(data, list):
                return data
        else:
            print(f"  [!] arxiv.py returned {result.returncode}")
    except subprocess.TimeoutExpired:
        print(f"  [!] arxiv.py timeout")
    except Exception as e:
        print(f"  [!] arxiv.py error: {e}")
    
    return []


def get_semanticscholar_data(arxiv_id: str) -> Optional[dict]:
    """
    Fetch citation data from Semantic Scholar with anti-crawl.
    
    Args:
        arxiv_id: arXiv paper ID
        
    Returns:
        Paper data dict or None
    """
    url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}?fields=title,citationCount,influentialCitationCount,tldr,year,fieldsOfStudy"
    
    try:
        data = anti_crawl.fetch(url, timeout=15)
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


def get_similar_papers(arxiv_id: str) -> list:
    """Get related papers using arxiv.py similar command."""
    cmd = ["python3", str(SCRIPTS_DIR / "arxiv.py"), "similar", arxiv_id, "--max", "5", "--json"]
    try:
        anti_crawl.wait()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception:
        pass
    return []


def get_paper_refs(arxiv_id: str) -> list:
    """Get references using arxiv.py refs command."""
    cmd = ["python3", str(SCRIPTS_DIR / "arxiv.py"), "refs", arxiv_id, "--max", "10", "--json"]
    try:
        anti_crawl.wait()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception:
        pass
    return []


def crawl_topic(topic_name: str, config: dict) -> list:
    """
    Crawl papers for a single topic with dedup.
    
    Args:
        topic_name: Topic name
        config: Topic configuration
        
    Returns:
        List of paper dicts
    """
    print(f"\n{'='*50}")
    print(f"Topic: {topic_name}")
    print(f"{'='*50}")
    
    all_papers = []
    seen_ids = set()
    
    for query in config["queries"]:
        for cat in config.get("categories", [None]):
            print(f"  Search: {query} (cat: {cat})")
            papers = search_arxiv_advanced(query, category=cat, max_results=15)
            
            new_count = 0
            for p in papers:
                pid = p.get("id", "")
                if pid and pid not in seen_ids:
                    # Check if already in dedup
                    if dedup.is_duplicate(pid, p.get("title")):
                        continue
                    
                    seen_ids.add(pid)
                    p["topic"] = topic_name
                    p["query"] = query
                    all_papers.append(p)
                    new_count += 1
            
            print(f"    Found {len(papers)} papers, {new_count} new")
    
    # Enrich top papers with Semantic Scholar data
    enrich_count = min(30, len(all_papers))
    print(f"  Enriching top {enrich_count} papers with citation data...")
    
    enriched = 0
    for p in all_papers[:enrich_count]:
        arxiv_id = p.get("id", "").split("v")[0]
        if arxiv_id:
            ss_data = get_semanticscholar_data(arxiv_id)
            if ss_data:
                p["citation_count"] = ss_data.get("citationCount", 0)
                p["influential_citations"] = ss_data.get("influentialCitationCount", 0)
                p["tldr"] = ss_data.get("tldr", {}).get("text", "") if ss_data.get("tldr") else ""
                p["fields_of_study"] = ss_data.get("fieldsOfStudy", [])
                enriched += 1
    
    print(f"    Enriched {enriched}/{enrich_count} papers")
    
    # Rank by citations + recency
    now = datetime.now().date()
    for p in all_papers:
        pub_date = p.get("published", "")[:10]
        if pub_date:
            try:
                days_old = (now - datetime.strptime(pub_date, "%Y-%m-%d").date()).days
                p["recency_score"] = max(0, 100 - days_old)
            except ValueError:
                p["recency_score"] = 50
        else:
            p["recency_score"] = 50
        
        citations = p.get("citation_count", 0) or 0
        p["citation_score"] = min(citations * 10, 100)
        p["total_score"] = p["recency_score"] * 0.4 + p["citation_score"] * 0.6
    
    all_papers.sort(key=lambda x: x.get("total_score", 0), reverse=True)
    
    print(f"  Found {len(all_papers)} unique papers")
    return all_papers


def crawl_all(download_top: int = 10) -> list:
    """
    Crawl all topics and save results.
    
    Args:
        download_top: Number of top papers to download PDFs
        
    Returns:
        List of all papers
    """
    print("\n" + "="*60)
    print("Enhanced Paper Crawler")
    print("="*60)
    
    # Print dedup stats
    dedup_stats = dedup.get_stats()
    print(f"\nDedup index: {dedup_stats['total_papers']} papers, {dedup_stats['total_hashes']} hashes")
    
    all_papers = []
    
    for topic_name, config in TOPICS.items():
        papers = crawl_topic(topic_name, config)
        all_papers.extend(papers)
    
    # Global dedup
    seen = set()
    unique = []
    for p in all_papers:
        pid = p.get("id", "")
        if pid and pid not in seen:
            seen.add(pid)
            unique.append(p)
    
    unique.sort(key=lambda x: x.get("total_score", 0), reverse=True)
    
    # Download PDFs
    print(f"\n=== Downloading Top {download_top} PDFs ===")
    downloaded = 0
    for p in unique[:download_top]:
        arxiv_id = p.get("id", "").split("v")[0]
        if arxiv_id:
            pdf_path = download_pdf(arxiv_id, p.get("topic", "misc"), p.get("title", ""))
            if pdf_path:
                p["pdf_path"] = str(pdf_path)
                downloaded += 1
    
    print(f"\nDownloaded {downloaded}/{download_top} PDFs")
    
    # Save results
    today = datetime.now().strftime("%Y-%m-%d")
    output_dir = DATA_DIR / today
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "papers_enhanced.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Total: {len(unique)} unique papers saved ===")
    print(f"Output: {output_file}")
    
    # Print top 10
    print("\n=== Top 10 by score ===")
    for i, p in enumerate(unique[:10], 1):
        title = p.get("title", "N/A")[:70]
        citations = p.get("citation_count", 0) or 0
        score = p.get("total_score", 0)
        pdf = "✓" if p.get("pdf_path") else "✗"
        print(f"{i}. [{score:.0f}] ({citations} cites) [PDF:{pdf}] {title}")
    
    # Print anti-crawl stats
    anti_stats = anti_crawl.get_stats()
    print(f"\nAnti-crawl: {anti_stats['request_count']} requests made")
    
    return unique


if __name__ == "__main__":
    crawl_all()
