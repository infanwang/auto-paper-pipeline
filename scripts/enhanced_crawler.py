#!/usr/bin/env python3
"""Enhanced paper crawler - uses arxiv.py + Semantic Scholar for richer data."""

import subprocess
import json
import urllib.request
import time
from pathlib import Path
from datetime import datetime

SCRIPTS_DIR = Path("/root/git/mimo/paper-pipeline/scripts")
DATA_DIR = Path("/root/git/mimo/paper-pipeline/data")

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

PDF_DIR = Path("/root/git/mimo/paper-pipeline/pdfs")


def download_pdf(arxiv_id, topic, title=""):
    """Download paper PDF from arXiv."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    topic_dir = PDF_DIR / topic.replace(" ", "_")
    topic_dir.mkdir(exist_ok=True)
    
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in title[:60]) if title else arxiv_id.replace("/", "_")
    pdf_path = topic_dir / f"{arxiv_id.replace('/', '_')}_{safe_name}.pdf"
    
    if pdf_path.exists():
        return pdf_path
    
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PaperPipeline/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            pdf_path.write_bytes(resp.read())
        print(f"    PDF: {pdf_path.name}")
        return pdf_path
    except Exception as e:
        print(f"    [!] PDF download failed: {e}")
        return None


def search_arxiv_advanced(query, category=None, max_results=20):
    """Search using arxiv.py script."""
    cmd = ["python3", str(SCRIPTS_DIR / "arxiv.py"), "search", query, "--max", str(max_results), "--sort", "date", "--json"]
    if category:
        cmd.extend(["--category", category])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            # arxiv.py returns {"total": N, "results": [...]}
            if isinstance(data, dict) and "results" in data:
                return data["results"]
            elif isinstance(data, list):
                return data
    except Exception as e:
        print(f"  [!] arxiv.py error: {e}")
    return []


def get_semanticscholar_data(arxiv_id):
    """Fetch citation data from Semantic Scholar."""
    url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}?fields=title,citationCount,influentialCitationCount,tldr,year,fieldsOfStudy"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PaperPipeline/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def get_similar_papers(arxiv_id):
    """Get related papers using arxiv.py similar command."""
    cmd = ["python3", str(SCRIPTS_DIR / "arxiv.py"), "similar", arxiv_id, "--max", "5", "--json"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception:
        pass
    return []


def get_paper_refs(arxiv_id):
    """Get references using arxiv.py refs command."""
    cmd = ["python3", str(SCRIPTS_DIR / "arxiv.py"), "refs", arxiv_id, "--max", "10", "--json"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception:
        pass
    return []


def crawl_topic(topic_name, config):
    """Crawl papers for a single topic."""
    print(f"\n{'='*50}")
    print(f"Topic: {topic_name}")
    print(f"{'='*50}")
    
    all_papers = []
    seen_ids = set()
    
    for query in config["queries"]:
        for cat in config.get("categories", [None]):
            print(f"  Search: {query} (cat: {cat})")
            papers = search_arxiv_advanced(query, category=cat, max_results=15)
            
            for p in papers:
                pid = p.get("id", "")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    p["topic"] = topic_name
                    p["query"] = query
                    all_papers.append(p)
            
            time.sleep(3)  # Rate limit
    
    # Enrich top 30 papers with Semantic Scholar data (rate limit)
    enrich_count = min(30, len(all_papers))
    print(f"  Enriching top {enrich_count} papers with citation data...")
    for p in all_papers[:enrich_count]:
        arxiv_id = p.get("id", "").split("v")[0]  # Remove version
        if arxiv_id:
            ss_data = get_semanticscholar_data(arxiv_id)
            if ss_data:
                p["citation_count"] = ss_data.get("citationCount", 0)
                p["influential_citations"] = ss_data.get("influentialCitationCount", 0)
                p["tldr"] = ss_data.get("tldr", {}).get("text", "") if ss_data.get("tldr") else ""
                p["fields_of_study"] = ss_data.get("fieldsOfStudy", [])
            time.sleep(1.2)  # Semantic Scholar rate limit
    
    # Rank by citations + recency
    now = datetime.now().date()
    for p in all_papers:
        pub_date = p.get("published", "")[:10]
        if pub_date:
            days_old = (now - datetime.strptime(pub_date, "%Y-%m-%d").date()).days
            p["recency_score"] = max(0, 100 - days_old)
        else:
            p["recency_score"] = 50
        
        citations = p.get("citation_count", 0) or 0
        p["citation_score"] = min(citations * 10, 100)
        p["total_score"] = p["recency_score"] * 0.4 + p["citation_score"] * 0.6
    
    all_papers.sort(key=lambda x: x.get("total_score", 0), reverse=True)
    
    print(f"  Found {len(all_papers)} unique papers")
    return all_papers


def crawl_all():
    """Crawl all topics and save results."""
    all_papers = []
    
    for topic_name, config in TOPICS.items():
        papers = crawl_topic(topic_name, config)
        all_papers.extend(papers)
    
    # Global dedup
    seen = set()
    unique = []
    for p in all_papers:
        pid = p.get("id", "")
        if pid not in seen:
            seen.add(pid)
            unique.append(p)
    
    unique.sort(key=lambda x: x.get("total_score", 0), reverse=True)
    
    # Download PDFs for top 10
    print("\n=== Downloading Top 10 PDFs ===")
    for p in unique[:10]:
        arxiv_id = p.get("id", "").split("v")[0]
        if arxiv_id:
            pdf_path = download_pdf(arxiv_id, p.get("topic", "misc"), p.get("title", ""))
            if pdf_path:
                p["pdf_path"] = str(pdf_path)
            time.sleep(2)  # Rate limit
    
    # Save
    today = datetime.now().strftime("%Y-%m-%d")
    output_dir = DATA_DIR / today
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "papers_enhanced.json", "w") as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Total: {len(unique)} unique papers saved ===")
    
    # Print top 10
    print("\n=== Top 10 by score ===")
    for i, p in enumerate(unique[:10], 1):
        title = p.get("title", "N/A")[:70]
        citations = p.get("citation_count", 0) or 0
        score = p.get("total_score", 0)
        pdf = "✓" if p.get("pdf_path") else "✗"
        print(f"{i}. [{score:.0f}] ({citations} cites) [PDF:{pdf}] {title}")
        print(f"   {p.get('url', '')}")
    
    return unique


if __name__ == "__main__":
    crawl_all()
