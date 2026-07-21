#!/usr/bin/env python3
"""ArXiv paper crawler - search, fetch, and rank papers."""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

ARXIV_API = "http://export.arxiv.org/api/query"
DATA_DIR = Path("/root/git/mimo/paper-pipeline/data")

def search_arxiv(query, category=None, max_results=50, sort_by="submittedDate", sort_order="descending"):
    """Search arXiv API and return papers."""
    params = {
        "search_query": f"all:{query}" + (f" AND cat:{category}" if category else ""),
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PaperPipeline/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read().decode("utf-8")
            break
        except Exception as e:
            if attempt == 2:
                print(f"  [!] Failed after 3 attempts: {e}")
                return []
            time.sleep(2 ** attempt)
    
    root = ET.fromstring(data)
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    
    papers = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        abstract = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
        published = entry.find("atom:published", ns).text[:10]
        updated = entry.find("atom:updated", ns).text[:10]
        
        arxiv_id = entry.find("atom:id", ns).text.split("/abs/")[-1]
        categories = [c.get("term") for c in entry.findall("atom:category", ns)]
        
        authors = []
        for author in entry.findall("atom:author", ns):
            name = author.find("atom:name", ns).text
            authors.append(name)
        
        pdf_link = ""
        for link in entry.findall("atom:link", ns):
            if link.get("title") == "pdf":
                pdf_link = link.get("href")
        
        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "published": published,
            "updated": updated,
            "categories": categories,
            "pdf_url": pdf_link,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
        })
    
    return papers


def filter_by_date(papers, since_date):
    """Filter papers published after since_date."""
    cutoff = datetime.strptime(since_date, "%Y-%m-%d").date()
    return [p for p in papers if datetime.strptime(p["published"], "%Y-%m-%d").date() >= cutoff]


def deduplicate(papers):
    """Remove duplicate papers by arxiv_id."""
    seen = set()
    unique = []
    for p in papers:
        if p["arxiv_id"] not in seen:
            seen.add(p["arxiv_id"])
            unique.append(p)
    return unique


def rank_papers(papers):
    """Rank papers by recency and citation potential."""
    for p in papers:
        days_old = (datetime.now().date() - datetime.strptime(p["published"], "%Y-%m-%d").date()).days
        p["recency_score"] = max(0, 100 - days_old)
        author_count = len(p["authors"])
        abstract_len = len(p["abstract"].split())
        p["potential_score"] = min(author_count * 5, 30) + min(abstract_len / 10, 20)
        p["total_score"] = p["recency_score"] * 0.6 + p["potential_score"] * 0.4
    
    return sorted(papers, key=lambda x: x["total_score"], reverse=True)


def crawl_all_topics(config_path="/root/git/mimo/paper-pipeline/config.yaml"):
    """Crawl papers for all configured topics."""
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    all_papers = []
    since = config["time_range"]
    
    for topic in config["topics"]:
        print(f"\n=== Searching: {topic['name']} ===")
        topic_papers = []
        
        for query in topic["queries"]:
            for cat in topic.get("arxiv_categories", [None]):
                print(f"  Query: {query} (cat: {cat})")
                results = search_arxiv(query, category=cat, max_results=30)
                topic_papers.extend(results)
                time.sleep(3)  # Rate limit
        
        topic_papers = deduplicate(topic_papers)
        topic_papers = filter_by_date(topic_papers, since)
        topic_papers = rank_papers(topic_papers)
        
        print(f"  Found {len(topic_papers)} papers since {since}")
        all_papers.extend([(topic["name"], p) for p in topic_papers[:20]])  # Top 20 per topic
    
    # Global dedup and rank
    seen_ids = set()
    unique_papers = []
    for topic_name, p in all_papers:
        if p["arxiv_id"] not in seen_ids:
            seen_ids.add(p["arxiv_id"])
            unique_papers.append((topic_name, p))
    
    unique_papers.sort(key=lambda x: x[1]["total_score"], reverse=True)
    
    # Save
    today = datetime.now().strftime("%Y-%m-%d")
    output_dir = DATA_DIR / today
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "papers.json", "w") as f:
        json.dump(unique_papers, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Total: {len(unique_papers)} unique papers saved ===")
    return unique_papers


if __name__ == "__main__":
    papers = crawl_all_topics()
    
    # Print top 10
    print("\n=== Top 10 Papers ===")
    for i, (topic, p) in enumerate(papers[:10], 1):
        print(f"{i}. [{topic}] {p['title'][:80]}...")
        print(f"   {p['published']} | {p['url']}")
