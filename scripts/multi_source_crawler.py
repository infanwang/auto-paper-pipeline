#!/usr/bin/env python3
"""Multi-source paper crawler - ACL, NeurIPS, ICML, arXiv."""

import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Import anti-crawl
from anti_crawl import AntiCrawl

anti_crawl = AntiCrawl(min_delay=2, max_delay=5, max_retries=3)


class PaperSource:
    """Base class for paper sources."""
    
    def __init__(self, name: str, base_url: str):
        self.name = name
        self.base_url = base_url
    
    def search(self, query: str, max_results: int = 20) -> List[Dict]:
        raise NotImplementedError
    
    def get_paper(self, paper_id: str) -> Optional[Dict]:
        raise NotImplementedError


class ACLSource(PaperSource):
    """ACL Anthology paper source."""
    
    def __init__(self):
        super().__init__("ACL Anthology", "https://api.aclanthology.org")
    
    def search(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search ACL Anthology."""
        url = f"{self.base_url}/search/?q={query}&format=json&limit={max_results}"
        
        try:
            data = anti_crawl.fetch(url, timeout=30)
            result = json.loads(data.decode("utf-8"))
            
            papers = []
            for item in result.get("hits", {}).get("hits", []):
                source = item.get("_source", {})
                papers.append({
                    "id": source.get("id", ""),
                    "title": source.get("title", ""),
                    "authors": source.get("authors", []),
                    "abstract": source.get("abstract", ""),
                    "year": source.get("year", ""),
                    "venue": source.get("venue", ""),
                    "url": source.get("url", ""),
                    "source": "acl",
                })
            
            return papers[:max_results]
        except Exception as e:
            print(f"  [!] ACL search error: {e}")
            return []


class NeurIPSSource(PaperSource):
    """NeurIPS paper source via OpenReview."""
    
    def __init__(self):
        super().__init__("NeurIPS", "https://api.openreview.net")
    
    def search(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search NeurIPS via OpenReview."""
        url = f"{self.base_url}/notes/search?query={query}&limit={max_results}&domain=NeurIPS.cc"
        
        try:
            data = anti_crawl.fetch(url, timeout=30)
            result = json.loads(data.decode("utf-8"))
            
            papers = []
            for item in result.get("notes", []):
                content = item.get("content", {})
                papers.append({
                    "id": item.get("id", ""),
                    "title": content.get("title", {}).get("value", ""),
                    "authors": [a.get("content", {}).get("name", {}).get("value", "") 
                               for a in item.get("authors", [])],
                    "abstract": content.get("abstract", {}).get("value", ""),
                    "year": item.get("cdate", "")[:4],
                    "venue": "NeurIPS",
                    "url": f"https://openreview.net/forum?id={item.get('id', '')}",
                    "source": "neurips",
                })
            
            return papers[:max_results]
        except Exception as e:
            print(f"  [!] NeurIPS search error: {e}")
            return []


class ICMLSource(PaperSource):
    """ICML paper source via OpenReview."""
    
    def __init__(self):
        super().__init__("ICML", "https://api.openreview.net")
    
    def search(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search ICML via OpenReview."""
        url = f"{self.base_url}/notes/search?query={query}&limit={max_results}&domain=ICML.cc"
        
        try:
            data = anti_crawl.fetch(url, timeout=30)
            result = json.loads(data.decode("utf-8"))
            
            papers = []
            for item in result.get("notes", []):
                content = item.get("content", {})
                papers.append({
                    "id": item.get("id", ""),
                    "title": content.get("title", {}).get("value", ""),
                    "authors": [a.get("content", {}).get("name", {}).get("value", "") 
                               for a in item.get("authors", [])],
                    "abstract": content.get("abstract", {}).get("value", ""),
                    "year": item.get("cdate", "")[:4],
                    "venue": "ICML",
                    "url": f"https://openreview.net/forum?id={item.get('id', '')}",
                    "source": "icml",
                })
            
            return papers[:max_results]
        except Exception as e:
            print(f"  [!] ICML search error: {e}")
            return []


class SemanticScholarSource(PaperSource):
    """Semantic Scholar paper source."""
    
    def __init__(self):
        super().__init__("Semantic Scholar", "https://api.semanticscholar.org")
    
    def search(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search Semantic Scholar."""
        url = f"{self.base_url}/graph/v1/paper/search?query={query}&limit={max_results}&fields=title,authors,abstract,year,venue,citationCount,externalIds"
        
        try:
            data = anti_crawl.fetch(url, timeout=30)
            result = json.loads(data.decode("utf-8"))
            
            papers = []
            for item in result.get("data", []):
                arxiv_id = item.get("externalIds", {}).get("ArXiv", "")
                papers.append({
                    "id": arxiv_id or item.get("paperId", ""),
                    "title": item.get("title", ""),
                    "authors": [a.get("name", "") for a in item.get("authors", [])],
                    "abstract": item.get("abstract", ""),
                    "year": item.get("year", ""),
                    "venue": item.get("venue", ""),
                    "citation_count": item.get("citationCount", 0),
                    "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
                    "source": "semantic_scholar",
                })
            
            return papers[:max_results]
        except Exception as e:
            print(f"  [!] Semantic Scholar search error: {e}")
            return []


class MultiSourceCrawler:
    """Multi-source paper crawler."""
    
    def __init__(self):
        self.sources = {
            "arxiv": None,  # Use existing arxiv.py
            "acl": ACLSource(),
            "neurips": NeurIPSSource(),
            "icml": ICMLSource(),
            "semantic_scholar": SemanticScholarSource(),
        }
    
    def search_all(self, query: str, sources: List[str] = None, max_results: int = 20) -> List[Dict]:
        """
        Search all sources.
        
        Args:
            query: Search query
            sources: List of sources to search (default: all)
            max_results: Max results per source
            
        Returns:
            List of papers from all sources
        """
        if sources is None:
            sources = list(self.sources.keys())
        
        all_papers = []
        seen_ids = set()
        
        for source_name in sources:
            source = self.sources.get(source_name)
            if source is None:
                continue
            
            print(f"  Searching {source.name}...")
            papers = source.search(query, max_results)
            
            for p in papers:
                pid = p.get("id", "")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    all_papers.append(p)
            
            print(f"    Found {len(papers)} papers")
        
        return all_papers
    
    def search_by_venue(self, venue: str, query: str, max_results: int = 50) -> List[Dict]:
        """
        Search by venue.
        
        Args:
            venue: Venue name (acl, neurips, icml)
            query: Search query
            max_results: Max results
            
        Returns:
            List of papers
        """
        return self.search_all(query, sources=[venue], max_results=max_results)
    
    def get_paper_details(self, paper_id: str, source: str = "semantic_scholar") -> Optional[Dict]:
        """
        Get paper details from a specific source.
        
        Args:
            paper_id: Paper ID
            source: Source name
            
        Returns:
            Paper details or None
        """
        source_obj = self.sources.get(source)
        if source_obj:
            return source_obj.get_paper(paper_id)
        return None


# Global instance
multi_crawler = MultiSourceCrawler()


if __name__ == "__main__":
    # Test multi-source search
    print("Testing multi-source search...")
    papers = multi_crawler.search_all("LLM agent", max_results=5)
    print(f"\nTotal papers found: {len(papers)}")
    for p in papers[:5]:
        print(f"  - [{p.get('source')}] {p.get('title', 'N/A')[:60]}...")
