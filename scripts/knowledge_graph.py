#!/usr/bin/env python3
"""Paper knowledge graph module."""

import json
from pathlib import Path
from typing import Dict, List, Set, Optional
from collections import defaultdict
from datetime import datetime


class PaperKnowledgeGraph:
    """Paper knowledge graph for relationships and trends."""
    
    def __init__(self, data_dir: str = "/root/git/mimo/paper-pipeline/data"):
        self.data_dir = Path(data_dir)
        self.graph = {
            "papers": {},
            "authors": {},
            "topics": {},
            "citations": defaultdict(list),
            "co_authors": defaultdict(set),
            "topic_papers": defaultdict(list),
            "temporal": defaultdict(list),
        }
    
    def add_paper(self, paper: dict):
        """
        Add paper to knowledge graph.
        
        Args:
            paper: Paper dictionary
        """
        paper_id = paper.get("id", "")
        if not paper_id:
            return
        
        # Add paper node
        self.graph["papers"][paper_id] = {
            "title": paper.get("title", ""),
            "authors": paper.get("authors", []),
            "abstract": paper.get("abstract", ""),
            "year": paper.get("year", paper.get("published_date", "")[:4]),
            "venue": paper.get("venue", ""),
            "topics": paper.get("topics", []),
            "citation_count": paper.get("citation_count", 0),
        }
        
        # Add author nodes and relationships
        authors = paper.get("authors", [])
        if isinstance(authors, list):
            for author in authors:
                if isinstance(author, dict):
                    author_name = author.get("name", "")
                else:
                    author_name = str(author)
                
                if author_name:
                    # Add author node
                    if author_name not in self.graph["authors"]:
                        self.graph["authors"][author_name] = {
                            "papers": [],
                            "co_authors": set(),
                        }
                    
                    self.graph["authors"][author_name]["papers"].append(paper_id)
                    
                    # Add co-author relationships
                    for other_author in authors:
                        if isinstance(other_author, dict):
                            other_name = other_author.get("name", "")
                        else:
                            other_name = str(other_author)
                        
                        if other_name and other_name != author_name:
                            self.graph["authors"][author_name]["co_authors"].add(other_name)
                            self.graph["co_authors"][author_name].add(other_name)
        
        # Add topic relationships
        topics = paper.get("topics", paper.get("categories", []))
        if isinstance(topics, list):
            for topic in topics:
                if topic not in self.graph["topics"]:
                    self.graph["topics"][topic] = {
                        "papers": [],
                        "keywords": [],
                    }
                
                self.graph["topics"][topic]["papers"].append(paper_id)
                self.graph["topic_papers"][topic].append(paper_id)
        
        # Add temporal relationship
        year = paper.get("year", paper.get("published_date", "")[:4])
        if year:
            self.graph["temporal"][year].append(paper_id)
    
    def add_citation(self, paper_id: str, cited_by: str):
        """Add citation relationship."""
        self.graph["citations"][paper_id].append(cited_by)
    
    def get_paper(self, paper_id: str) -> Optional[Dict]:
        """Get paper by ID."""
        return self.graph["papers"].get(paper_id)
    
    def get_author_papers(self, author_name: str) -> List[str]:
        """Get papers by author."""
        author = self.graph["authors"].get(author_name, {})
        return author.get("papers", [])
    
    def get_co_authors(self, author_name: str) -> Set[str]:
        """Get co-authors of an author."""
        return self.graph["co_authors"].get(author_name, set())
    
    def get_topic_papers(self, topic: str) -> List[str]:
        """Get papers in a topic."""
        return self.graph["topic_papers"].get(topic, [])
    
    def get_temporal_papers(self, year: str) -> List[str]:
        """Get papers from a specific year."""
        return self.graph["temporal"].get(year, [])
    
    def get_cited_by(self, paper_id: str) -> List[str]:
        """Get papers that cite this paper."""
        return self.graph["citations"].get(paper_id, [])
    
    def get_top_authors(self, n: int = 10) -> List[tuple]:
        """Get top authors by paper count."""
        author_counts = []
        for author, data in self.graph["authors"].items():
            author_counts.append((author, len(data.get("papers", []))))
        
        return sorted(author_counts, key=lambda x: x[1], reverse=True)[:n]
    
    def get_top_topics(self, n: int = 10) -> List[tuple]:
        """Get top topics by paper count."""
        topic_counts = []
        for topic, data in self.graph["topics"].items():
            topic_counts.append((topic, len(data.get("papers", []))))
        
        return sorted(topic_counts, key=lambda x: x[1], reverse=True)[:n]
    
    def get_trending_topics(self, recent_years: int = 2) -> List[tuple]:
        """Get trending topics in recent years."""
        current_year = datetime.now().year
        recent_topics = defaultdict(int)
        
        for year_str, paper_ids in self.graph["temporal"].items():
            try:
                year = int(year_str)
                if year >= current_year - recent_years:
                    for paper_id in paper_ids:
                        paper = self.graph["papers"].get(paper_id, {})
                        for topic in paper.get("topics", []):
                            recent_topics[topic] += 1
            except ValueError:
                continue
        
        return sorted(recent_topics.items(), key=lambda x: x[1], reverse=True)[:10]
    
    def save_graph(self, filepath: str = None):
        """Save knowledge graph to file."""
        if filepath is None:
            filepath = str(self.data_dir / "knowledge_graph.json")
        
        # Convert sets to lists for JSON serialization
        serializable = {
            "papers": self.graph["papers"],
            "authors": {
                name: {
                    "papers": data["papers"],
                    "co_authors": list(data["co_authors"]),
                }
                for name, data in self.graph["authors"].items()
            },
            "topics": self.graph["topics"],
            "citations": dict(self.graph["citations"]),
            "co_authors": {
                name: list(co_authors)
                for name, co_authors in self.graph["co_authors"].items()
            },
            "topic_papers": dict(self.graph["topic_papers"]),
            "temporal": dict(self.graph["temporal"]),
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
    
    def load_graph(self, filepath: str = None):
        """Load knowledge graph from file."""
        if filepath is None:
            filepath = str(self.data_dir / "knowledge_graph.json")
        
        if not Path(filepath).exists():
            return
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.graph["papers"] = data.get("papers", {})
        self.graph["topics"] = data.get("topics", {})
        self.graph["citations"] = defaultdict(list, data.get("citations", {}))
        self.graph["topic_papers"] = defaultdict(list, data.get("topic_papers", {}))
        self.graph["temporal"] = defaultdict(list, data.get("temporal", {}))
        
        # Restore authors
        for name, author_data in data.get("authors", {}).items():
            self.graph["authors"][name] = {
                "papers": author_data.get("papers", []),
                "co_authors": set(author_data.get("co_authors", [])),
            }
        
        # Restore co_authors
        for name, co_authors in data.get("co_authors", {}).items():
            self.graph["co_authors"][name] = set(co_authors)
    
    def get_stats(self) -> Dict:
        """Get knowledge graph statistics."""
        return {
            "total_papers": len(self.graph["papers"]),
            "total_authors": len(self.graph["authors"]),
            "total_topics": len(self.graph["topics"]),
            "total_citations": sum(len(v) for v in self.graph["citations"].values()),
            "top_authors": self.get_top_authors(5),
            "top_topics": self.get_top_topics(5),
        }


# Global instance
knowledge_graph = PaperKnowledgeGraph()


if __name__ == "__main__":
    # Test knowledge graph
    print("Testing paper knowledge graph...")
    
    # Add test papers
    test_papers = [
        {
            "id": "2607.20268",
            "title": "PoTRE: Test-Time Reasoning",
            "authors": ["Anmol Kankariya", "Sercan Ö. Arık"],
            "topics": ["AI_Agent", "reasoning"],
            "year": "2026",
        },
        {
            "id": "2607.20064",
            "title": "PRO-LONG: Programmatic Memory",
            "authors": ["Alexis Fox"],
            "topics": ["AI_Agent", "long-horizon"],
            "year": "2026",
        },
    ]
    
    for paper in test_papers:
        knowledge_graph.add_paper(paper)
    
    print(f"\nStats: {knowledge_graph.get_stats()}")
    print(f"Top authors: {knowledge_graph.get_top_authors(3)}")
    print(f"Top topics: {knowledge_graph.get_top_topics(3)}")
