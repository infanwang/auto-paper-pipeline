#!/usr/bin/env python3
"""Paper recommendation system."""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Recommendation:
    """Paper recommendation."""
    paper_id: str
    title: str
    score: float
    reason: str
    similarity: float = 0.0


class PaperRecommender:
    """Paper recommendation system based on content and citations."""
    
    def __init__(self, data_dir: str = "/root/git/mimo/paper-pipeline/data"):
        self.data_dir = Path(data_dir)
        self.papers = {}
        self.tfidf = {}
        self.idf = {}
    
    def load_papers(self, filepath: str = None):
        """Load papers from JSON file."""
        if filepath is None:
            # Load all pipeline files
            for json_file in self.data_dir.glob("pipeline_*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for paper in data.get("papers", []):
                        paper_id = paper.get("id", "")
                        if paper_id:
                            self.papers[paper_id] = paper
                except Exception:
                    continue
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
                     'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                     'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
                     'once', 'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
                     'neither', 'each', 'every', 'all', 'any', 'few', 'more', 'most', 'other',
                     'some', 'such', 'no', 'only', 'own', 'same', 'than', 'too', 'very',
                     'just', 'because', 'if', 'when', 'where', 'how', 'what', 'which', 'who',
                     'whom', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'we', 'our',
                     'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they',
                     'them', 'their'}
        return [t for t in tokens if t not in stopwords and len(t) > 2]
    
    def compute_tf(self, text: str) -> Dict[str, float]:
        """Compute term frequency."""
        tokens = self.tokenize(text)
        tf = defaultdict(float)
        for token in tokens:
            tf[token] += 1
        # Normalize
        total = len(tokens) if tokens else 1
        return {k: v / total for k, v in tf.items()}
    
    def compute_idf(self, documents: List[str]):
        """Compute inverse document frequency."""
        doc_count = len(documents)
        df = defaultdict(int)
        
        for doc in documents:
            tokens = set(self.tokenize(doc))
            for token in tokens:
                df[token] += 1
        
        self.idf = {token: math.log(doc_count / count) for token, count in df.items()}
    
    def compute_tfidf(self, text: str) -> Dict[str, float]:
        """Compute TF-IDF vector."""
        tf = self.compute_tf(text)
        tfidf = {}
        for token, tf_val in tf.items():
            idf_val = self.idf.get(token, 1.0)
            tfidf[token] = tf_val * idf_val
        return tfidf
    
    def cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Compute cosine similarity between two vectors."""
        # Get common tokens
        common = set(vec1.keys()) & set(vec2.keys())
        
        if not common:
            return 0.0
        
        # Compute dot product and magnitudes
        dot = sum(vec1[k] * vec2[k] for k in common)
        mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot / (mag1 * mag2)
    
    def get_paper_text(self, paper: Dict) -> str:
        """Get text representation of paper."""
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        topics = " ".join(paper.get("topics", paper.get("categories", [])))
        return f"{title} {abstract} {topics}"
    
    def find_similar_papers(
        self,
        paper_id: str,
        n: int = 5,
        threshold: float = 0.1,
    ) -> List[Tuple[str, float]]:
        """
        Find similar papers to a given paper.
        
        Args:
            paper_id: Reference paper ID
            n: Number of recommendations
            threshold: Minimum similarity score
            
        Returns:
            List of (paper_id, similarity_score) tuples
        """
        if paper_id not in self.papers:
            return []
        
        ref_paper = self.papers[paper_id]
        ref_text = self.get_paper_text(ref_paper)
        ref_tfidf = self.compute_tfidf(ref_text)
        
        similarities = []
        
        for pid, paper in self.papers.items():
            if pid == paper_id:
                continue
            
            text = self.get_paper_text(paper)
            tfidf = self.compute_tfidf(text)
            sim = self.cosine_similarity(ref_tfidf, tfidf)
            
            if sim >= threshold:
                similarities.append((pid, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:n]
    
    def recommend_for_user(
        self,
        user_papers: List[str],
        n: int = 10,
    ) -> List[Recommendation]:
        """
        Recommend papers based on user's reading history.
        
        Args:
            user_papers: List of paper IDs user has read
            n: Number of recommendations
            
        Returns:
            List of Recommendation objects
        """
        # Build user profile from read papers
        user_text = ""
        for pid in user_papers:
            if pid in self.papers:
                user_text += " " + self.get_paper_text(self.papers[pid])
        
        if not user_text:
            return []
        
        user_tfidf = self.compute_tfidf(user_text)
        
        # Score all papers
        scores = []
        for pid, paper in self.papers.items():
            if pid in user_papers:
                continue
            
            text = self.get_paper_text(paper)
            tfidf = self.compute_tfidf(text)
            sim = self.cosine_similarity(user_tfidf, tfidf)
            
            if sim > 0:
                scores.append((pid, sim))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Create recommendations
        recommendations = []
        for pid, score in scores[:n]:
            paper = self.papers[pid]
            recommendations.append(Recommendation(
                paper_id=pid,
                title=paper.get("title", ""),
                score=score,
                reason="基于您的阅读历史",
                similarity=score,
            ))
        
        return recommendations
    
    def get_trending_papers(self, n: int = 10) -> List[Dict]:
        """Get trending papers based on citations and recency."""
        scored_papers = []
        
        for pid, paper in self.papers.items():
            citations = paper.get("citation_count", 0) or 0
            score = min(citations * 10, 100)
            scored_papers.append((pid, paper, score))
        
        scored_papers.sort(key=lambda x: x[2], reverse=True)
        
        return [
            {"paper": paper, "score": score}
            for pid, paper, score in scored_papers[:n]
        ]


# Global instance
recommender = PaperRecommender()


if __name__ == "__main__":
    # Test recommendation system
    print("Testing paper recommendation system...")
    
    # Load papers
    recommender.load_papers()
    print(f"Loaded {len(recommender.papers)} papers")
    
    # Compute IDF
    documents = [recommender.get_paper_text(p) for p in recommender.papers.values()]
    recommender.compute_idf(documents)
    
    # Get trending papers
    trending = recommender.get_trending_papers(5)
    print(f"\nTrending papers:")
    for item in trending:
        paper = item["paper"]
        print(f"  - {paper.get('title', 'N/A')[:60]}...")
