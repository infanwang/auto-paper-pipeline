#!/usr/bin/env python3
"""Deduplication module for paper pipeline."""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher
from typing import Optional


DEDUP_INDEX = Path("/root/git/mimo/paper-pipeline/data/dedup_index.json")


class Deduplicator:
    """Handles paper deduplication across runs."""
    
    def __init__(self, index_path: Optional[Path] = None):
        """
        Initialize deduplicator.
        
        Args:
            index_path: Path to dedup index file
        """
        self.index_path = index_path or DEDUP_INDEX
        self.index = self._load_index()
    
    def _load_index(self) -> dict:
        """Load dedup index from file."""
        if self.index_path.exists():
            try:
                return json.loads(self.index_path.read_text(encoding="utf-8"))
            except Exception:
                return {"papers": {}, "hashes": {}}
        return {"papers": {}, "hashes": {}}
    
    def _save_index(self):
        """Save dedup index to file."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps(self.index, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
    
    def is_duplicate(
        self,
        paper_id: str,
        title: Optional[str] = None,
        similarity_threshold: float = 0.85
    ) -> bool:
        """
        Check if paper is a duplicate.
        
        Args:
            paper_id: arXiv ID
            title: Paper title for similarity check
            similarity_threshold: Title similarity threshold (0-1)
            
        Returns:
            True if duplicate, False otherwise
        """
        # ID-based dedup
        if paper_id in self.index["papers"]:
            return True
        
        # Title similarity dedup
        if title:
            title_lower = title.lower().strip()
            for existing in self.index["papers"].values():
                existing_title = existing.get("title", "").lower().strip()
                if existing_title:
                    similarity = SequenceMatcher(None, title_lower, existing_title).ratio()
                    if similarity > similarity_threshold:
                        return True
        
        return False
    
    def file_hash(self, file_path: str) -> str:
        """
        Calculate MD5 hash of file.
        
        Args:
            file_path: Path to file
            
        Returns:
            MD5 hash string
        """
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def is_file_duplicate(self, file_path: str) -> bool:
        """
        Check if file is a duplicate by content hash.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            True if duplicate, False otherwise
        """
        if not Path(file_path).exists():
            return False
        
        file_hash = self.file_hash(file_path)
        return file_hash in self.index["hashes"]
    
    def register(
        self,
        paper_id: str,
        title: str,
        pdf_path: Optional[str] = None,
        metadata: Optional[dict] = None
    ):
        """
        Register paper in dedup index.
        
        Args:
            paper_id: arXiv ID
            title: Paper title
            pdf_path: Path to downloaded PDF
            metadata: Additional metadata
        """
        entry = {
            "title": title,
            "downloaded_at": datetime.now().isoformat(),
        }
        
        if pdf_path:
            entry["pdf_path"] = pdf_path
            # Calculate and store file hash
            if Path(pdf_path).exists():
                file_hash = self.file_hash(pdf_path)
                self.index["hashes"][file_hash] = paper_id
        
        if metadata:
            entry.update(metadata)
        
        self.index["papers"][paper_id] = entry
        self._save_index()
    
    def get_paper(self, paper_id: str) -> Optional[dict]:
        """
        Get paper info from index.
        
        Args:
            paper_id: arXiv ID
            
        Returns:
            Paper info dict or None
        """
        return self.index["papers"].get(paper_id)
    
    def get_stats(self) -> dict:
        """Get dedup statistics."""
        return {
            "total_papers": len(self.index["papers"]),
            "total_hashes": len(self.index["hashes"]),
            "index_path": str(self.index_path),
        }
    
    def cleanup(self, verify_files: bool = True):
        """
        Clean up dedup index by removing entries with missing files.
        
        Args:
            verify_files: If True, check if files exist
        """
        if not verify_files:
            return
        
        removed = 0
        for paper_id, entry in list(self.index["papers"].items()):
            pdf_path = entry.get("pdf_path")
            if pdf_path and not Path(pdf_path).exists():
                # Remove from papers
                del self.index["papers"][paper_id]
                
                # Remove associated hash
                for hash_val, pid in list(self.index["hashes"].items()):
                    if pid == paper_id:
                        del self.index["hashes"][hash_val]
                
                removed += 1
        
        if removed > 0:
            self._save_index()
            print(f"Cleaned up {removed} entries from dedup index")


# Global instance
dedup = Deduplicator()
