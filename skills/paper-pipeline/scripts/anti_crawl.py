#!/usr/bin/env python3
"""Anti-crawling measures for paper pipeline."""

import random
import time
import urllib.request
import urllib.error
from typing import Optional, Callable, Any


USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
]


class AntiCrawl:
    """Anti-crawling handler with User-Agent rotation and retry logic."""
    
    def __init__(
        self,
        min_delay: float = 3.0,
        max_delay: float = 8.0,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        initial_backoff: float = 5.0,
    ):
        """
        Initialize anti-crawl handler.
        
        Args:
            min_delay: Minimum delay between requests (seconds)
            max_delay: Maximum delay between requests (seconds)
            max_retries: Maximum number of retries on failure
            backoff_factor: Exponential backoff multiplier
            initial_backoff: Initial backoff time (seconds)
        """
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.initial_backoff = initial_backoff
        self.last_request_time = 0.0
        self.request_count = 0
    
    def get_headers(self, referer: Optional[str] = None) -> dict:
        """
        Get randomized headers for request.
        
        Args:
            referer: Optional referer URL
            
        Returns:
            Dictionary of headers
        """
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        }
        
        if referer:
            headers["Referer"] = referer
            headers["Sec-Fetch-Site"] = "same-origin"
        
        return headers
    
    def wait(self):
        """Wait for random delay between requests."""
        elapsed = time.time() - self.last_request_time
        delay = random.uniform(self.min_delay, self.max_delay)
        
        if elapsed < delay:
            sleep_time = delay - elapsed
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def create_request(self, url: str, referer: Optional[str] = None) -> urllib.request.Request:
        """
        Create a request with anti-crawl headers.
        
        Args:
            url: URL to request
            referer: Optional referer URL
            
        Returns:
            urllib.request.Request object
        """
        headers = self.get_headers(referer)
        return urllib.request.Request(url, headers=headers)
    
    def fetch(
        self,
        url: str,
        timeout: int = 60,
        referer: Optional[str] = None,
    ) -> bytes:
        """
        Fetch URL with anti-crawl measures and retry logic.
        
        Args:
            url: URL to fetch
            timeout: Request timeout in seconds
            referer: Optional referer URL
            
        Returns:
            Response bytes
            
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                # Wait before request
                self.wait()
                
                # Create and execute request
                req = self.create_request(url, referer)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.read()
                    
            except urllib.error.HTTPError as e:
                last_exception = e
                if e.code == 429:
                    # Rate limited - use longer backoff
                    wait_time = self.initial_backoff * (self.backoff_factor ** attempt) * 2
                    print(f"    Rate limited (429). Waiting {wait_time:.0f}s...")
                    time.sleep(wait_time)
                elif e.code >= 500:
                    # Server error - retry with backoff
                    wait_time = self.initial_backoff * (self.backoff_factor ** attempt)
                    print(f"    Server error ({e.code}). Retry {attempt + 1}/{self.max_retries}...")
                    time.sleep(wait_time)
                else:
                    # Client error - don't retry
                    raise
                    
            except Exception as e:
                last_exception = e
                wait_time = self.initial_backoff * (self.backoff_factor ** attempt)
                print(f"    Request failed: {e}. Retry {attempt + 1}/{self.max_retries}...")
                time.sleep(wait_time)
        
        # All retries failed
        raise last_exception
    
    def download_pdf(
        self,
        url: str,
        save_path: str,
        timeout: int = 120,
        referer: Optional[str] = None,
    ) -> bool:
        """
        Download PDF with anti-crawl measures and retry logic.
        
        Args:
            url: PDF URL
            save_path: Path to save PDF
            timeout: Download timeout in seconds
            referer: Optional referer URL
            
        Returns:
            True if successful, False otherwise
        """
        import os
        
        for attempt in range(self.max_retries):
            try:
                # Wait before request
                self.wait()
                
                # Create request
                req = self.create_request(url, referer)
                
                # Download with timeout
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = resp.read()
                
                # Validate PDF (check header)
                if not data[:5] == b'%PDF-':
                    print(f"    Invalid PDF header, retrying...")
                    continue
                
                # Save file
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'wb') as f:
                    f.write(data)
                
                return True
                
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait_time = self.initial_backoff * (self.backoff_factor ** attempt) * 2
                    print(f"    Rate limited (429). Waiting {wait_time:.0f}s...")
                    time.sleep(wait_time)
                elif e.code >= 500:
                    wait_time = self.initial_backoff * (self.backoff_factor ** attempt)
                    print(f"    Server error ({e.code}). Retry {attempt + 1}/{self.max_retries}...")
                    time.sleep(wait_time)
                else:
                    print(f"    Download failed: HTTP {e.code}")
                    return False
                    
            except Exception as e:
                wait_time = self.initial_backoff * (self.backoff_factor ** attempt)
                print(f"    Download failed: {e}. Retry {attempt + 1}/{self.max_retries}...")
                time.sleep(wait_time)
        
        print(f"    Download failed after {self.max_retries} retries")
        return False
    
    def get_stats(self) -> dict:
        """Get anti-crawl statistics."""
        return {
            "request_count": self.request_count,
            "last_request_time": self.last_request_time,
        }


# Global instance
anti_crawl = AntiCrawl()
