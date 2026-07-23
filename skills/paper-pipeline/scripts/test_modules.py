#!/usr/bin/env python3
"""Test script for anti-crawl and dedup modules."""

import sys
import time
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from anti_crawl import AntiCrawl
from dedup import Deduplicator


def test_anti_crawl():
    """Test anti-crawl module."""
    print("="*50)
    print("Testing Anti-Crawl Module")
    print("="*50)
    
    ac = AntiCrawl(min_delay=1, max_delay=2, max_retries=2)
    
    # Test headers
    headers = ac.get_headers()
    print(f"\n1. Headers generated:")
    print(f"   User-Agent: {headers['User-Agent'][:50]}...")
    print(f"   Accept: {headers['Accept'][:50]}...")
    
    # Test wait
    print(f"\n2. Testing wait mechanism...")
    start = time.time()
    ac.wait()
    elapsed = time.time() - start
    print(f"   Waited {elapsed:.2f}s (expected 1-2s)")
    
    # Test request creation
    print(f"\n3. Testing request creation...")
    req = ac.create_request("https://example.com", referer="https://google.com")
    print(f"   URL: {req.full_url}")
    print(f"   Headers: {len(req.headers)} headers")
    
    print(f"\n✓ Anti-crawl module tests passed\n")


def test_dedup():
    """Test dedup module."""
    print("="*50)
    print("Testing Dedup Module")
    print("="*50)
    
    # Use test index
    test_index = Path("/tmp/test_dedup_index.json")
    dedup = Deduplicator(index_path=test_index)
    
    # Test 1: Register and check duplicate
    print(f"\n1. Testing ID-based dedup...")
    dedup.register("2607.00001", "Test Paper 1")
    assert dedup.is_duplicate("2607.00001"), "Should be duplicate"
    assert not dedup.is_duplicate("2607.00002"), "Should not be duplicate"
    print(f"   ✓ ID dedup works")
    
    # Test 2: Title similarity dedup
    print(f"\n2. Testing title similarity dedup...")
    dedup.register("2607.00003", "A Novel Approach to Machine Learning")
    assert dedup.is_duplicate("2607.00004", "A Novel Approach to Machine Learning"), "Should be duplicate (same title)"
    assert dedup.is_duplicate("2607.00005", "A Novel Approach to Machine Learning Systems"), "Should be duplicate (similar title)"
    assert not dedup.is_duplicate("2607.00006", "Completely Different Paper"), "Should not be duplicate"
    print(f"   ✓ Title dedup works")
    
    # Test 3: File hash dedup
    print(f"\n3. Testing file hash dedup...")
    test_file = Path("/tmp/test_paper.pdf")
    test_file.write_bytes(b"%PDF-1.4 test content")
    
    dedup.register("2607.00010", "Test File Paper", str(test_file))
    assert dedup.is_file_duplicate(str(test_file)), "Should be file duplicate"
    print(f"   ✓ File hash dedup works")
    
    # Test 4: Get paper info
    print(f"\n4. Testing get_paper...")
    paper = dedup.get_paper("2607.00001")
    assert paper is not None, "Should find paper"
    assert paper["title"] == "Test Paper 1", "Title mismatch"
    print(f"   ✓ get_paper works")
    
    # Test 5: Stats
    print(f"\n5. Testing stats...")
    stats = dedup.get_stats()
    print(f"   Papers: {stats['total_papers']}")
    print(f"   Hashes: {stats['total_hashes']}")
    assert stats["total_papers"] > 0, "Should have papers"
    print(f"   ✓ Stats work")
    
    # Cleanup
    test_file.unlink(missing_ok=True)
    test_index.unlink(missing_ok=True)
    
    print(f"\n✓ Dedup module tests passed\n")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Paper Pipeline Module Tests")
    print("="*60 + "\n")
    
    try:
        test_anti_crawl()
        test_dedup()
        
        print("="*60)
        print("All tests passed! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
