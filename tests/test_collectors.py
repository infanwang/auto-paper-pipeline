"""单元测试 - 论文采集器"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from collectors.arxiv import ArxivCollector, SemanticScholarCollector


class TestArxivCollector:
    """ArXiv采集器测试"""
    
    def test_init(self):
        """测试初始化"""
        collector = ArxivCollector(rate_limit=1.0)
        assert collector.rate_limit == 1.0
        assert collector.last_request_time == 0
    
    def test_throttle(self):
        """测试限流"""
        collector = ArxivCollector(rate_limit=0.1)
        collector._throttle()
        assert collector.last_request_time > 0
    
    @patch('collectors.arxiv.urllib.request.urlopen')
    def test_search_success(self, mock_urlopen):
        """测试搜索成功"""
        mock_response = MagicMock()
        mock_response.read.return_value = b'''<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <id>http://arxiv.org/abs/2607.16193v1</id>
                <title>Test Paper</title>
                <summary>Test abstract content</summary>
                <published>2026-07-17T00:00:00Z</published>
                <author><name>Test Author</name></author>
                <category term="cs.AI"/>
            </entry>
        </feed>'''
        mock_urlopen.return_value = mock_response
        
        collector = ArxivCollector(rate_limit=0)
        papers = collector.search("test query", max_results=5, days_back=30)
        
        assert len(papers) == 1
        assert papers[0]['id'] == '2607.16193v1'
        assert papers[0]['title'] == 'Test Paper'
    
    @patch('collectors.arxiv.urllib.request.urlopen')
    def test_search_failure(self, mock_urlopen):
        """测试搜索失败"""
        mock_urlopen.side_effect = Exception("Network error")
        
        collector = ArxivCollector(rate_limit=0)
        papers = collector.search("test query")
        
        assert papers == []
    
    def test_search_multi_topic(self):
        """测试多主题搜索"""
        topics = {
            "AI_Agent": {
                "queries": ["agent tool use"],
                "categories": ["cs.AI"]
            }
        }
        
        collector = ArxivCollector(rate_limit=0)
        with patch.object(collector, 'search', return_value=[]):
            papers = collector.search_multi_topic(topics, days_back=7)
            assert isinstance(papers, list)


class TestSemanticScholarCollector:
    """Semantic Scholar采集器测试"""
    
    def test_init(self):
        """测试初始化"""
        collector = SemanticScholarCollector()
        assert collector.BASE_URL == "https://api.semanticscholar.org/graph/v1"
    
    @patch('collectors.arxiv.urllib.request.urlopen')
    def test_get_paper_details(self, mock_urlopen):
        """测试获取论文详情"""
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"citationCount": 42, "influentialCitationCount": 5, "tldr": {"text": "Test TLDR"}}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response
        
        collector = SemanticScholarCollector()
        details = collector.get_paper_details("2607.16193")
        
        assert details['citation_count'] == 42
        assert details['influential_citations'] == 5
        assert details['tldr'] == "Test TLDR"
    
    @patch('collectors.arxiv.urllib.request.urlopen')
    def test_get_paper_details_failure(self, mock_urlopen):
        """测试获取详情失败"""
        mock_urlopen.side_effect = Exception("API error")
        
        collector = SemanticScholarCollector()
        details = collector.get_paper_details("invalid_id")
        
        assert details is None
