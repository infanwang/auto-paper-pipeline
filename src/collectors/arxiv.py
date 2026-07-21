"""ArXiv 论文采集器 - V2.0"""

import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional


class ArxivCollector:
    """ArXiv API 论文采集器（带重试）"""
    
    BASE_URL = "https://export.arxiv.org/api/query"
    NS = {'atom': 'http://www.w3.org/2005/Atom'}
    
    def __init__(self, rate_limit: float = 3.0, max_retries: int = 3):
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.last_request_time = 0
    
    def _throttle(self):
        """请求限流"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def search(self, query: str, category: str = None, 
               max_results: int = 20, days_back: int = 7) -> List[Dict]:
        """搜索论文"""
        self._throttle()
        
        # 构建查询
        search_query = f"all:{query}"
        if category:
            search_query += f"+AND+cat:{category}"
        
        # 时间过滤
        cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        # URL编码
        import urllib.parse
        params = urllib.parse.urlencode({
            'search_query': search_query,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        })
        url = f"{self.BASE_URL}?{params}"
        
        data = None
        for attempt in range(self.max_retries):
            try:
                self._throttle()
                req = urllib.request.Request(url, headers={'User-Agent': 'PaperPipeline/2.0'})
                data = urllib.request.urlopen(req, timeout=30).read().decode()
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < self.max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    print(f"  ArXiv限流，等待{wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                print(f"  ArXiv搜索失败: {e}")
                return []
            except Exception as e:
                print(f"  ArXiv搜索失败: {e}")
                return []
        
        if data is None:
            return []
        
        try:
            root = ET.fromstring(data)
            papers = []
            
            for entry in root.findall('atom:entry', self.NS):
                arxiv_id = entry.find('atom:id', self.NS).text.split('/abs/')[-1]
                title = entry.find('atom:title', self.NS).text.strip().replace('\n', ' ')
                published = entry.find('atom:published', self.NS).text[:10]
                
                if published < cutoff:
                    continue
                
                abstract = entry.find('atom:summary', self.NS).text.strip()
                
                authors = []
                for author in entry.findall('atom:author', self.NS):
                    name = author.find('atom:name', self.NS).text
                    authors.append({'name': name})
                
                categories = [c.get('term') for c in entry.findall('atom:category', self.NS)]
                
                papers.append({
                    'id': arxiv_id,
                    'title': title,
                    'abstract': abstract,
                    'authors': authors,
                    'categories': categories,
                    'published_date': published,
                    'source': 'arxiv'
                })
            
            return papers
        except Exception as e:
            print(f"  ArXiv解析失败: {e}")
            return []
    
    def search_multi_topic(self, topics: Dict, days_back: int = 7) -> List[Dict]:
        """多主题搜索"""
        all_papers = []
        seen_ids = set()
        
        for topic_name, config in topics.items():
            for query in config['queries']:
                for category in config.get('categories', [None]):
                    papers = self.search(query, category=category, 
                                       max_results=15, days_back=days_back)
                    
                    for paper in papers:
                        if paper['id'] not in seen_ids:
                            paper['topic'] = topic_name
                            paper['query'] = query
                            all_papers.append(paper)
                            seen_ids.add(paper['id'])
        
        return all_papers


class SemanticScholarCollector:
    """Semantic Scholar 引用数据采集器"""
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
    
    def get_paper_details(self, arxiv_id: str) -> Optional[Dict]:
        """获取论文详情（含引用数）"""
        url = f"{self.BASE_URL}/paper/arXiv:{arxiv_id}?fields=citationCount,influentialCitationCount,tldr"
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'PaperPipeline/2.0'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                return {
                    'citation_count': data.get('citationCount', 0),
                    'influential_citations': data.get('influentialCitationCount', 0),
                    'tldr': data.get('tldr', {}).get('text', '') if data.get('tldr') else ''
                }
        except Exception:
            return None
    
    def batch_enrich(self, papers: List[Dict], limit: int = 30) -> List[Dict]:
        """批量增强论文数据"""
        for paper in papers[:limit]:
            details = self.get_paper_details(paper['id'])
            if details:
                paper.update(details)
            time.sleep(1)  # 限流
        
        return papers
