"""单元测试 - Web模块"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from web.static_generator import StaticSiteGenerator


class TestStaticSiteGenerator:
    """静态站点生成器测试"""
    
    def test_init(self):
        """测试初始化"""
        gen = StaticSiteGenerator("test_docs")
        assert gen.output_dir == Path("test_docs")
    
    def test_generate_index(self, sample_papers):
        """测试主页生成"""
        gen = StaticSiteGenerator("test_docs")
        gen._generate_index(sample_papers, "2026-07-21")
        
        index_file = Path("test_docs/index.html")
        assert index_file.exists()
        
        content = index_file.read_text()
        assert "AI Paper Pipeline" in content
        assert "2026-07-21" in content
        assert "20" in content  # 论文数量
    
    def test_generate_topic_page(self, sample_papers):
        """测试领域页面生成"""
        gen = StaticSiteGenerator("test_docs")
        gen._generate_topic_page("AI_Agent", sample_papers[:2], "2026-07-21")
        
        topic_file = Path("test_docs/AI_Agent/index.html")
        assert topic_file.exists()
        
        content = topic_file.read_text()
        assert "AI_Agent" in content
        assert "2篇" in content
    
    def test_generate_paper_page(self, sample_paper):
        """测试论文页面生成"""
        gen = StaticSiteGenerator("test_docs")
        gen._generate_paper_page(sample_paper)
        
        paper_file = Path("test_docs/papers/2607.16193.html")
        assert paper_file.exists()
        
        content = paper_file.read_text()
        assert "Knowing the Self" in content  # 使用实际标题
        assert "2607.16193" in content
    
    def test_generate_full(self, sample_papers):
        """测试完整生成"""
        gen = StaticSiteGenerator("test_docs")
        gen.generate(sample_papers, "2026-07-21")
        
        assert Path("test_docs/index.html").exists()
        assert Path("test_docs/AI_Agent/index.html").exists()
        assert Path("test_docs/LLM推理优化/index.html").exists()
