#!/usr/bin/env python3
"""Generate multilingual paper reports."""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Import multilingual module
from multilingual import (
    Language, LANGUAGE_NAMES, LanguageDetector, MultilingualTranslator,
    MultilingualSearch, MultilingualFormatter
)


class MultilingualReportGenerator:
    """Generate paper reports in multiple languages."""
    
    def __init__(self, data_dir: str = "/root/git/mimo/paper-pipeline/data"):
        self.data_dir = Path(data_dir)
        self.detector = LanguageDetector()
        self.translator = MultilingualTranslator()
        self.search = MultilingualSearch()
        self.formatter = MultilingualFormatter()
    
    def load_papers(self, date: str = None) -> List[Dict]:
        """Load papers from pipeline data."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        pipeline_file = self.data_dir / f"pipeline_{date}.json"
        if not pipeline_file.exists():
            print(f"  [!] Pipeline file not found: {pipeline_file}")
            return []
        
        with open(pipeline_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return data.get("papers", [])
    
    def generate_english_report(self, papers: List[Dict], date: str) -> str:
        """Generate English report."""
        lines = [
            "# AI Paper Daily Report",
            f"\n**Date**: {date}",
            f"**Total Papers**: {len(papers)}",
            "",
        ]
        
        # Language distribution
        lang_stats = self.search.get_language_stats(papers)
        lines.append("## Language Distribution")
        for lang_code, count in lang_stats.items():
            lang_name = LANGUAGE_NAMES.get(
                {v.value: v for v in Language}.get(lang_code, Language.ENGLISH),
                lang_code
            )
            lines.append(f"- {lang_name}: {count}")
        
        # Top papers
        lines.append("\n## Top Papers")
        for i, paper in enumerate(papers[:10], 1):
            lines.append(f"\n### {i}. {paper.get('title', 'N/A')}")
            lines.append(f"- **ArXiv**: {paper.get('id', 'N/A')}")
            lines.append(f"- **Date**: {paper.get('published_date', 'N/A')}")
            lines.append(f"- **Score**: {paper.get('llm_score', 'N/A')}/10")
            lines.append(f"- **Topics**: {', '.join(paper.get('topics', []))}")
            
            # Abstract snippet
            abstract = paper.get("abstract", "")
            if abstract:
                lines.append(f"- **Abstract**: {abstract[:200]}...")
        
        return "\n".join(lines)
    
    def generate_chinese_report(self, papers: List[Dict], date: str) -> str:
        """Generate Chinese report."""
        lines = [
            "# AI论文日报",
            f"\n**日期**: {date}",
            f"**论文总数**: {len(papers)}",
            "",
        ]
        
        # Language distribution
        lang_stats = self.search.get_language_stats(papers)
        lines.append("## 语言分布")
        for lang_code, count in lang_stats.items():
            lang_name = LANGUAGE_NAMES.get(
                {v.value: v for v in Language}.get(lang_code, Language.ENGLISH),
                lang_code
            )
            lines.append(f"- {lang_name}: {count}")
        
        # Topic distribution
        topic_counts = {}
        for paper in papers:
            for topic in paper.get("topics", []):
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        lines.append("\n## 主题分布")
        for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- {topic}: {count}篇")
        
        # Top papers
        lines.append("\n## 热门论文")
        for i, paper in enumerate(papers[:10], 1):
            title = paper.get("title", "N/A")
            # Translate title to Chinese
            title_zh = self.translator.translate_text(title, target_lang="zh")
            if title_zh:
                title = f"{title_zh}\n  ({title})"
            
            lines.append(f"\n### {i}. {title}")
            lines.append(f"- **ArXiv**: {paper.get('id', 'N/A')}")
            lines.append(f"- **日期**: {paper.get('published_date', 'N/A')}")
            lines.append(f"- **评分**: {paper.get('llm_score', 'N/A')}/10")
            lines.append(f"- **主题**: {', '.join(paper.get('topics', []))}")
            
            # Abstract snippet
            abstract = paper.get("abstract", "")
            if abstract:
                abstract_zh = self.translator.translate_text(abstract[:300], target_lang="zh")
                if abstract_zh:
                    lines.append(f"- **摘要**: {abstract_zh[:200]}...")
        
        return "\n".join(lines)
    
    def generate_french_report(self, papers: List[Dict], date: str) -> str:
        """Generate French report."""
        lines = [
            "# Rapport Quotidien sur les Articles IA",
            f"\n**Date**: {date}",
            f"**Nombre total d'articles**: {len(papers)}",
            "",
        ]
        
        # Language distribution
        lang_stats = self.search.get_language_stats(papers)
        lines.append("## Distribution des langues")
        for lang_code, count in lang_stats.items():
            lang_name = LANGUAGE_NAMES.get(
                {v.value: v for v in Language}.get(lang_code, Language.ENGLISH),
                lang_code
            )
            lines.append(f"- {lang_name}: {count}")
        
        # Top papers
        lines.append("\n## Articles principaux")
        for i, paper in enumerate(papers[:10], 1):
            title = paper.get("title", "N/A")
            title_fr = self.translator.translate_text(title, target_lang="fr")
            if title_fr:
                title = f"{title_fr}\n  ({title})"
            
            lines.append(f"\n### {i}. {title}")
            lines.append(f"- **ArXiv**: {paper.get('id', 'N/A')}")
            lines.append(f"- **Date**: {paper.get('published_date', 'N/A')}")
            lines.append(f"- **Score**: {paper.get('llm_score', 'N/A')}/10")
        
        return "\n".join(lines)
    
    def generate_all_reports(self, date: str = None) -> Dict[str, str]:
        """Generate reports in all supported languages."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        print(f"Generating multilingual reports for {date}...")
        
        # Load papers
        papers = self.load_papers(date)
        if not papers:
            print("  [!] No papers found")
            return {}
        
        print(f"  Loaded {len(papers)} papers")
        
        # Generate reports
        reports = {
            "en": self.generate_english_report(papers, date),
            "zh": self.generate_chinese_report(papers, date),
            "fr": self.generate_french_report(papers, date),
        }
        
        # Save reports
        report_dir = self.data_dir.parent / "reports" / "multilingual"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        for lang, content in reports.items():
            report_file = report_dir / f"report_{date}_{lang}.md"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  Saved: {report_file.name}")
        
        return reports


def main():
    """Main function."""
    generator = MultilingualReportGenerator()
    reports = generator.generate_all_reports()
    
    print("\n" + "="*60)
    print("Multilingual reports generated successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
