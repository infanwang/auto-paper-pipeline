#!/usr/bin/env python3
"""Multi-language paper support module.

Supports all 7 UN official languages:
- English (en)
- French (fr)
- Spanish (es)
- Russian (ru)
- Chinese (zh)
- Arabic (ar)
- Portuguese (pt)
"""

import json
import re
import urllib.request
import urllib.parse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class Language(Enum):
    """UN official languages."""
    ENGLISH = "en"
    FRENCH = "fr"
    SPANISH = "es"
    RUSSIAN = "ru"
    CHINESE = "zh"
    ARABIC = "ar"
    PORTUGUESE = "pt"


# Language names in native script
LANGUAGE_NAMES = {
    Language.ENGLISH: "English",
    Language.FRENCH: "Français",
    Language.SPANISH: "Español",
    Language.RUSSIAN: "Русский",
    Language.CHINESE: "中文",
    Language.ARABIC: "العربية",
    Language.PORTUGUESE: "Português",
}

# RTL languages
RTL_LANGUAGES = {Language.ARABIC}

# Common words for language detection
LANGUAGE_PATTERNS = {
    Language.ENGLISH: r'\b(the|is|are|was|were|have|has|had|do|does|did|will|would|could|should|may|might|can|and|but|or|not|this|that|with|from|for|in|on|at|to|by|of|as|into|through|during|before|after|above|below|between|out|off|over|under|again|further|then|once)\b',
    Language.FRENCH: r'\b(le|la|les|un|une|des|est|sont|avoir|fait|faire|peut|être|ce|cette|avec|pour|dans|sur|par|de|du|des|au|aux|et|mais|ou|ne|pas|plus|aussi|bien|tout|très|comme|mais|donc|car|ni|soit|si|quand|où|comment|pourquoi|qui|que|quel|quelle|dont)\b',
    Language.SPANISH: r'\b(el|la|los|las|un|una|unos|unas|es|son|estar|tener|hacer|poder|deber|como|con|para|por|en|de|del|al|sin|sobre|entre|hasta|desde|durante|ante|según|tras|y|o|pero|sino|ni|que|este|esta|ese|esa|aquel|aquella)\b',
    Language.RUSSIAN: r'[а-яА-ЯёЁ]{3,}',
    Language.CHINESE: r'[\u4e00-\u9fff]{2,}',
    Language.ARABIC: r'[\u0600-\u06ff]{2,}',
    Language.PORTUGUESE: r'\b(o|a|os|as|um|uma|uns|umas|é|são|estar|ter|fazer|poder|dever|como|com|para|por|em|de|do|da|dos|das|ao|aos|sem|sobre|entre|até|desde|durante|após|antes|segundo|conforme|e|mas|ou|nem|que|este|esta|esse|essa|aquele|aquela)\b',
}


@dataclass
class PaperTranslation:
    """Paper translation in a specific language."""
    language: Language
    title: str
    abstract: str
    is_original: bool = False


@dataclass
class MultilingualPaper:
    """Paper with multi-language support."""
    paper_id: str
    original_language: Language
    translations: Dict[Language, PaperTranslation]
    authors: List[str]
    topics: List[str]
    year: str
    venue: str
    citation_count: int = 0


class LanguageDetector:
    """Detect paper language based on content."""
    
    def __init__(self):
        self.patterns = {}
        for lang, pattern in LANGUAGE_PATTERNS.items():
            try:
                self.patterns[lang] = re.compile(pattern, re.IGNORECASE)
            except re.error:
                self.patterns[lang] = None
    
    def detect(self, text: str) -> Language:
        """
        Detect language of text.
        
        Args:
            text: Input text
            
        Returns:
            Detected language
        """
        if not text:
            return Language.ENGLISH
        
        scores = {}
        
        for lang, pattern in self.patterns.items():
            if pattern is None:
                scores[lang] = 0
                continue
            
            matches = pattern.findall(text)
            scores[lang] = len(matches)
        
        # Special handling for Chinese and Arabic (character-based)
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        arabic_chars = len(re.findall(r'[\u0600-\u06ff]', text))
        
        if chinese_chars > 0:
            scores[Language.CHINESE] = max(scores.get(Language.CHINESE, 0), chinese_chars * 10)
        
        if arabic_chars > 0:
            scores[Language.ARABIC] = max(scores.get(Language.ARABIC, 0), arabic_chars * 10)
        
        # Get language with highest score
        if not scores or max(scores.values()) == 0:
            return Language.ENGLISH
        
        return max(scores, key=scores.get)
    
    def detect_from_metadata(self, paper: Dict) -> Language:
        """
        Detect language from paper metadata.
        
        Args:
            paper: Paper dictionary
            
        Returns:
            Detected language
        """
        # Check explicit language field
        lang_code = paper.get("language", "")
        if lang_code:
            for lang in Language:
                if lang.value == lang_code:
                    return lang
        
        # Detect from abstract
        abstract = paper.get("abstract", "")
        if abstract:
            return self.detect(abstract)
        
        # Detect from title
        title = paper.get("title", "")
        if title:
            return self.detect(title)
        
        return Language.ENGLISH


class MultilingualTranslator:
    """Translate papers to multiple languages."""
    
    def __init__(self):
        self.cache = {}
    
    def translate_text(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "en",
    ) -> Optional[str]:
        """
        Translate text using Google Translate.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text or None
        """
        if not text:
            return None
        
        # Check cache
        cache_key = f"{text[:100]}_{target_lang}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            encoded_text = urllib.parse.quote(text[:5000])
            url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={source_lang}&tl={target_lang}&dt=t&q={encoded_text}"
            
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
                result = json.loads(data.decode("utf-8"))
                
                translated = "".join([item[0] for item in result[0] if item[0]])
                
                if translated:
                    self.cache[cache_key] = translated
                    return translated
        except Exception as e:
            print(f"  [!] Translation error: {e}")
        
        return None
    
    def translate_paper(
        self,
        paper: Dict,
        target_languages: List[Language] = None,
    ) -> MultilingualPaper:
        """
        Translate paper to multiple languages.
        
        Args:
            paper: Paper dictionary
            target_languages: List of target languages
            
        Returns:
            MultilingualPaper object
        """
        if target_languages is None:
            target_languages = list(Language)
        
        detector = LanguageDetector()
        original_lang = detector.detect_from_metadata(paper)
        
        translations = {}
        
        for lang in target_languages:
            if lang == original_lang:
                # Original language
                translations[lang] = PaperTranslation(
                    language=lang,
                    title=paper.get("title", ""),
                    abstract=paper.get("abstract", ""),
                    is_original=True,
                )
            else:
                # Translate
                title = self.translate_text(
                    paper.get("title", ""),
                    source_lang=original_lang.value,
                    target_lang=lang.value,
                )
                
                abstract = self.translate_text(
                    paper.get("abstract", ""),
                    source_lang=original_lang.value,
                    target_lang=lang.value,
                )
                
                translations[lang] = PaperTranslation(
                    language=lang,
                    title=title or paper.get("title", ""),
                    abstract=abstract or paper.get("abstract", ""),
                    is_original=False,
                )
        
        return MultilingualPaper(
            paper_id=paper.get("id", ""),
            original_language=original_lang,
            translations=translations,
            authors=paper.get("authors", []),
            topics=paper.get("topics", []),
            year=paper.get("year", ""),
            venue=paper.get("venue", ""),
            citation_count=paper.get("citation_count", 0),
        )


class MultilingualSearch:
    """Multi-language paper search."""
    
    def __init__(self):
        self.detector = LanguageDetector()
        self.translator = MultilingualTranslator()
    
    def search_with_translation(
        self,
        query: str,
        target_lang: Language = Language.ENGLISH,
        max_results: int = 20,
    ) -> List[Dict]:
        """
        Search papers and translate results.
        
        Args:
            query: Search query
            target_lang: Target language for results
            max_results: Maximum results
            
        Returns:
            List of papers with translations
        """
        # Import here to avoid circular imports
        from multi_source_crawler import MultiSourceCrawler
        
        crawler = MultiSourceCrawler()
        
        # Search in original language
        papers = crawler.search_all(query, max_results=max_results)
        
        # Translate if needed
        if target_lang != Language.ENGLISH:
            translated_papers = []
            for paper in papers:
                multilingual = self.translator.translate_paper(paper, [target_lang])
                translated_papers.append({
                    **paper,
                    f"title_{target_lang.value}": multilingual.translations[target_lang].title,
                    f"abstract_{target_lang.value}": multilingual.translations[target_lang].abstract,
                })
            return translated_papers
        
        return papers
    
    def filter_by_language(
        self,
        papers: List[Dict],
        language: Language,
    ) -> List[Dict]:
        """
        Filter papers by language.
        
        Args:
            papers: List of papers
            language: Target language
            
        Returns:
            Filtered papers
        """
        filtered = []
        for paper in papers:
            paper_lang = self.detector.detect_from_metadata(paper)
            if paper_lang == language:
                filtered.append(paper)
        
        return filtered
    
    def get_language_stats(self, papers: List[Dict]) -> Dict[str, int]:
        """
        Get language distribution of papers.
        
        Args:
            papers: List of papers
            
        Returns:
            Language statistics
        """
        stats = defaultdict(int)
        
        for paper in papers:
            lang = self.detector.detect_from_metadata(paper)
            stats[lang.value] += 1
        
        return dict(stats)


# Global instances
language_detector = LanguageDetector()
multilingual_translator = MultilingualTranslator()
multilingual_search = MultilingualSearch()


if __name__ == "__main__":
    # Test multi-language support
    print("Testing multi-language support...")
    print()
    
    # Test language detection
    test_texts = {
        "English": "We propose a novel method for machine learning.",
        "French": "Nous proposons une nouvelle méthode pour l'apprentissage automatique.",
        "Spanish": "Proponemos un nuevo método para el aprendizaje automático.",
        "Russian": "Мы предлагаем новый метод машинного обучения.",
        "Chinese": "我们提出了一种新的机器学习方法。",
        "Arabic": "نقترح طريقة جديدة للتعلم الآلي.",
        "Portuguese": "Propomos um novo método para aprendizado de máquina.",
    }
    
    print("Language detection:")
    for lang_name, text in test_texts.items():
        detected = language_detector.detect(text)
        print(f"  {lang_name}: {detected.value} (expected: {lang_name[:2].lower()})")
    
    print()
    
    # Test translation
    print("Translation test:")
    test_text = "We propose a novel method for machine learning."
    for lang in Language:
        translated = multilingual_translator.translate_text(test_text, target_lang=lang.value)
        if translated:
            print(f"  {lang.value}: {translated[:50]}...")
        else:
            print(f"  {lang.value}: [translation failed]")
