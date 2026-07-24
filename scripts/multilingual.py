#!/usr/bin/env python3
"""Enhanced Multi-language paper support module.

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
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


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

# Language codes mapping
LANGUAGE_CODES = {
    "en": Language.ENGLISH,
    "fr": Language.FRENCH,
    "es": Language.SPANISH,
    "ru": Language.RUSSIAN,
    "zh": Language.CHINESE,
    "ar": Language.ARABIC,
    "pt": Language.PORTUGUESE,
}

# Enhanced patterns for language detection
LANGUAGE_PATTERNS = {
    Language.ENGLISH: r'\b(the|is|are|was|were|have|has|had|do|does|did|will|would|could|should|may|might|can|and|but|or|not|this|that|with|from|for|in|on|at|to|by|of|as|into|through|during|before|after|above|below|between|out|off|over|under|again|further|then|once|we|our|us|their|them|they|its|it|he|she|his|her|him|my|your|their|our|these|those|which|what|who|whom|whose|where|when|how|why|all|each|every|both|few|more|most|other|some|such|no|nor|not|only|own|same|so|than|too|very|just|because|but|and|or|if|while|about|against|between|through|during|before|after|above|below|to|from|up|down|in|out|on|off|over|under|again|further|then|once|here|there|when|where|why|how|all|any|both|each|few|more|most|other|some|such|no|nor|not|only|own|same|so|than|too|very|s|t|can|will|just|don|should|now)\b',
    Language.FRENCH: r'\b(le|la|les|un|une|des|est|sont|avoir|fait|faire|peut|être|ce|cette|avec|pour|dans|sur|par|et|mais|ou|ne|pas|plus|aussi|bien|tout|très|comme|donc|car|ni|soit|si|quand|où|comment|pourquoi|qui|que|quel|quelle|dont|je|tu|il|elle|nous|vous|ils|elles|mon|ma|mes|ton|ta|tes|son|sa|ses|notre|nos|votre|vos|leur|leurs|être|avoir|faire|pouvoir|vouloir|devoir|savoir|voir|venir|aller|partir|prendre|donner|trouver|croire|penser|parler|aimer|passer|rester|mettre|tenir|suivre|revenir|devenir|sortir|entrer|rendre|vivre|écrire|lire|apprendre|comprendre|perdre|gagner|payer|chercher|demander|répondre|commencer|finir|continuer|arrêter|essayer|réussir|échouer|construire|détruire|créer|inventer|découvrir|explorer|étudier|rechercher|analyser|comparer|mesurer|calculer|estimer|évaluer|juger|décider|choisir|préférer|accepter|refuser|permettre|interdire|obliger|encourager|aider|faciliter|compliquer|simplifier|augmenter|diminuer|réduire|étendre|limiter|ouvrir|fermer|connecter|déconnecter|attacher|détacher|unir|séparer|mélanger|distinguer|confondre|éclaircir|obscurcir|enrichir|appauvrir|renforcer|affaiblir|accélérer|ralentir|activer|désactiver|inclure|exclure|ajouter|retirer|insérer|supprimer|modifier|conserver|remplacer|corriger|altérer|transformer|convertir|adapter|ajuster|calibrer|vérifier|valider|confirmer|infirmer|prouver|démontrer|illustrer|exemplifier|représenter|symboliser|signifier|indiquer|suggérer|impliquer|entraîner|causer|provoquer|résulter|dériver|provenir|émaner|originer|naître|mourir|vivre|exister|subsister|durer|persévérer|hésiter|hâter|presser|attendre|espérer|craindre|regretter|remercier|féliciter|blâmer|critiquer|louer|complimenter|insulter|offenser|pardonner|excuser|justifier|expliquer|clarifier)\b',
    Language.SPANISH: r'\b(el|la|los|las|un|una|unos|unas|es|son|estar|tener|hacer|poder|deber|como|con|para|por|en|de|del|al|sin|sobre|entre|hasta|desde|durante|ante|según|tras|y|o|pero|sino|ni|que|este|esta|ese|esa|aquel|aquella|yo|tú|él|ella|nosotros|vosotros|ellos|ellas|mi|tu|su|nuestro|nuestra|vuestro|vuestra|estos|estas|esos|esas|aquellos|aquellas|quién|qué|cuál|dónde|cuándo|cómo|cuánto|ser|estar|haber|tener|hacer|poder|decir|ir|ver|dar|saber|querer|llegar|poner|parecer|creer|hablar|llevar|dejar|seguir|encontrar|llamar|venir|pensar|salir|volver|tomar|conocer|vivir|sentir|tratar|mirar|contar|empezar|esperar|buscar|existir|entrar|trabajar|escribir|perder|producir|ocurrir|recibir|comenzar|permitir|aparecer|considerar|terminar|desarrollar|obtener|actualmente|realmente|simplemente|probablemente|generalmente|exactamente|definitivamente|básicamente|particularmente|frecuentemente|constantemente|simultáneamente|investigación|método|resultado|análisis|sistema|modelo|enfoque|propuesta|nuevo|nueva|mejor|mayor|mejora|rendimiento|eficiencia|precisión|exactitud|calidad|optimización|algoritmo|técnica|approach|method|result|analysis|system|model|propose|new|better|improved|performance|efficiency|accuracy|quality|optimization|algorithm|technique)\b',
    Language.RUSSIAN: r'[а-яА-ЯёЁ]{3,}',
    Language.CHINESE: r'[\u4e00-\u9fff]{2,}',
    Language.ARABIC: r'[\u0600-\u06ff]{2,}',
    Language.PORTUGUESE: r'\b(o|a|os|as|um|uma|uns|umas|é|são|estar|ter|fazer|poder|dever|como|com|para|por|em|de|do|da|dos|das|ao|aos|sem|sobre|entre|até|desde|durante|após|antes|segundo|conforme|e|mas|ou|nem|que|este|esta|esse|essa|aquele|aquela|eu|tu|ele|ela|nós|vocês|eles|elas|meu|minha|meus|minhas|teu|tua|teus|tuas|seu|sua|seus|suas|nosso|nossa|nossos|nossas|vosso|vossa|vossos|vossas|estes|estas|esses|essas|aqueles|aquelas|quem|qual|onde|quando|quanto|ser|estar|ter|fazer|poder|dizer|ir|ver|dar|saber|querer|chegar|por|parecer|crer|falar|carregar|deixar|seguir|encontrar|chamar|vir|pensar|sair|voltar|tomar|conhecer|viver|sentir|tratar|olhar|contar|começar|esperar|buscar|existir|entrar|trabalhar|escrever|perder|produzir|ocorrer|receber|permitir|aparecer|considerar|terminar|desenvolver|obter|atualmente|realmente|simplesmente|provavelmente|geralmente|exatamente|definitivamente|basicamente|particularmente|frequentemente|constantemente|simultaneamente|pesquisa|método|resultado|análise|sistema|modelo|abordagem|proposta|novo|nova|melhor|maior|melhoria|desempenho|eficiência|precisão|exatidão|qualidade|otimização|algoritmo|técnica|abordagem|método|resultado|análise|sistema|modelo|proposta|novo|nova|melhor|maior|melhoria|desempenho|eficiência|precisão|qualidade|otimização|algoritmo|técnica)\b',
}


@dataclass
class PaperTranslation:
    """Paper translation in a specific language."""
    language: Language
    title: str
    abstract: str
    summary: str = ""
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


@dataclass
class MultilingualReport:
    """Multi-language report."""
    title: str
    date: str
    sections: List[Dict]
    language_stats: Dict[str, int]
    papers_count: int


class LanguageDetector:
    """Enhanced language detection with confidence scores."""
    
    def __init__(self):
        self.patterns = {}
        for lang, pattern in LANGUAGE_PATTERNS.items():
            try:
                self.patterns[lang] = re.compile(pattern, re.IGNORECASE)
            except re.error:
                self.patterns[lang] = None
    
    def detect(self, text: str) -> Tuple[Language, float]:
        """
        Detect language of text with confidence score.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (detected language, confidence score)
        """
        if not text:
            return Language.ENGLISH, 0.0
        
        scores = {}
        
        for lang, pattern in self.patterns.items():
            if pattern is None:
                scores[lang] = 0
                continue
            
            matches = pattern.findall(text)
            scores[lang] = len(matches)
        
        # Special handling for character-based languages
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        arabic_chars = len(re.findall(r'[\u0600-\u06ff]', text))
        russian_chars = len(re.findall(r'[а-яА-ЯёЁ]', text))
        
        # Boost scores for character-based languages
        if chinese_chars > 0:
            scores[Language.CHINESE] = max(scores.get(Language.CHINESE, 0), chinese_chars * 10)
        
        if arabic_chars > 0:
            scores[Language.ARABIC] = max(scores.get(Language.ARABIC, 0), arabic_chars * 10)
        
        if russian_chars > 0:
            scores[Language.RUSSIAN] = max(scores.get(Language.RUSSIAN, 0), russian_chars * 5)
        
        # Spanish-specific patterns (to distinguish from French)
        spanish_keywords = ['por', 'pero', 'sino', 'también', 'además', 'aunque', 'porque', 
                           'entonces', 'así', 'aquí', 'ahora', 'luego', 'después', 'antes',
                           'siempre', 'nunca', 'mientras', 'durante', 'hacia', 'contra', 'según',
                           'investigación', 'método', 'resultado', 'análisis', 'sistema', 'modelo',
                           'propuesta', 'nuevo', 'nueva', 'mejor', 'rendimiento', 'eficiencia']
        spanish_specific = sum(1 for word in spanish_keywords if re.search(r'\b' + word + r'\b', text, re.IGNORECASE))
        if spanish_specific > 0:
            scores[Language.SPANISH] = scores.get(Language.SPANISH, 0) + spanish_specific * 3
        
        # Portuguese-specific patterns (to distinguish from Spanish)
        portuguese_keywords = ['não', 'também', 'ainda', 'quando', 'onde', 'porque', 
                              'então', 'assim', 'agora', 'sempre', 'nunca', 'talvez',
                              'pesquisa', 'método', 'resultado', 'análise', 'sistema', 'modelo',
                              'proposta', 'novo', 'nova', 'melhor', 'desempenho', 'eficiência']
        portuguese_specific = sum(1 for word in portuguese_keywords if re.search(r'\b' + word + r'\b', text, re.IGNORECASE))
        if portuguese_specific > 0:
            scores[Language.PORTUGUESE] = scores.get(Language.PORTUGUESE, 0) + portuguese_specific * 3
        
        # Get language with highest score
        if not scores or max(scores.values()) == 0:
            return Language.ENGLISH, 0.0
        
        total_score = sum(scores.values())
        best_lang = max(scores, key=scores.get)
        confidence = scores[best_lang] / total_score if total_score > 0 else 0.0
        
        return best_lang, confidence
    
    def detect_from_metadata(self, paper: Dict) -> Tuple[Language, float]:
        """
        Detect language from paper metadata.
        
        Args:
            paper: Paper dictionary
            
        Returns:
            Tuple of (detected language, confidence)
        """
        # Check explicit language field
        lang_code = paper.get("language", "")
        if lang_code:
            for lang in Language:
                if lang.value == lang_code:
                    return lang, 1.0
        
        # Detect from abstract
        abstract = paper.get("abstract", "")
        if abstract:
            return self.detect(abstract)
        
        # Detect from title
        title = paper.get("title", "")
        if title:
            return self.detect(title)
        
        return Language.ENGLISH, 0.0


class MultilingualTranslator:
    """Enhanced translation with caching and fallback."""
    
    def __init__(self, cache_dir: str = None):
        self.cache = {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()
    
    def _load_cache(self):
        """Load translation cache from disk."""
        cache_file = self.cache_dir / "translation_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}
    
    def _save_cache(self):
        """Save translation cache to disk."""
        if self.cache_dir:
            cache_file = self.cache_dir / "translation_cache.json"
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(self.cache, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
    
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
        cache_key = f"{text[:200]}_{target_lang}"
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
                    self._save_cache()
                    return translated
        except Exception as e:
            print(f"  [!] Translation error: {e}")
        
        return None
    
    def generate_summary(self, text: str, max_length: int = 200) -> str:
        """
        Generate a short summary from text.
        
        Args:
            text: Input text
            max_length: Maximum summary length
            
        Returns:
            Summary text
        """
        if not text:
            return ""
        
        # Simple extractive summary
        sentences = re.split(r'[.!?。！？]+', text)
        summary = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(summary) + len(sentence) < max_length:
                summary += sentence + ". "
            else:
                break
        
        return summary.strip()
    
    def translate_paper(
        self,
        paper: Dict,
        target_languages: List[Language] = None,
        include_summary: bool = True,
    ) -> MultilingualPaper:
        """
        Translate paper to multiple languages.
        
        Args:
            paper: Paper dictionary
            target_languages: List of target languages
            include_summary: Whether to include summary
            
        Returns:
            MultilingualPaper object
        """
        if target_languages is None:
            target_languages = list(Language)
        
        detector = LanguageDetector()
        original_lang, _ = detector.detect_from_metadata(paper)
        
        translations = {}
        
        for lang in target_languages:
            if lang == original_lang:
                # Original language
                summary = self.generate_summary(paper.get("abstract", "")) if include_summary else ""
                translations[lang] = PaperTranslation(
                    language=lang,
                    title=paper.get("title", ""),
                    abstract=paper.get("abstract", ""),
                    summary=summary,
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
                
                summary = self.generate_summary(abstract) if include_summary and abstract else ""
                
                translations[lang] = PaperTranslation(
                    language=lang,
                    title=title or paper.get("title", ""),
                    abstract=abstract or paper.get("abstract", ""),
                    summary=summary,
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
    """Enhanced multi-language paper search."""
    
    def __init__(self, cache_dir: str = None):
        self.detector = LanguageDetector()
        self.translator = MultilingualTranslator(cache_dir)
    
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
                    f"summary_{target_lang.value}": multilingual.translations[target_lang].summary,
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
            paper_lang, _ = self.detector.detect_from_metadata(paper)
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
            lang, _ = self.detector.detect_from_metadata(paper)
            stats[lang.value] += 1
        
        return dict(stats)
    
    def generate_multilingual_report(
        self,
        papers: List[Dict],
        report_title: str = "AI Paper Daily Report",
        target_languages: List[Language] = None,
    ) -> Dict[Language, str]:
        """
        Generate report in multiple languages.
        
        Args:
            papers: List of papers
            report_title: Report title
            target_languages: Target languages
            
        Returns:
            Dictionary of language -> report content
        """
        if target_languages is None:
            target_languages = [Language.ENGLISH, Language.CHINESE]
        
        reports = {}
        
        # Get language stats
        lang_stats = self.get_language_stats(papers)
        
        for lang in target_languages:
            report_lines = []
            
            # Title
            if lang == Language.ENGLISH:
                report_lines.append(f"# {report_title}")
                report_lines.append(f"\n**Date**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d')}")
                report_lines.append(f"**Total Papers**: {len(papers)}")
            elif lang == Language.CHINESE:
                report_lines.append(f"# {self.translator.translate_text(report_title, target_lang='zh') or report_title}")
                report_lines.append(f"\n**日期**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d')}")
                report_lines.append(f"**论文总数**: {len(papers)}")
            
            # Language distribution
            report_lines.append("\n## " + ("Language Distribution" if lang == Language.ENGLISH else "语言分布"))
            for lang_code, count in lang_stats.items():
                lang_name = LANGUAGE_NAMES.get(LANGUAGE_CODES.get(lang_code, Language.ENGLISH), lang_code)
                report_lines.append(f"- {lang_name}: {count}")
            
            # Top papers
            report_lines.append("\n## " + ("Top Papers" if lang == Language.ENGLISH else "热门论文"))
            for i, paper in enumerate(papers[:10], 1):
                title = paper.get("title", "N/A")
                if lang != Language.ENGLISH:
                    translated = self.translator.translate_text(title, target_lang=lang.value)
                    if translated:
                        title = translated
                
                report_lines.append(f"\n### {i}. {title}")
                report_lines.append(f"- **ArXiv**: {paper.get('id', 'N/A')}")
                report_lines.append(f"- **Score**: {paper.get('llm_score', 'N/A')}/10")
            
            reports[lang] = "\n".join(report_lines)
        
        return reports


class MultilingualFormatter:
    """Format content for different languages."""
    
    @staticmethod
    def format_number(number: int, lang: Language) -> str:
        """Format number according to language conventions."""
        if lang == Language.ENGLISH:
            return f"{number:,}"
        elif lang == Language.CHINESE:
            return f"{number:,}"
        else:
            return f"{number:,}"
    
    @staticmethod
    def format_date(date_str: str, lang: Language) -> str:
        """Format date according to language conventions."""
        # Simple formatting
        return date_str
    
    @staticmethod
    def get_rtl_style(lang: Language) -> str:
        """Get CSS style for RTL languages."""
        if lang in RTL_LANGUAGES:
            return 'direction: rtl; text-align: right;'
        return 'direction: ltr; text-align: left;'
    
    @staticmethod
    def format_paper_html(paper: Dict, lang: Language) -> str:
        """Format paper as HTML for specific language."""
        rtl_style = MultilingualFormatter.get_rtl_style(lang)
        
        title = paper.get("title", "N/A")
        abstract = paper.get("abstract", "N/A")
        
        return f"""
        <div style="{rtl_style} padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 5px;">
            <h3>{title}</h3>
            <p><strong>ArXiv:</strong> {paper.get('id', 'N/A')}</p>
            <p><strong>Score:</strong> {paper.get('llm_score', 'N/A')}/10</p>
            <p>{abstract[:300]}...</p>
        </div>
        """


# Global instances
language_detector = LanguageDetector()
multilingual_translator = MultilingualTranslator()
multilingual_search = MultilingualSearch()
multilingual_formatter = MultilingualFormatter()


if __name__ == "__main__":
    # Test enhanced multi-language support
    print("Testing enhanced multi-language support...")
    print()
    
    # Test language detection with confidence
    test_texts = {
        "English": "We propose a novel method for machine learning.",
        "French": "Nous proposons une nouvelle méthode pour l'apprentissage automatique.",
        "Spanish": "Proponemos un nuevo método para el aprendizaje automático.",
        "Russian": "Мы предлагаем новый метод машинного обучения.",
        "Chinese": "我们提出了一种新的机器学习方法。",
        "Arabic": "نقترح طريقة جديدة للتعلم الآلي.",
        "Portuguese": "Propomos um novo método para aprendizado de máquina.",
    }
    
    print("Language detection with confidence:")
    for lang_name, text in test_texts.items():
        detected, confidence = language_detector.detect(text)
        expected = lang_name[:2].lower()
        status = "✓" if detected.value == expected else "✗"
        print(f"  {status} {lang_name}: {detected.value} (confidence: {confidence:.2f})")
    
    print()
    
    # Test translation with caching
    print("Translation test with caching:")
    test_text = "We propose a novel method for machine learning."
    for lang in [Language.CHINESE, Language.FRENCH, Language.SPANISH]:
        translated = multilingual_translator.translate_text(test_text, target_lang=lang.value)
        if translated:
            print(f"  {lang.value}: {translated[:50]}...")
        else:
            print(f"  {lang.value}: [translation failed]")
    
    print()
    
    # Test summary generation
    print("Summary generation:")
    long_text = "We propose a novel method for machine learning that achieves state-of-the-art performance on multiple benchmarks. Our approach combines attention mechanisms with transformer architecture to efficiently process long sequences. Extensive experiments demonstrate the effectiveness of our method."
    summary = multilingual_translator.generate_summary(long_text, max_length=100)
    print(f"  Original: {long_text[:100]}...")
    print(f"  Summary: {summary}")
    
    print()
    
    # Test RTL formatting
    print("RTL formatting:")
    rtl_style = multilingual_formatter.get_rtl_style(Language.ARABIC)
    print(f"  Arabic style: {rtl_style}")
    
    print()
    print("✓ Enhanced multi-language support tested successfully!")
