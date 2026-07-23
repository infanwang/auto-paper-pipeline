#!/usr/bin/env python3
"""Paper abstract translation module."""

import json
import urllib.request
import urllib.error
from typing import Optional
from anti_crawl import AntiCrawl

anti_crawl = AntiCrawl(min_delay=1, max_delay=3, max_retries=2)


class AbstractTranslator:
    """Translate paper abstracts using free translation APIs."""
    
    def __init__(self):
        self.cache = {}
    
    def translate_with_google(self, text: str, target_lang: str = "zh-CN") -> Optional[str]:
        """
        Translate using Google Translate (unofficial).
        
        Args:
            text: Text to translate
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
            import urllib.parse
            encoded_text = urllib.parse.quote(text[:5000])  # Limit length
            url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl={target_lang}&dt=t&q={encoded_text}"
            
            data = anti_crawl.fetch(url, timeout=10)
            result = json.loads(data.decode("utf-8"))
            
            # Extract translated text
            translated = "".join([item[0] for item in result[0] if item[0]])
            
            # Cache result
            self.cache[cache_key] = translated
            
            return translated
        except Exception as e:
            print(f"  [!] Translation error: {e}")
            return None
    
    def translate_with_mymemory(self, text: str, target_lang: str = "zh-CN") -> Optional[str]:
        """
        Translate using MyMemory API (free tier).
        
        Args:
            text: Text to translate
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
            import urllib.parse
            encoded_text = urllib.parse.quote(text[:500])
            url = f"https://api.mymemory.translated.net/get?q={encoded_text}&langpair=en|{target_lang}"
            
            data = anti_crawl.fetch(url, timeout=10)
            result = json.loads(data.decode("utf-8"))
            
            translated = result.get("responseData", {}).get("translatedText", "")
            
            if translated:
                self.cache[cache_key] = translated
                return translated
        except Exception as e:
            print(f"  [!] MyMemory translation error: {e}")
        
        return None
    
    def translate(self, text: str, target_lang: str = "zh-CN", method: str = "google") -> Optional[str]:
        """
        Translate text.
        
        Args:
            text: Text to translate
            target_lang: Target language code
            method: Translation method (google or mymemory)
            
        Returns:
            Translated text or None
        """
        if method == "google":
            return self.translate_with_google(text, target_lang)
        elif method == "mymemory":
            return self.translate_with_mymemory(text, target_lang)
        else:
            return self.translate_with_google(text, target_lang)
    
    def translate_paper(self, paper: dict, target_lang: str = "zh-CN") -> dict:
        """
        Translate paper abstract and add to paper dict.
        
        Args:
            paper: Paper dictionary
            target_lang: Target language code
            
        Returns:
            Updated paper dictionary
        """
        abstract = paper.get("abstract", "")
        if abstract:
            translated = self.translate(abstract, target_lang)
            if translated:
                paper["abstract_zh"] = translated
        
        return paper
    
    def batch_translate(self, papers: list, target_lang: str = "zh-CN") -> list:
        """
        Batch translate paper abstracts.
        
        Args:
            papers: List of paper dictionaries
            target_lang: Target language code
            
        Returns:
            List of papers with translated abstracts
        """
        translated_papers = []
        
        for i, paper in enumerate(papers):
            print(f"  Translating {i+1}/{len(papers)}: {paper.get('title', 'N/A')[:50]}...")
            translated_paper = self.translate_paper(paper, target_lang)
            translated_papers.append(translated_paper)
        
        return translated_papers


# Global instance
translator = AbstractTranslator()


if __name__ == "__main__":
    # Test translation
    print("Testing abstract translation...")
    
    test_abstract = "We introduce PoTRE, a novel heterogeneous framework that decouples inference into four agents for complex reasoning tasks."
    
    translated = translator.translate(test_abstract)
    print(f"\nOriginal: {test_abstract}")
    print(f"Translated: {translated}")
