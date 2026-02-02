"""
Gemini-powered sentiment analysis for forex news.

Uses Gemini Flash for fast, accurate, context-aware sentiment analysis.
Much cheaper than OpenAI and has generous free tier (15 RPM, 1M tokens/day).
"""
import os
import logging
import hashlib
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

LOG = logging.getLogger("bot.gemini_sentiment")


@dataclass
class SentimentResult:
    """Result from Gemini sentiment analysis."""
    headline: str
    sentiment: float  # -1 (very bearish) to +1 (very bullish)
    confidence: float  # 0-1 how confident the model is
    currencies: List[str]  # Currencies mentioned
    reasoning: str  # Brief explanation
    cached: bool = False


class GeminiSentimentAnalyzer:
    """
    Analyze forex news sentiment using Google Gemini Flash.

    Features:
    - Much cheaper than OpenAI (free tier: 15 RPM, 1M tokens/day)
    - Caching to avoid re-analyzing same headlines
    - Specific forex/currency focus
    - Returns sentiment per currency mentioned
    """

    # Cache results to avoid re-analyzing (headline hash -> result)
    _cache: Dict[str, SentimentResult] = {}
    _cache_max_size = 500

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._model = None
        self._initialized = False
        self._disabled = False
        self._last_request_time = 0.0
        self._min_request_interval = 4.0  # 15 RPM = 4 seconds between requests

    def _init_client(self) -> bool:
        """Initialize Gemini client."""
        if self._initialized:
            return self._model is not None

        self._initialized = True

        if not self.api_key:
            LOG.warning("Gemini API key not configured - falling back to keyword matching")
            return False

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel('gemini-2.0-flash')
            LOG.info("Gemini sentiment analyzer initialized (gemini-2.0-flash)")
            return True
        except ImportError:
            LOG.warning("google-generativeai package not installed - run: pip install google-generativeai")
            return False
        except Exception as e:
            LOG.warning("Failed to initialize Gemini: %s", e)
            return False

    def _get_cache_key(self, headline: str) -> str:
        """Generate cache key for headline."""
        return hashlib.md5(headline.lower().strip().encode()).hexdigest()

    def _parse_response(self, response_text: str, headline: str) -> SentimentResult:
        """Parse the model response into SentimentResult."""
        try:
            lines = response_text.strip().split('\n')
            sentiment = 0.0
            confidence = 0.5
            currencies = []
            reasoning = ""

            for line in lines:
                line = line.strip()
                if line.startswith("SENTIMENT:"):
                    try:
                        sentiment = float(line.split(":")[1].strip())
                        sentiment = max(-1.0, min(1.0, sentiment))
                    except:
                        pass
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":")[1].strip())
                        confidence = max(0.0, min(1.0, confidence))
                    except:
                        pass
                elif line.startswith("CURRENCIES:"):
                    curr_str = line.split(":")[1].strip()
                    currencies = [c.strip().upper() for c in curr_str.split(",") if c.strip()]
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()

            return SentimentResult(
                headline=headline,
                sentiment=sentiment,
                confidence=confidence,
                currencies=currencies,
                reasoning=reasoning,
            )
        except Exception as e:
            LOG.debug("Failed to parse response: %s", e)
            return SentimentResult(
                headline=headline,
                sentiment=0.0,
                confidence=0.0,
                currencies=[],
                reasoning="Parse error",
            )

    def analyze_headline(self, headline: str) -> SentimentResult:
        """
        Analyze a single headline for forex sentiment.

        Returns:
            SentimentResult with sentiment score and metadata
        """
        # Skip if disabled
        if self._disabled:
            return SentimentResult(
                headline=headline,
                sentiment=0.0,
                confidence=0.0,
                currencies=[],
                reasoning="Gemini disabled",
            )

        # Check cache first
        cache_key = self._get_cache_key(headline)
        if cache_key in self._cache:
            result = self._cache[cache_key]
            result.cached = True
            return result

        # Initialize client if needed
        if not self._init_client():
            return SentimentResult(
                headline=headline,
                sentiment=0.0,
                confidence=0.0,
                currencies=[],
                reasoning="Gemini not available",
            )

        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

        prompt = f"""You are a forex market sentiment analyzer. Analyze this news headline for its impact on currency values.

Output format (exactly):
SENTIMENT: <number from -1.0 to 1.0>
CONFIDENCE: <number from 0.0 to 1.0>
CURRENCIES: <comma-separated list like USD, EUR, GBP>
REASONING: <brief 5-10 word explanation>

Sentiment scale:
-1.0 = Very bearish (currency will fall significantly)
-0.5 = Moderately bearish
0.0 = Neutral / no impact
+0.5 = Moderately bullish
+1.0 = Very bullish (currency will rise significantly)

Consider:
- Central bank policy (hawkish = bullish for currency, dovish = bearish)
- Economic data (strong = bullish, weak = bearish)
- Risk sentiment (risk-off = bullish for USD, JPY, CHF)
- Political stability, trade relations, etc.

Headline: {headline}"""

        try:
            response = self._model.generate_content(prompt)
            response_text = response.text
            result = self._parse_response(response_text, headline)

            # Cache the result
            if len(self._cache) >= self._cache_max_size:
                keys_to_remove = list(self._cache.keys())[:100]
                for key in keys_to_remove:
                    del self._cache[key]

            self._cache[cache_key] = result

            LOG.debug("Analyzed: '%s' -> %.2f (%s)",
                     headline[:50], result.sentiment, result.currencies)

            return result

        except Exception as e:
            error_str = str(e)

            if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                LOG.debug("Gemini rate limited, increasing delay")
                self._min_request_interval = min(10.0, self._min_request_interval + 2.0)
            else:
                LOG.warning("Gemini analysis failed: %s", e)

            return SentimentResult(
                headline=headline,
                sentiment=0.0,
                confidence=0.0,
                currencies=[],
                reasoning=f"API error: {str(e)[:30]}",
            )

    def analyze_headlines_batch(self, headlines: List[str]) -> List[SentimentResult]:
        """Analyze multiple headlines efficiently using caching."""
        results = []
        uncached = []
        uncached_indices = []

        for i, headline in enumerate(headlines):
            cache_key = self._get_cache_key(headline)
            if cache_key in self._cache:
                result = self._cache[cache_key]
                result.cached = True
                results.append((i, result))
            else:
                uncached.append(headline)
                uncached_indices.append(i)

        for headline, idx in zip(uncached, uncached_indices):
            result = self.analyze_headline(headline)
            results.append((idx, result))

        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def get_currency_sentiment(self, headlines: List[str], currency: str) -> Tuple[float, int, float]:
        """Get aggregate sentiment for a specific currency."""
        results = self.analyze_headlines_batch(headlines)
        relevant = [r for r in results if currency in r.currencies]

        if not relevant:
            return 0.0, 0, 0.0

        total_weight = sum(r.confidence for r in relevant)
        if total_weight == 0:
            avg_sentiment = sum(r.sentiment for r in relevant) / len(relevant)
            avg_confidence = 0.5
        else:
            avg_sentiment = sum(r.sentiment * r.confidence for r in relevant) / total_weight
            avg_confidence = total_weight / len(relevant)

        return avg_sentiment, len(relevant), avg_confidence

    def get_pair_sentiment(self, headlines: List[str], symbol: str) -> Tuple[float, str]:
        """
        Get sentiment for a currency pair.

        Returns:
            (sentiment_score, recommendation)
        """
        base = symbol[:3]
        quote = symbol[3:]

        base_sent, base_count, base_conf = self.get_currency_sentiment(headlines, base)
        quote_sent, quote_count, quote_conf = self.get_currency_sentiment(headlines, quote)

        pair_sentiment = base_sent - quote_sent

        if pair_sentiment > 0.2:
            recommendation = "bullish"
        elif pair_sentiment < -0.2:
            recommendation = "bearish"
        else:
            recommendation = "neutral"

        LOG.info("Gemini Sentiment for %s: base(%s)=%.2f quote(%s)=%.2f -> pair=%.2f (%s)",
                 symbol, base, base_sent, quote, quote_sent, pair_sentiment, recommendation)

        return pair_sentiment, recommendation

    def log_analysis_report(self, headlines: List[str], symbol: str):
        """Log detailed sentiment analysis report."""
        results = self.analyze_headlines_batch(headlines)
        base = symbol[:3]
        quote = symbol[3:]

        LOG.info("=" * 60)
        LOG.info("GEMINI SENTIMENT ANALYSIS: %s", symbol)
        LOG.info("=" * 60)

        relevant = [r for r in results if base in r.currencies or quote in r.currencies]

        for r in relevant[:5]:
            cached_str = " (cached)" if r.cached else ""
            LOG.info("  [%.2f] %s%s", r.sentiment, r.headline[:60], cached_str)
            LOG.info("         -> %s | %s", r.currencies, r.reasoning)

        pair_sent, rec = self.get_pair_sentiment(headlines, symbol)
        LOG.info("-" * 60)
        LOG.info("Pair Sentiment: %.2f -> %s", pair_sent, rec.upper())
        LOG.info("=" * 60)


# Singleton instance
_analyzer: Optional[GeminiSentimentAnalyzer] = None


def get_gemini_analyzer() -> GeminiSentimentAnalyzer:
    """Get or create the singleton Gemini analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = GeminiSentimentAnalyzer()
    return _analyzer
