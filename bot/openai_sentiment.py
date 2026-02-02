"""
OpenAI-powered sentiment analysis for forex news.

Uses GPT-4o-mini for accurate, context-aware sentiment analysis.
Much better than keyword matching for nuanced headlines like:
- "Fed pauses hikes despite strong data" (dovish, not hawkish)
- "Dollar falls on profit-taking" (temporary, not bearish trend)
"""
import os
import logging
import hashlib
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone
from dataclasses import dataclass

LOG = logging.getLogger("bot.openai_sentiment")


@dataclass
class SentimentResult:
    """Result from OpenAI sentiment analysis."""
    headline: str
    sentiment: float  # -1 (very bearish) to +1 (very bullish)
    confidence: float  # 0-1 how confident the model is
    currencies: List[str]  # Currencies mentioned
    reasoning: str  # Brief explanation
    cached: bool = False


class OpenAISentimentAnalyzer:
    """
    Analyze forex news sentiment using OpenAI GPT-4o-mini.

    Features:
    - Batch processing to reduce API calls
    - Caching to avoid re-analyzing same headlines
    - Specific forex/currency focus
    - Returns sentiment per currency mentioned
    """

    # Cache results to avoid re-analyzing (headline hash -> result)
    _cache: Dict[str, SentimentResult] = {}
    _cache_max_size = 500

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None
        self._initialized = False
        self._disabled = False  # Set to True if quota exceeded
        self._last_request_time = 0.0  # Rate limiting
        self._min_request_interval = 1.0  # 1 second between requests (Tier 1 = 3 RPM)

    def _init_client(self) -> bool:
        """Initialize OpenAI client."""
        if self._initialized:
            return self._client is not None

        self._initialized = True

        if not self.api_key:
            LOG.warning("OpenAI API key not configured - falling back to keyword matching")
            return False

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
            LOG.info("OpenAI sentiment analyzer initialized")
            return True
        except ImportError:
            LOG.warning("openai package not installed - run: pip install openai")
            return False
        except Exception as e:
            LOG.warning("Failed to initialize OpenAI: %s", e)
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
        # Skip if disabled due to quota issues
        if self._disabled:
            return SentimentResult(
                headline=headline,
                sentiment=0.0,
                confidence=0.0,
                currencies=[],
                reasoning="OpenAI disabled (quota)",
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
                reasoning="OpenAI not available",
            )

        # Rate limiting - wait if needed to avoid 429 errors
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

        try:
            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a forex market sentiment analyzer. Analyze news headlines for their impact on currency values.

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
- Political stability, trade relations, etc."""
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this forex headline:\n\n{headline}"
                    }
                ],
                max_tokens=100,
                temperature=0.1,  # Low temp for consistent analysis
            )

            response_text = response.choices[0].message.content
            result = self._parse_response(response_text, headline)

            # Cache the result
            if len(self._cache) >= self._cache_max_size:
                # Remove oldest entries (first 100)
                keys_to_remove = list(self._cache.keys())[:100]
                for key in keys_to_remove:
                    del self._cache[key]

            self._cache[cache_key] = result

            LOG.debug("Analyzed: '%s' -> %.2f (%s)",
                     headline[:50], result.sentiment, result.currencies)

            return result

        except Exception as e:
            error_str = str(e)
            LOG.warning("OpenAI analysis failed: %s", e)

            # Check for quota/billing errors - disable to avoid repeated failures
            if "insufficient_quota" in error_str or "exceeded your current quota" in error_str:
                LOG.warning("OpenAI quota exceeded - disabling OpenAI sentiment for this session")
                LOG.warning("Add billing at https://platform.openai.com/settings/organization/billing")
                self._disabled = True

            return SentimentResult(
                headline=headline,
                sentiment=0.0,
                confidence=0.0,
                currencies=[],
                reasoning=f"API error: {str(e)[:30]}",
            )

    def analyze_headlines_batch(self, headlines: List[str]) -> List[SentimentResult]:
        """
        Analyze multiple headlines efficiently.

        Uses caching to avoid redundant API calls.
        """
        results = []
        uncached = []
        uncached_indices = []

        # Check cache for each headline
        for i, headline in enumerate(headlines):
            cache_key = self._get_cache_key(headline)
            if cache_key in self._cache:
                result = self._cache[cache_key]
                result.cached = True
                results.append((i, result))
            else:
                uncached.append(headline)
                uncached_indices.append(i)

        # Analyze uncached headlines
        for headline, idx in zip(uncached, uncached_indices):
            result = self.analyze_headline(headline)
            results.append((idx, result))

        # Sort by original index and return
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def get_currency_sentiment(self, headlines: List[str], currency: str) -> Tuple[float, int, float]:
        """
        Get aggregate sentiment for a specific currency from headlines.

        Args:
            headlines: List of news headlines
            currency: Currency code (e.g., "USD", "EUR")

        Returns:
            (average_sentiment, article_count, average_confidence)
        """
        results = self.analyze_headlines_batch(headlines)

        relevant = [r for r in results if currency in r.currencies]

        if not relevant:
            return 0.0, 0, 0.0

        # Weight by confidence
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

        Args:
            headlines: List of news headlines
            symbol: Currency pair (e.g., "EURUSD")

        Returns:
            (sentiment_score, recommendation)
            - Positive = bullish on the pair (buy)
            - Negative = bearish on the pair (sell)
        """
        base = symbol[:3]  # EUR in EURUSD
        quote = symbol[3:]  # USD in EURUSD

        base_sent, base_count, base_conf = self.get_currency_sentiment(headlines, base)
        quote_sent, quote_count, quote_conf = self.get_currency_sentiment(headlines, quote)

        # Pair sentiment = base sentiment - quote sentiment
        # If EUR is bullish (+0.5) and USD is bearish (-0.3), EURUSD sentiment = +0.8
        pair_sentiment = base_sent - quote_sent

        # Determine recommendation
        if pair_sentiment > 0.2:
            recommendation = "bullish"
        elif pair_sentiment < -0.2:
            recommendation = "bearish"
        else:
            recommendation = "neutral"

        LOG.info("OpenAI Sentiment for %s: base(%s)=%.2f quote(%s)=%.2f -> pair=%.2f (%s)",
                 symbol, base, base_sent, quote, quote_sent, pair_sentiment, recommendation)

        return pair_sentiment, recommendation

    def log_analysis_report(self, headlines: List[str], symbol: str):
        """Log detailed sentiment analysis report."""
        results = self.analyze_headlines_batch(headlines)

        base = symbol[:3]
        quote = symbol[3:]

        LOG.info("=" * 60)
        LOG.info("OPENAI SENTIMENT ANALYSIS: %s", symbol)
        LOG.info("=" * 60)

        relevant = [r for r in results if base in r.currencies or quote in r.currencies]

        for r in relevant[:5]:  # Top 5 relevant
            cached_str = " (cached)" if r.cached else ""
            LOG.info("  [%.2f] %s%s", r.sentiment, r.headline[:60], cached_str)
            LOG.info("         -> %s | %s", r.currencies, r.reasoning)

        pair_sent, rec = self.get_pair_sentiment(headlines, symbol)
        LOG.info("-" * 60)
        LOG.info("Pair Sentiment: %.2f -> %s", pair_sent, rec.upper())
        LOG.info("=" * 60)


# Singleton instance
_analyzer: Optional[OpenAISentimentAnalyzer] = None


def get_openai_analyzer() -> OpenAISentimentAnalyzer:
    """Get or create the singleton OpenAI analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = OpenAISentimentAnalyzer()
    return _analyzer
