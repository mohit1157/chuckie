"""
Free News Feeds - RSS-based financial news without API limits.

Replaces Twitter API with free RSS feeds from:
1. ForexLive - Real-time forex news
2. FXStreet - Currency news and analysis
3. Investing.com - Market news
4. Reuters - Business news
5. Bloomberg (via Google News) - Financial headlines

No API keys required. No rate limits.
"""
import logging
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import urllib.request
import urllib.error

LOG = logging.getLogger("bot.free_news")


@dataclass
class NewsItem:
    """A news item from RSS feed."""
    title: str
    source: str
    url: str
    published: datetime
    sentiment: float  # -1 to +1
    currencies_affected: List[str]
    impact: str  # "high", "medium", "low"


class FreeNewsFetcher:
    """
    Fetch financial news from free RSS feeds.

    No API keys needed. No rate limits.
    """

    # Free RSS feeds for forex/financial news
    RSS_FEEDS = {
        "forexlive": {
            "url": "https://www.forexlive.com/feed",
            "name": "ForexLive",
        },
        "fxstreet": {
            "url": "https://www.fxstreet.com/rss/news",
            "name": "FXStreet",
        },
        "dailyfx": {
            "url": "https://www.dailyfx.com/feeds/market-news",
            "name": "DailyFX",
        },
        "investing_forex": {
            "url": "https://www.investing.com/rss/news_285.rss",
            "name": "Investing.com Forex",
        },
        "investing_economy": {
            "url": "https://www.investing.com/rss/news_95.rss",
            "name": "Investing.com Economy",
        },
    }

    # Backup: Google News search for financial terms
    GOOGLE_NEWS_SEARCHES = [
        "forex+market",
        "federal+reserve",
        "ECB+interest+rate",
        "USD+EUR+exchange",
    ]

    # Sentiment analysis keywords - comprehensive forex/economic terms
    BULLISH_WORDS = [
        # Price action
        "surge", "soar", "jump", "rally", "gain", "rise", "climb", "spike",
        "breakout", "advance", "rebound", "bounce", "recover", "uptick",
        # Sentiment
        "bullish", "strong", "robust", "solid", "optimism", "optimistic",
        "confident", "positive", "upbeat", "encouraging", "promising",
        # Economic
        "growth", "expansion", "boom", "acceleration", "improving",
        "beat", "exceed", "surpass", "outperform", "better than expected",
        # Central bank / Policy
        "hawkish", "tightening", "rate hike", "hike", "restrictive",
        # Technical
        "support", "higher", "peak", "highs", "uptrend", "momentum",
        "buy", "long", "accumulation", "demand",
    ]

    BEARISH_WORDS = [
        # Price action
        "drop", "fall", "plunge", "crash", "sink", "decline", "tumble",
        "slide", "slump", "collapse", "plummet", "downturn", "selloff",
        "breakdown", "retreat", "downtick",
        # Sentiment
        "bearish", "weak", "soft", "fragile", "pessimism", "pessimistic",
        "concern", "worried", "fear", "anxiety", "uncertainty", "risk-off",
        # Economic
        "recession", "contraction", "slowdown", "stagnation", "deteriorating",
        "miss", "disappoint", "below expectations", "worse than expected",
        # Central bank / Policy
        "dovish", "easing", "rate cut", "cut", "accommodative", "stimulus",
        # Technical
        "resistance", "lower", "bottom", "lows", "downtrend", "selling",
        "sell", "short", "distribution", "supply",
        # Crisis
        "crisis", "emergency", "default", "warning", "alert",
    ]

    # High impact keywords
    HIGH_IMPACT_KEYWORDS = [
        "breaking", "just in", "fed", "fomc", "powell", "rate decision",
        "ecb", "lagarde", "boe", "inflation", "cpi", "nfp", "payroll",
        "gdp", "recession", "crisis", "emergency", "surprise",
        "trump", "tariff", "trade war", "sanctions",
    ]

    # Currency detection
    CURRENCY_PATTERNS = {
        "USD": ["usd", "dollar", "greenback", "fed", "fomc", "powell", "us economy", "american"],
        "EUR": ["eur", "euro", "ecb", "lagarde", "eurozone", "german", "france"],
        "GBP": ["gbp", "pound", "sterling", "boe", "bank of england", "uk", "british"],
        "JPY": ["jpy", "yen", "boj", "japan", "kuroda", "ueda"],
        "AUD": ["aud", "aussie", "rba", "australia"],
        "CAD": ["cad", "loonie", "boc", "canada", "canadian"],
        "CHF": ["chf", "franc", "snb", "swiss"],
        "NZD": ["nzd", "kiwi", "rbnz", "new zealand"],
        "CNY": ["cny", "yuan", "china", "pboc", "chinese"],
        "XAU": ["gold", "xau", "precious metal"],
        "OIL": ["oil", "crude", "wti", "brent", "opec"],
    }

    def __init__(self):
        self._cache: List[NewsItem] = []
        self._cache_time: Optional[datetime] = None
        self._cache_duration = 600  # 10 minutes

    def _parse_date(self, date_str: str) -> datetime:
        """Parse various RSS date formats."""
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S GMT",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue

        return datetime.now(timezone.utc)

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text."""
        text_lower = text.lower()

        bullish = sum(1 for w in self.BULLISH_WORDS if w in text_lower)
        bearish = sum(1 for w in self.BEARISH_WORDS if w in text_lower)

        total = bullish + bearish
        if total == 0:
            return 0.0

        return (bullish - bearish) / total

    def _detect_currencies(self, text: str) -> List[str]:
        """Detect currencies mentioned in text."""
        text_lower = text.lower()
        currencies = []

        for currency, patterns in self.CURRENCY_PATTERNS.items():
            if any(p in text_lower for p in patterns):
                currencies.append(currency)

        return currencies if currencies else ["USD"]

    def _assess_impact(self, text: str) -> str:
        """Assess market impact of news."""
        text_lower = text.lower()

        high_count = sum(1 for kw in self.HIGH_IMPACT_KEYWORDS if kw in text_lower)

        if high_count >= 2 or any(kw in text_lower for kw in ["breaking", "fomc", "rate decision", "nfp"]):
            return "high"
        elif high_count >= 1:
            return "medium"
        return "low"

    def _fetch_rss(self, url: str, source_name: str) -> List[NewsItem]:
        """Fetch and parse RSS feed."""
        items = []

        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "Mozilla/5.0 (compatible; ForexBot/1.0)")

            with urllib.request.urlopen(req, timeout=10) as response:
                content = response.read().decode("utf-8", errors="ignore")

            # Parse XML
            root = ET.fromstring(content)

            # Handle different RSS formats
            for item in root.findall(".//item"):
                title_elem = item.find("title")
                link_elem = item.find("link")
                pub_elem = item.find("pubDate")

                if title_elem is None or title_elem.text is None:
                    continue

                title = title_elem.text.strip()
                url = link_elem.text.strip() if link_elem is not None and link_elem.text else ""
                pub_date = self._parse_date(pub_elem.text) if pub_elem is not None and pub_elem.text else datetime.now(timezone.utc)

                items.append(NewsItem(
                    title=title,
                    source=source_name,
                    url=url,
                    published=pub_date,
                    sentiment=self._analyze_sentiment(title),
                    currencies_affected=self._detect_currencies(title),
                    impact=self._assess_impact(title),
                ))

            LOG.debug("Fetched %d items from %s", len(items), source_name)

        except ET.ParseError as e:
            LOG.warning("XML parse error for %s: %s", source_name, e)
        except urllib.error.URLError as e:
            LOG.warning("URL error for %s: %s", source_name, e)
        except Exception as e:
            LOG.warning("Error fetching %s: %s", source_name, e)

        return items

    def fetch_all_news(self, max_age_hours: int = 24) -> List[NewsItem]:
        """Fetch news from all RSS feeds."""
        now = datetime.now(timezone.utc)

        # Check cache
        if self._cache_time and (now - self._cache_time).total_seconds() < self._cache_duration:
            return self._cache

        all_items = []

        # Fetch from all RSS feeds
        for feed_id, feed_info in self.RSS_FEEDS.items():
            items = self._fetch_rss(feed_info["url"], feed_info["name"])
            all_items.extend(items)

        # Filter by age
        cutoff = now - timedelta(hours=max_age_hours)
        all_items = [item for item in all_items if item.published > cutoff]

        # Sort by published date (newest first)
        all_items.sort(key=lambda x: x.published, reverse=True)

        # Deduplicate by title similarity
        seen_titles = set()
        unique_items = []
        for item in all_items:
            # Simple dedup: first 50 chars of lowercase title
            key = item.title.lower()[:50]
            if key not in seen_titles:
                seen_titles.add(key)
                unique_items.append(item)

        self._cache = unique_items
        self._cache_time = now

        LOG.info("Fetched %d unique news items from %d RSS feeds",
                len(unique_items), len(self.RSS_FEEDS))

        return unique_items

    def get_breaking_news(self, minutes: int = 30) -> List[NewsItem]:
        """Get high-impact news from the last N minutes."""
        all_news = self.fetch_all_news(max_age_hours=1)
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)

        breaking = [
            item for item in all_news
            if item.published > cutoff and item.impact == "high"
        ]

        return breaking

    def get_sentiment_for_pair(self, symbol: str) -> Dict:
        """
        Get sentiment for a currency pair from RSS news.

        Returns:
            {
                "base_sentiment": float,
                "quote_sentiment": float,
                "overall_bias": str,
                "news_count": int,
                "breaking_news": bool,
                "top_headlines": List[str],
            }
        """
        base = symbol[:3].upper()
        quote = symbol[3:6].upper()

        all_news = self.fetch_all_news(max_age_hours=12)

        base_news = [n for n in all_news if base in n.currencies_affected]
        quote_news = [n for n in all_news if quote in n.currencies_affected]

        base_sentiment = sum(n.sentiment for n in base_news) / len(base_news) if base_news else 0
        quote_sentiment = sum(n.sentiment for n in quote_news) / len(quote_news) if quote_news else 0

        # For pair: bullish base or bearish quote = bullish pair
        overall = base_sentiment - quote_sentiment

        if overall > 0.15:
            bias = "bullish"
        elif overall < -0.15:
            bias = "bearish"
        else:
            bias = "neutral"

        # Check for breaking news
        breaking = self.get_breaking_news(minutes=30)
        has_breaking = any(
            base in n.currencies_affected or quote in n.currencies_affected
            for n in breaking
        )

        # Get top headlines
        relevant_news = [n for n in all_news if base in n.currencies_affected or quote in n.currencies_affected]
        top_headlines = [n.title for n in relevant_news[:5]]

        return {
            "base_sentiment": base_sentiment,
            "quote_sentiment": quote_sentiment,
            "overall_bias": bias,
            "news_count": len(base_news) + len(quote_news),
            "breaking_news": has_breaking,
            "top_headlines": top_headlines,
        }

    def should_block_trading(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if trading should be blocked due to breaking news.

        Returns:
            (should_block, reason)
        """
        breaking = self.get_breaking_news(minutes=15)

        base = symbol[:3].upper()
        quote = symbol[3:6].upper()

        for news in breaking:
            if base in news.currencies_affected or quote in news.currencies_affected:
                return True, f"Breaking news: {news.title[:80]}..."

        return False, ""

    def log_news_report(self, symbol: str = "EURUSD"):
        """Log current news report."""
        sentiment = self.get_sentiment_for_pair(symbol)
        breaking = self.get_breaking_news(minutes=60)

        LOG.info("=" * 60)
        LOG.info("RSS NEWS REPORT: %s", symbol)
        LOG.info("=" * 60)
        LOG.info("Base (%s) sentiment: %.2f", symbol[:3], sentiment["base_sentiment"])
        LOG.info("Quote (%s) sentiment: %.2f", symbol[3:6], sentiment["quote_sentiment"])
        LOG.info("Overall bias: %s", sentiment["overall_bias"].upper())
        LOG.info("News articles: %d", sentiment["news_count"])
        LOG.info("-" * 60)

        if sentiment["top_headlines"]:
            LOG.info("Top Headlines:")
            for headline in sentiment["top_headlines"][:3]:
                LOG.info("  - %s", headline[:70])

        if breaking:
            LOG.info("-" * 60)
            LOG.info("BREAKING NEWS (%d items):", len(breaking))
            for news in breaking[:3]:
                LOG.info("  [%s] %s", news.source, news.title[:60])

        LOG.info("=" * 60)


# Singleton instance
_fetcher: Optional[FreeNewsFetcher] = None

def get_free_news() -> FreeNewsFetcher:
    """Get the free news fetcher instance."""
    global _fetcher
    if _fetcher is None:
        _fetcher = FreeNewsFetcher()
    return _fetcher
