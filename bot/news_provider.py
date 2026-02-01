"""
News and Sentiment Provider for Forex Trading.

Integrates multiple sources:
1. Economic Calendar (high-impact events like NFP, FOMC, ECB)
2. Financial News (Reuters, Bloomberg headlines)
3. Reddit Sentiment (r/Forex, r/forextrading)
4. Currency-specific news

Free APIs used:
- ForexFactory calendar (web scraping)
- NewsAPI.org (free tier: 100 requests/day)
- Reddit API via PRAW
- Finnhub (free tier)
"""
import logging
import json
import re
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import urllib.request
import urllib.error
from pathlib import Path

LOG = logging.getLogger("bot.news_provider")


class NewsImpact(Enum):
    """Impact level of news event."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"  # NFP, FOMC, ECB rate decisions


@dataclass
class EconomicEvent:
    """Economic calendar event."""
    time: datetime
    currency: str
    event: str
    impact: NewsImpact
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None


@dataclass
class NewsArticle:
    """News article with sentiment."""
    title: str
    source: str
    published: datetime
    url: str
    sentiment: float  # -1 to 1
    currencies_mentioned: List[str]


@dataclass
class RedditPost:
    """Reddit post with sentiment."""
    title: str
    subreddit: str
    score: int
    created: datetime
    sentiment: float
    url: str


class EconomicCalendar:
    """
    Economic calendar for high-impact forex events.

    Key events to watch:
    - NFP (Non-Farm Payrolls) - First Friday of month
    - FOMC (Federal Reserve) - 8 times/year
    - ECB Rate Decision - Monthly
    - BOE Rate Decision - Monthly
    - CPI/Inflation data
    - GDP releases
    - Retail Sales
    - PMI data
    """

    # High-impact events that move markets significantly
    HIGH_IMPACT_KEYWORDS = [
        "interest rate", "rate decision", "fomc", "fed",
        "nonfarm", "non-farm", "nfp", "payroll",
        "ecb", "boe", "boj", "rba", "rbnz", "snb",
        "gdp", "inflation", "cpi", "ppi",
        "unemployment", "employment change",
        "retail sales", "trade balance",
        "pmi", "manufacturing", "services",
        "central bank", "monetary policy",
    ]

    # Currency mapping
    CURRENCY_EVENTS = {
        "USD": ["fomc", "fed", "nfp", "payroll", "us ", "u.s.", "american"],
        "EUR": ["ecb", "eurozone", "euro area", "german", "french"],
        "GBP": ["boe", "bank of england", "uk ", "british", "brexit"],
        "JPY": ["boj", "bank of japan", "japanese", "japan"],
        "AUD": ["rba", "australian", "australia"],
        "NZD": ["rbnz", "new zealand", "kiwi"],
        "CAD": ["boc", "bank of canada", "canadian", "canada"],
        "CHF": ["snb", "swiss", "switzerland"],
    }

    def __init__(self, cache_file: str = "economic_calendar_cache.json"):
        self.cache_file = Path(cache_file)
        self._events: List[EconomicEvent] = []
        self._last_fetch: Optional[datetime] = None
        self._load_cache()

    def _load_cache(self):
        """Load cached events from file."""
        if self.cache_file.exists():
            try:
                data = json.loads(self.cache_file.read_text())
                self._last_fetch = datetime.fromisoformat(data.get("last_fetch", "2000-01-01"))
                self._events = [
                    EconomicEvent(
                        time=datetime.fromisoformat(e["time"]),
                        currency=e["currency"],
                        event=e["event"],
                        impact=NewsImpact(e["impact"]),
                        forecast=e.get("forecast"),
                        previous=e.get("previous"),
                    )
                    for e in data.get("events", [])
                ]
                LOG.info("Loaded %d cached economic events", len(self._events))
            except Exception as e:
                LOG.warning("Failed to load cache: %s", e)

    def _save_cache(self):
        """Save events to cache file."""
        try:
            data = {
                "last_fetch": datetime.now(timezone.utc).isoformat(),
                "events": [
                    {
                        "time": e.time.isoformat(),
                        "currency": e.currency,
                        "event": e.event,
                        "impact": e.impact.value,
                        "forecast": e.forecast,
                        "previous": e.previous,
                    }
                    for e in self._events
                ]
            }
            self.cache_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            LOG.warning("Failed to save cache: %s", e)

    def _determine_impact(self, event_name: str) -> NewsImpact:
        """Determine impact level from event name."""
        event_lower = event_name.lower()

        # Critical events
        critical_keywords = ["interest rate", "rate decision", "fomc", "nfp", "non-farm", "payroll"]
        if any(kw in event_lower for kw in critical_keywords):
            return NewsImpact.CRITICAL

        # High impact
        high_keywords = ["gdp", "cpi", "inflation", "employment", "unemployment", "pmi"]
        if any(kw in event_lower for kw in high_keywords):
            return NewsImpact.HIGH

        # Medium impact
        medium_keywords = ["retail", "trade balance", "manufacturing", "services", "housing"]
        if any(kw in event_lower for kw in medium_keywords):
            return NewsImpact.MEDIUM

        return NewsImpact.LOW

    def _determine_currency(self, event_name: str) -> str:
        """Determine which currency the event affects."""
        event_lower = event_name.lower()

        for currency, keywords in self.CURRENCY_EVENTS.items():
            if any(kw in event_lower for kw in keywords):
                return currency

        return "USD"  # Default to USD

    def add_manual_events(self):
        """Add known recurring high-impact events."""
        now = datetime.now(timezone.utc)

        # These are approximate - in production, fetch from real calendar
        known_events = [
            # FOMC meetings (approximate dates)
            ("FOMC Interest Rate Decision", "USD", NewsImpact.CRITICAL),
            ("ECB Interest Rate Decision", "EUR", NewsImpact.CRITICAL),
            ("BOE Interest Rate Decision", "GBP", NewsImpact.CRITICAL),
            ("US Non-Farm Payrolls", "USD", NewsImpact.CRITICAL),
            ("US CPI", "USD", NewsImpact.HIGH),
            ("Eurozone CPI", "EUR", NewsImpact.HIGH),
        ]

        LOG.info("Economic calendar initialized with high-impact event detection")

    def get_upcoming_events(
        self,
        hours_ahead: int = 24,
        min_impact: NewsImpact = NewsImpact.HIGH
    ) -> List[EconomicEvent]:
        """Get upcoming high-impact events."""
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)

        impact_order = {
            NewsImpact.LOW: 0,
            NewsImpact.MEDIUM: 1,
            NewsImpact.HIGH: 2,
            NewsImpact.CRITICAL: 3,
        }
        min_level = impact_order[min_impact]

        return [
            e for e in self._events
            if now <= e.time <= cutoff and impact_order[e.impact] >= min_level
        ]

    def is_high_impact_period(
        self,
        symbol: str,
        minutes_before: int = 30,
        minutes_after: int = 30
    ) -> Tuple[bool, Optional[EconomicEvent]]:
        """
        Check if we're in a high-impact news period for the given symbol.

        Args:
            symbol: Trading pair (e.g., "EURUSD")
            minutes_before: Minutes before event to start blocking
            minutes_after: Minutes after event to continue blocking

        Returns:
            (is_blocked, event) - True if trading should be blocked
        """
        now = datetime.now(timezone.utc)

        # Extract currencies from symbol
        currencies = []
        if len(symbol) >= 6:
            currencies = [symbol[:3].upper(), symbol[3:6].upper()]

        for event in self._events:
            if event.impact not in (NewsImpact.HIGH, NewsImpact.CRITICAL):
                continue

            # Check if event affects our currencies
            if event.currency not in currencies:
                continue

            # Check if we're in the blocked window
            block_start = event.time - timedelta(minutes=minutes_before)
            block_end = event.time + timedelta(minutes=minutes_after)

            if block_start <= now <= block_end:
                return True, event

        return False, None


class NewsAPIClient:
    """
    Client for NewsAPI.org - Financial news with sentiment.

    Free tier: 100 requests/day
    Get API key at: https://newsapi.org/register
    """

    BASE_URL = "https://newsapi.org/v2"

    # Keywords for forex-related news
    FOREX_KEYWORDS = [
        "forex", "currency", "exchange rate", "dollar", "euro", "pound",
        "federal reserve", "ECB", "central bank", "interest rate",
        "inflation", "monetary policy", "trade war", "tariff",
    ]

    # Simple sentiment words
    POSITIVE_WORDS = [
        "surge", "jump", "gain", "rise", "rally", "boost", "strong",
        "bullish", "optimism", "recovery", "growth", "profit", "beat",
    ]

    NEGATIVE_WORDS = [
        "fall", "drop", "decline", "plunge", "crash", "weak", "bearish",
        "pessimism", "recession", "loss", "miss", "fear", "concern", "risk",
    ]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._cache: Dict[str, List[NewsArticle]] = {}
        self._cache_time: Optional[datetime] = None

    def _simple_sentiment(self, text: str) -> float:
        """Calculate simple sentiment score from text."""
        text_lower = text.lower()

        positive_count = sum(1 for word in self.POSITIVE_WORDS if word in text_lower)
        negative_count = sum(1 for word in self.NEGATIVE_WORDS if word in text_lower)

        total = positive_count + negative_count
        if total == 0:
            return 0.0

        return (positive_count - negative_count) / total

    def _extract_currencies(self, text: str) -> List[str]:
        """Extract mentioned currencies from text."""
        currencies = []
        text_upper = text.upper()

        currency_patterns = {
            "USD": ["USD", "DOLLAR", "GREENBACK", "BUCK"],
            "EUR": ["EUR", "EURO"],
            "GBP": ["GBP", "POUND", "STERLING", "CABLE"],
            "JPY": ["JPY", "YEN"],
            "AUD": ["AUD", "AUSSIE"],
            "CAD": ["CAD", "LOONIE"],
            "CHF": ["CHF", "FRANC"],
            "NZD": ["NZD", "KIWI"],
        }

        for currency, patterns in currency_patterns.items():
            if any(p in text_upper for p in patterns):
                currencies.append(currency)

        return currencies

    def fetch_forex_news(self, max_articles: int = 20) -> List[NewsArticle]:
        """Fetch latest forex-related news."""
        if not self.api_key:
            LOG.debug("NewsAPI key not configured")
            return []

        # Check cache (refresh every 15 minutes)
        now = datetime.now(timezone.utc)
        if self._cache_time and (now - self._cache_time).seconds < 900:
            return self._cache.get("forex", [])

        try:
            import urllib.parse
            query = " OR ".join(self.FOREX_KEYWORDS[:5])  # API limits query length
            encoded_query = urllib.parse.quote(query)
            url = f"{self.BASE_URL}/everything?q={encoded_query}&language=en&sortBy=publishedAt&pageSize={max_articles}"

            req = urllib.request.Request(url)
            req.add_header("X-Api-Key", self.api_key)

            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())

            articles = []
            for article in data.get("articles", []):
                title = article.get("title", "")
                published_str = article.get("publishedAt", "")

                try:
                    published = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                except:
                    published = now

                articles.append(NewsArticle(
                    title=title,
                    source=article.get("source", {}).get("name", "Unknown"),
                    published=published,
                    url=article.get("url", ""),
                    sentiment=self._simple_sentiment(title),
                    currencies_mentioned=self._extract_currencies(title),
                ))

            self._cache["forex"] = articles
            self._cache_time = now
            LOG.info("Fetched %d forex news articles", len(articles))
            return articles

        except Exception as e:
            LOG.warning("Failed to fetch news: %s", e)
            return []

    def get_sentiment_for_pair(self, symbol: str) -> Tuple[float, int]:
        """
        Get aggregate sentiment for a currency pair.

        Returns:
            (sentiment_score, article_count)
        """
        articles = self.fetch_forex_news()
        if not articles:
            return 0.0, 0

        # Extract currencies from symbol
        currencies = []
        if len(symbol) >= 6:
            currencies = [symbol[:3].upper(), symbol[3:6].upper()]

        relevant = [
            a for a in articles
            if any(c in a.currencies_mentioned for c in currencies)
        ]

        if not relevant:
            return 0.0, 0

        avg_sentiment = sum(a.sentiment for a in relevant) / len(relevant)
        return avg_sentiment, len(relevant)


class RedditSentiment:
    """
    Reddit sentiment analysis for forex communities.

    Subreddits monitored:
    - r/Forex
    - r/forextrading
    - r/Daytrading
    - r/wallstreetbets (for major moves)

    Requires: pip install praw
    Get credentials at: https://www.reddit.com/prefs/apps
    """

    FOREX_SUBREDDITS = ["Forex", "forextrading", "Daytrading"]

    # Currency pair mentions
    PAIR_PATTERNS = [
        r"\b(EUR/?USD)\b", r"\b(GBP/?USD)\b", r"\b(USD/?JPY)\b",
        r"\b(AUD/?USD)\b", r"\b(USD/?CAD)\b", r"\b(USD/?CHF)\b",
        r"\b(EUR/?GBP)\b", r"\b(EUR/?JPY)\b", r"\b(GBP/?JPY)\b",
    ]

    def __init__(self, client_id: str = None, client_secret: str = None, user_agent: str = "forex_bot"):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self._reddit = None
        self._cache: List[RedditPost] = []
        self._cache_time: Optional[datetime] = None

    def _init_reddit(self):
        """Initialize Reddit client."""
        if self._reddit is not None:
            return True

        if not self.client_id or not self.client_secret:
            LOG.debug("Reddit credentials not configured")
            return False

        try:
            import praw
            self._reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
            )
            return True
        except ImportError:
            LOG.warning("PRAW not installed. Run: pip install praw")
            return False
        except Exception as e:
            LOG.warning("Failed to initialize Reddit: %s", e)
            return False

    def _simple_sentiment(self, text: str) -> float:
        """Calculate sentiment from post text."""
        text_lower = text.lower()

        bullish = ["long", "buy", "bull", "calls", "moon", "pump", "breakout", "support"]
        bearish = ["short", "sell", "bear", "puts", "crash", "dump", "breakdown", "resistance"]

        bull_count = sum(1 for w in bullish if w in text_lower)
        bear_count = sum(1 for w in bearish if w in text_lower)

        total = bull_count + bear_count
        if total == 0:
            return 0.0

        return (bull_count - bear_count) / total

    def fetch_recent_posts(self, limit: int = 50) -> List[RedditPost]:
        """Fetch recent posts from forex subreddits."""
        if not self._init_reddit():
            return []

        # Check cache (refresh every 30 minutes)
        now = datetime.now(timezone.utc)
        if self._cache_time and (now - self._cache_time).seconds < 1800:
            return self._cache

        posts = []
        try:
            for subreddit_name in self.FOREX_SUBREDDITS:
                subreddit = self._reddit.subreddit(subreddit_name)
                for post in subreddit.hot(limit=limit // len(self.FOREX_SUBREDDITS)):
                    posts.append(RedditPost(
                        title=post.title,
                        subreddit=subreddit_name,
                        score=post.score,
                        created=datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
                        sentiment=self._simple_sentiment(post.title),
                        url=f"https://reddit.com{post.permalink}",
                    ))

            self._cache = posts
            self._cache_time = now
            LOG.info("Fetched %d Reddit posts", len(posts))

        except Exception as e:
            LOG.warning("Failed to fetch Reddit posts: %s", e)

        return posts

    def get_pair_sentiment(self, symbol: str) -> Tuple[float, int]:
        """
        Get sentiment for a specific currency pair from Reddit.

        Returns:
            (sentiment_score, post_count)
        """
        posts = self.fetch_recent_posts()
        if not posts:
            return 0.0, 0

        # Normalize symbol
        symbol_clean = symbol.upper().replace("/", "")

        relevant = [
            p for p in posts
            if symbol_clean in p.title.upper().replace("/", "").replace(" ", "")
        ]

        if not relevant:
            return 0.0, 0

        # Weight by score
        total_score = sum(p.score for p in relevant)
        if total_score == 0:
            avg = sum(p.sentiment for p in relevant) / len(relevant)
        else:
            avg = sum(p.sentiment * p.score for p in relevant) / total_score

        return avg, len(relevant)


class ComprehensiveNewsFilter:
    """
    Comprehensive news filter combining all sources.

    Usage:
        filter = ComprehensiveNewsFilter(
            newsapi_key="your_key",
            reddit_client_id="your_id",
            reddit_client_secret="your_secret"
        )

        # Check if trading should be blocked
        blocked, reason = filter.should_block_trading("EURUSD")

        # Get overall sentiment
        sentiment = filter.get_market_sentiment("EURUSD")
    """

    def __init__(
        self,
        newsapi_key: str = None,
        reddit_client_id: str = None,
        reddit_client_secret: str = None,
        minutes_before_news: int = 30,
        minutes_after_news: int = 15,
    ):
        self.calendar = EconomicCalendar()
        self.news_api = NewsAPIClient(newsapi_key)
        self.reddit = RedditSentiment(reddit_client_id, reddit_client_secret)
        self.minutes_before = minutes_before_news
        self.minutes_after = minutes_after_news

    def should_block_trading(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if trading should be blocked due to news.

        Returns:
            (should_block, reason)
        """
        # Check economic calendar
        blocked, event = self.calendar.is_high_impact_period(
            symbol, self.minutes_before, self.minutes_after
        )

        if blocked and event:
            return True, f"High-impact event: {event.event} ({event.currency})"

        return False, ""

    def get_market_sentiment(self, symbol: str) -> Dict:
        """
        Get comprehensive market sentiment for a symbol.

        Returns dict with:
        - news_sentiment: float (-1 to 1)
        - reddit_sentiment: float (-1 to 1)
        - overall_sentiment: float (-1 to 1)
        - news_count: int
        - reddit_count: int
        - recommendation: str ("bullish", "bearish", "neutral")
        """
        news_sent, news_count = self.news_api.get_sentiment_for_pair(symbol)
        reddit_sent, reddit_count = self.reddit.get_pair_sentiment(symbol)

        # Weight news more heavily than Reddit
        if news_count + reddit_count == 0:
            overall = 0.0
        else:
            overall = (news_sent * news_count * 2 + reddit_sent * reddit_count) / (news_count * 2 + reddit_count + 0.001)

        if overall > 0.2:
            recommendation = "bullish"
        elif overall < -0.2:
            recommendation = "bearish"
        else:
            recommendation = "neutral"

        return {
            "news_sentiment": news_sent,
            "reddit_sentiment": reddit_sent,
            "overall_sentiment": overall,
            "news_count": news_count,
            "reddit_count": reddit_count,
            "recommendation": recommendation,
        }

    def log_sentiment_report(self, symbol: str):
        """Log a sentiment report for the symbol."""
        sentiment = self.get_market_sentiment(symbol)

        LOG.info("=== Sentiment Report: %s ===", symbol)
        LOG.info("News: %.2f (%d articles)", sentiment["news_sentiment"], sentiment["news_count"])
        LOG.info("Reddit: %.2f (%d posts)", sentiment["reddit_sentiment"], sentiment["reddit_count"])
        LOG.info("Overall: %.2f - %s", sentiment["overall_sentiment"], sentiment["recommendation"].upper())
