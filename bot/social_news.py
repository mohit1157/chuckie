"""
Social Media News Integration for Forex Trading.

Monitors key accounts for market-moving news:
1. Twitter/X: @DeItaone (Walter Bloomberg - breaking financial news)
2. Truth Social: @realDonaldTrump (President Trump - policy/market statements)

These accounts often post market-moving news before traditional media.
"""
import os
import logging
import json
import re
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import urllib.request
import urllib.error

LOG = logging.getLogger("bot.social_news")


@dataclass
class SocialPost:
    """A social media post with sentiment."""
    platform: str  # "twitter" or "truth_social"
    username: str
    text: str
    timestamp: datetime
    sentiment: float  # -1 to +1
    market_impact: str  # "high", "medium", "low"
    currencies_affected: List[str]
    url: str


class TwitterClient:
    """
    Twitter/X API client for financial news accounts.

    Key accounts monitored:
    - @DeItaone (Walter Bloomberg): Breaking financial news, Fed, earnings
    - @zaborhedge: Hedge fund news
    - @FirstSquawk: Breaking market news
    - @LiveSquawk: Real-time market updates

    Requires Twitter API v2 Bearer Token.
    Get at: https://developer.twitter.com/en/portal/dashboard
    """

    BASE_URL = "https://api.twitter.com/2"

    # Key accounts for forex/market news
    MONITORED_ACCOUNTS = {
        "DeItaone": "Walter Bloomberg - Breaking news",
        "FirstSquawk": "Breaking market news",
        "LiveSquawk": "Real-time updates",
        "zaborhedge": "Hedge fund news",
        "ForexLive": "Forex specific news",
        "ReutersGMF": "Reuters Global Markets",
    }

    # Keywords that indicate high market impact
    HIGH_IMPACT_KEYWORDS = [
        "breaking", "just in", "fed", "fomc", "powell", "rate",
        "trump", "biden", "tariff", "china", "trade war",
        "ecb", "lagarde", "boe", "inflation", "cpi", "gdp",
        "employment", "payroll", "nfp", "jobs",
        "oil", "opec", "crude", "wti", "brent",
        "war", "invasion", "attack", "military",
        "default", "crisis", "crash", "collapse",
    ]

    # Currency detection patterns
    CURRENCY_PATTERNS = {
        "USD": ["usd", "dollar", "greenback", "fed", "fomc", "powell", "us ", "american"],
        "EUR": ["eur", "euro", "ecb", "lagarde", "eurozone", "german"],
        "GBP": ["gbp", "pound", "sterling", "boe", "bank of england", "uk ", "british"],
        "JPY": ["jpy", "yen", "boj", "japan", "kuroda", "ueda"],
        "AUD": ["aud", "aussie", "rba", "australia"],
        "CAD": ["cad", "loonie", "boc", "canada", "oil"],
        "CHF": ["chf", "franc", "snb", "swiss"],
        "CNY": ["cny", "yuan", "renminbi", "china", "pboc"],
    }

    # Sentiment words
    BULLISH_WORDS = [
        "surge", "soar", "jump", "rally", "gain", "rise", "up",
        "bullish", "strong", "growth", "beat", "exceed", "optimism",
        "recovery", "support", "buy", "long", "higher", "peak",
    ]

    BEARISH_WORDS = [
        "drop", "fall", "plunge", "crash", "sink", "decline", "down",
        "bearish", "weak", "recession", "miss", "disappoint", "fear",
        "crisis", "sell", "short", "lower", "bottom", "cut",
    ]

    def __init__(self, bearer_token: str = None):
        self.bearer_token = bearer_token or os.getenv("TWITTER_BEARER_TOKEN")
        self._cache: List[SocialPost] = []
        self._cache_time: Optional[datetime] = None
        self._user_ids: Dict[str, str] = {}  # username -> user_id

    def _make_request(self, endpoint: str) -> Optional[dict]:
        """Make authenticated request to Twitter API."""
        if not self.bearer_token:
            LOG.debug("Twitter bearer token not configured")
            return None

        try:
            url = f"{self.BASE_URL}/{endpoint}"
            req = urllib.request.Request(url)
            req.add_header("Authorization", f"Bearer {self.bearer_token}")

            with urllib.request.urlopen(req, timeout=10) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            LOG.warning("Twitter API error: %s", e)
            return None
        except Exception as e:
            LOG.warning("Twitter request failed: %s", e)
            return None

    def _get_user_id(self, username: str) -> Optional[str]:
        """Get Twitter user ID from username."""
        if username in self._user_ids:
            return self._user_ids[username]

        data = self._make_request(f"users/by/username/{username}")
        if data and "data" in data:
            user_id = data["data"]["id"]
            self._user_ids[username] = user_id
            return user_id
        return None

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of tweet text."""
        text_lower = text.lower()

        bullish_count = sum(1 for word in self.BULLISH_WORDS if word in text_lower)
        bearish_count = sum(1 for word in self.BEARISH_WORDS if word in text_lower)

        total = bullish_count + bearish_count
        if total == 0:
            return 0.0

        return (bullish_count - bearish_count) / total

    def _detect_currencies(self, text: str) -> List[str]:
        """Detect which currencies are mentioned in the text."""
        text_lower = text.lower()
        currencies = []

        for currency, patterns in self.CURRENCY_PATTERNS.items():
            if any(p in text_lower for p in patterns):
                currencies.append(currency)

        return currencies if currencies else ["USD"]  # Default to USD

    def _assess_impact(self, text: str) -> str:
        """Assess market impact level of the news."""
        text_lower = text.lower()

        # Count high-impact keywords
        impact_count = sum(1 for kw in self.HIGH_IMPACT_KEYWORDS if kw in text_lower)

        if impact_count >= 3 or any(kw in text_lower for kw in ["breaking", "just in", "fomc", "fed"]):
            return "high"
        elif impact_count >= 1:
            return "medium"
        else:
            return "low"

    def fetch_recent_tweets(self, username: str, max_results: int = 10) -> List[SocialPost]:
        """Fetch recent tweets from a specific account."""
        user_id = self._get_user_id(username)
        if not user_id:
            return []

        # Get recent tweets
        endpoint = f"users/{user_id}/tweets?max_results={max_results}&tweet.fields=created_at,text"
        data = self._make_request(endpoint)

        if not data or "data" not in data:
            return []

        posts = []
        for tweet in data["data"]:
            try:
                created_at = datetime.fromisoformat(tweet["created_at"].replace("Z", "+00:00"))
                text = tweet["text"]

                posts.append(SocialPost(
                    platform="twitter",
                    username=username,
                    text=text,
                    timestamp=created_at,
                    sentiment=self._analyze_sentiment(text),
                    market_impact=self._assess_impact(text),
                    currencies_affected=self._detect_currencies(text),
                    url=f"https://twitter.com/{username}/status/{tweet['id']}",
                ))
            except Exception as e:
                LOG.warning("Error parsing tweet: %s", e)

        return posts

    def fetch_all_monitored(self) -> List[SocialPost]:
        """Fetch recent tweets from all monitored accounts."""
        # Check cache (refresh every 2 minutes)
        now = datetime.now(timezone.utc)
        if self._cache_time and (now - self._cache_time).seconds < 120:
            return self._cache

        all_posts = []
        for username in self.MONITORED_ACCOUNTS.keys():
            posts = self.fetch_recent_tweets(username, max_results=5)
            all_posts.extend(posts)

        # Sort by timestamp
        all_posts.sort(key=lambda x: x.timestamp, reverse=True)

        self._cache = all_posts
        self._cache_time = now

        LOG.info("Fetched %d tweets from monitored accounts", len(all_posts))
        return all_posts

    def get_breaking_news(self, minutes: int = 15) -> List[SocialPost]:
        """Get breaking news from the last N minutes."""
        all_posts = self.fetch_all_monitored()
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)

        breaking = [p for p in all_posts if p.timestamp > cutoff and p.market_impact == "high"]
        return breaking


class TruthSocialClient:
    """
    Truth Social client for monitoring political/market news.

    Key account:
    - @realDonaldTrump: President Trump's statements on trade, tariffs, Fed, economy

    Note: Truth Social API is limited. This uses public RSS/web methods.
    """

    # Keywords that indicate market-moving Trump statements
    MARKET_KEYWORDS = [
        "tariff", "china", "trade", "fed", "powell", "rate", "interest",
        "economy", "jobs", "market", "stock", "dollar", "currency",
        "oil", "gas", "energy", "opec", "saudi",
        "mexico", "canada", "usmca", "nafta",
        "europe", "eu", "germany", "nato",
        "tax", "inflation", "growth", "gdp",
    ]

    BULLISH_USD_KEYWORDS = [
        "strong dollar", "america first", "winning", "great economy",
        "jobs up", "growth", "tax cut", "deregulation",
    ]

    BEARISH_USD_KEYWORDS = [
        "weak dollar", "fed", "cut rates", "too strong",
        "china", "tariff war", "trade war",
    ]

    def __init__(self):
        self._cache: List[SocialPost] = []
        self._cache_time: Optional[datetime] = None

    def _analyze_trump_sentiment(self, text: str) -> Tuple[float, List[str]]:
        """
        Analyze Trump statement for market sentiment.

        Returns:
            (sentiment, affected_currencies)
        """
        text_lower = text.lower()
        currencies = ["USD"]  # Trump statements primarily affect USD

        # Check for specific currency mentions
        if any(w in text_lower for w in ["china", "yuan", "cny"]):
            currencies.append("CNY")
        if any(w in text_lower for w in ["europe", "eu ", "euro"]):
            currencies.append("EUR")
        if any(w in text_lower for w in ["mexico", "peso"]):
            currencies.append("MXN")
        if any(w in text_lower for w in ["canada", "loonie"]):
            currencies.append("CAD")

        # Analyze sentiment
        bullish = sum(1 for kw in self.BULLISH_USD_KEYWORDS if kw in text_lower)
        bearish = sum(1 for kw in self.BEARISH_USD_KEYWORDS if kw in text_lower)

        total = bullish + bearish
        if total == 0:
            sentiment = 0.0
        else:
            sentiment = (bullish - bearish) / total

        return sentiment, currencies

    def _is_market_relevant(self, text: str) -> bool:
        """Check if post is market-relevant."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.MARKET_KEYWORDS)

    def fetch_recent_posts(self) -> List[SocialPost]:
        """
        Fetch recent Truth Social posts.

        Note: This is a placeholder. In production, you would:
        1. Use Truth Social API (if available)
        2. Use a third-party service that aggregates Truth Social
        3. Use RSS feeds if available
        """
        # Check cache
        now = datetime.now(timezone.utc)
        if self._cache_time and (now - self._cache_time).seconds < 300:
            return self._cache

        # Placeholder - in production, implement actual fetching
        # For now, return empty list
        LOG.debug("Truth Social fetching not implemented - requires API access")

        self._cache = []
        self._cache_time = now
        return []

    def get_market_moving_posts(self, hours: int = 24) -> List[SocialPost]:
        """Get market-moving posts from the last N hours."""
        posts = self.fetch_recent_posts()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        market_posts = [
            p for p in posts
            if p.timestamp > cutoff and self._is_market_relevant(p.text)
        ]
        return market_posts


class SocialNewsAggregator:
    """
    Aggregates news from all social media sources.

    Usage:
        aggregator = SocialNewsAggregator()

        # Check for breaking news
        breaking = aggregator.get_breaking_news()
        if breaking:
            for post in breaking:
                print(f"BREAKING: {post.text}")

        # Get sentiment for a currency pair
        sentiment = aggregator.get_sentiment_for_pair("EURUSD")
    """

    def __init__(
        self,
        twitter_token: str = None,
    ):
        self.twitter = TwitterClient(twitter_token)
        self.truth_social = TruthSocialClient()

    def get_all_recent_posts(self, minutes: int = 30) -> List[SocialPost]:
        """Get all recent posts from all sources."""
        posts = []

        # Twitter
        posts.extend(self.twitter.fetch_all_monitored())

        # Truth Social
        posts.extend(self.truth_social.fetch_recent_posts())

        # Filter by time
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        posts = [p for p in posts if p.timestamp > cutoff]

        # Sort by timestamp
        posts.sort(key=lambda x: x.timestamp, reverse=True)

        return posts

    def get_breaking_news(self, minutes: int = 15) -> List[SocialPost]:
        """Get high-impact breaking news."""
        posts = self.get_all_recent_posts(minutes=minutes)
        return [p for p in posts if p.market_impact == "high"]

    def get_sentiment_for_pair(self, symbol: str) -> Dict:
        """
        Get social media sentiment for a currency pair.

        Returns:
            {
                "base_sentiment": float,
                "quote_sentiment": float,
                "overall_bias": str,  # "bullish", "bearish", "neutral"
                "post_count": int,
                "breaking_news": bool,
            }
        """
        base = symbol[:3].upper()
        quote = symbol[3:6].upper()

        posts = self.get_all_recent_posts(minutes=60)

        base_posts = [p for p in posts if base in p.currencies_affected]
        quote_posts = [p for p in posts if quote in p.currencies_affected]

        base_sentiment = sum(p.sentiment for p in base_posts) / len(base_posts) if base_posts else 0
        quote_sentiment = sum(p.sentiment for p in quote_posts) / len(quote_posts) if quote_posts else 0

        # For pair, bullish base or bearish quote = bullish pair
        overall = base_sentiment - quote_sentiment

        if overall > 0.2:
            bias = "bullish"
        elif overall < -0.2:
            bias = "bearish"
        else:
            bias = "neutral"

        breaking = any(p.market_impact == "high" for p in posts if base in p.currencies_affected or quote in p.currencies_affected)

        return {
            "base_sentiment": base_sentiment,
            "quote_sentiment": quote_sentiment,
            "overall_bias": bias,
            "post_count": len(base_posts) + len(quote_posts),
            "breaking_news": breaking,
        }

    def should_block_trading(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if trading should be blocked due to breaking news.

        Returns:
            (should_block, reason)
        """
        breaking = self.get_breaking_news(minutes=10)

        base = symbol[:3].upper()
        quote = symbol[3:6].upper()

        for post in breaking:
            if base in post.currencies_affected or quote in post.currencies_affected:
                return True, f"Breaking news from @{post.username}: {post.text[:100]}..."

        return False, ""

    def log_social_report(self, symbol: str = "EURUSD"):
        """Log current social media sentiment report."""
        sentiment = self.get_sentiment_for_pair(symbol)
        breaking = self.get_breaking_news(minutes=30)

        LOG.info("=" * 50)
        LOG.info("SOCIAL MEDIA SENTIMENT REPORT: %s", symbol)
        LOG.info("=" * 50)
        LOG.info("Base (%s) sentiment: %.2f", symbol[:3], sentiment["base_sentiment"])
        LOG.info("Quote (%s) sentiment: %.2f", symbol[3:6], sentiment["quote_sentiment"])
        LOG.info("Overall bias: %s", sentiment["overall_bias"].upper())
        LOG.info("Posts analyzed: %d", sentiment["post_count"])
        LOG.info("-" * 50)

        if breaking:
            LOG.info("BREAKING NEWS (%d posts):", len(breaking))
            for post in breaking[:5]:
                LOG.info("  @%s: %s", post.username, post.text[:80])
        else:
            LOG.info("No breaking news")

        LOG.info("=" * 50)


# Convenience function
def get_social_news() -> SocialNewsAggregator:
    """Get social news aggregator instance."""
    return SocialNewsAggregator(
        twitter_token=os.getenv("TWITTER_BEARER_TOKEN"),
    )
