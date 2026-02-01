"""
News Filter - Blocks trading during high-impact events.

Integrates with:
1. Economic Calendar (NFP, FOMC, ECB, etc.)
2. NewsAPI.org for financial news sentiment
3. FREE RSS Feeds (ForexLive, FXStreet, DailyFX, Investing.com)
4. Reddit sentiment (r/Forex, r/forextrading) - optional
5. Twitter/X - optional (requires paid API)

Configure API keys in .env file (optional - RSS feeds work without keys).
"""
import os
import logging
from typing import Tuple, Optional, Dict
from .config import AppConfig

LOG = logging.getLogger("bot.news")


class NewsFilter:
    """
    News filter that blocks trading during high-impact events.

    Providers:
    - "stub": Never blocks (default, for testing)
    - "calendar": Economic calendar only
    - "full": Calendar + news sentiment + RSS feeds + optional APIs

    Free sources (no API key needed):
    - RSS feeds from ForexLive, FXStreet, DailyFX, Investing.com

    Optional API Keys (set in .env):
    - NEWSAPI_KEY: From https://newsapi.org (100 req/day free)
    - REDDIT_CLIENT_ID/SECRET: From reddit.com/prefs/apps
    - TWITTER_BEARER_TOKEN: From developer.twitter.com (paid only)
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._provider = None
        self._social_news = None
        self._free_news = None  # Free RSS feeds
        self._init_provider()

    def _init_provider(self):
        """Initialize the news provider based on config."""
        provider_name = self.cfg.news_filter.provider

        if provider_name == "stub":
            LOG.info("News filter: STUB mode (not blocking)")
            return

        try:
            from .news_provider import ComprehensiveNewsFilter

            self._provider = ComprehensiveNewsFilter(
                newsapi_key=os.getenv("NEWSAPI_KEY"),
                reddit_client_id=os.getenv("REDDIT_CLIENT_ID"),
                reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
                minutes_before_news=self.cfg.news_filter.minutes_before,
                minutes_after_news=self.cfg.news_filter.minutes_after,
            )
            LOG.info("News filter: %s mode initialized", provider_name.upper())

        except ImportError as e:
            LOG.warning("Failed to import news provider: %s", e)
        except Exception as e:
            LOG.warning("Failed to initialize news provider: %s", e)

        # Initialize FREE RSS news feeds (no API key needed)
        try:
            from .free_news_feeds import FreeNewsFetcher

            self._free_news = FreeNewsFetcher()
            LOG.info("Free RSS news: ForexLive, FXStreet, DailyFX, Investing.com")

        except ImportError as e:
            LOG.warning("Failed to import free news feeds: %s", e)
        except Exception as e:
            LOG.warning("Failed to initialize free news feeds: %s", e)

        # Initialize social news (Twitter, Truth Social) - optional, may fail on free tier
        try:
            from .social_news import SocialNewsAggregator

            self._social_news = SocialNewsAggregator(
                twitter_token=os.getenv("TWITTER_BEARER_TOKEN"),
            )
            LOG.info("Social news: Twitter/X initialized (may require paid API)")

        except ImportError as e:
            LOG.warning("Failed to import social news: %s", e)
        except Exception as e:
            LOG.warning("Failed to initialize social news: %s", e)

    def block_new_entries(self) -> bool:
        """
        Check if new entries should be blocked.

        Returns:
            True if trading should be blocked.
        """
        if self.cfg.news_filter.provider == "stub":
            return False

        # Check economic calendar and news API
        if self._provider is not None:
            try:
                blocked, reason = self._provider.should_block_trading(self.cfg.symbol)
                if blocked:
                    LOG.warning("Trading blocked: %s", reason)
                    return True
            except Exception as e:
                LOG.warning("Error checking news provider: %s", e)

        # Check social media for breaking news
        if self._social_news is not None:
            try:
                blocked, reason = self._social_news.should_block_trading(self.cfg.symbol)
                if blocked:
                    LOG.warning("Trading blocked (social): %s", reason)
                    return True
            except Exception as e:
                LOG.warning("Error checking social news: %s", e)

        return False

    def get_sentiment(self) -> Optional[Dict]:
        """
        Get market sentiment for the configured symbol.

        Returns:
            Dict with sentiment scores, or None if unavailable.
        """
        result = {
            "news_sentiment": 0,
            "reddit_sentiment": 0,
            "social_sentiment": 0,
            "rss_sentiment": 0,  # Free RSS feeds
            "overall_sentiment": 0,
            "recommendation": "neutral",
        }

        # Get FREE RSS news sentiment (primary source - no API limits)
        if self._free_news is not None:
            try:
                rss_data = self._free_news.get_sentiment_for_pair(self.cfg.symbol)
                if rss_data:
                    bias = rss_data.get("overall_bias", "neutral")
                    if bias == "bullish":
                        result["rss_sentiment"] = 0.5
                    elif bias == "bearish":
                        result["rss_sentiment"] = -0.5
                    LOG.debug("RSS sentiment for %s: %s (%d articles)",
                             self.cfg.symbol, bias, rss_data.get("news_count", 0))
            except Exception as e:
                LOG.warning("Error getting RSS sentiment: %s", e)

        # Get news provider sentiment (NewsAPI - may have limits)
        if self._provider is not None:
            try:
                provider_sentiment = self._provider.get_market_sentiment(self.cfg.symbol)
                if provider_sentiment:
                    result["news_sentiment"] = provider_sentiment.get("news_sentiment", 0)
                    result["reddit_sentiment"] = provider_sentiment.get("reddit_sentiment", 0)
            except Exception as e:
                LOG.warning("Error getting provider sentiment: %s", e)

        # Get social media sentiment (Twitter - may fail on free tier)
        if self._social_news is not None:
            try:
                social = self._social_news.get_sentiment_for_pair(self.cfg.symbol)
                if social:
                    bias = social.get("overall_bias", "neutral")
                    if bias == "bullish":
                        result["social_sentiment"] = 0.5
                    elif bias == "bearish":
                        result["social_sentiment"] = -0.5
                    else:
                        result["social_sentiment"] = 0
            except Exception as e:
                LOG.debug("Social sentiment unavailable: %s", e)

        # Calculate overall sentiment (weight RSS higher since it's always available)
        sentiments = []
        if result["rss_sentiment"] != 0:
            sentiments.append(result["rss_sentiment"] * 1.5)  # Weight RSS higher
        if result["news_sentiment"] != 0:
            sentiments.append(result["news_sentiment"])
        if result["reddit_sentiment"] != 0:
            sentiments.append(result["reddit_sentiment"])
        if result["social_sentiment"] != 0:
            sentiments.append(result["social_sentiment"])

        if sentiments:
            result["overall_sentiment"] = sum(sentiments) / len(sentiments)

        # Determine recommendation
        overall = result["overall_sentiment"]
        if overall > 0.2:
            result["recommendation"] = "bullish"
        elif overall < -0.2:
            result["recommendation"] = "bearish"
        else:
            result["recommendation"] = "neutral"

        return result

    def should_trade_direction(self) -> Tuple[bool, bool, str]:
        """
        Get recommended trading direction based on sentiment.

        Returns:
            (allow_buy, allow_sell, reason)

        Example:
            allow_buy, allow_sell, reason = news_filter.should_trade_direction()
            if signal.side == "BUY" and not allow_buy:
                return  # Skip this signal
        """
        sentiment = self.get_sentiment()

        if sentiment is None:
            return True, True, "No sentiment data"

        overall = sentiment.get("overall_sentiment", 0)
        recommendation = sentiment.get("recommendation", "neutral")

        if recommendation == "bullish":
            return True, False, f"Bullish sentiment ({overall:.2f})"
        elif recommendation == "bearish":
            return False, True, f"Bearish sentiment ({overall:.2f})"
        else:
            return True, True, f"Neutral sentiment ({overall:.2f})"

    def log_status(self):
        """Log current news/sentiment status."""
        if self._provider is None and self._social_news is None and self._free_news is None:
            LOG.info("News filter: Disabled or not configured")
            return

        # Log FREE RSS news (primary source)
        if self._free_news is not None:
            try:
                self._free_news.log_news_report(self.cfg.symbol)
            except Exception as e:
                LOG.warning("Error logging RSS news: %s", e)

        # Log news provider sentiment
        if self._provider is not None:
            try:
                self._provider.log_sentiment_report(self.cfg.symbol)
            except Exception as e:
                LOG.warning("Error logging provider sentiment: %s", e)

        # Log social media sentiment (may fail on free Twitter tier)
        if self._social_news is not None:
            try:
                self._social_news.log_social_report(self.cfg.symbol)
            except Exception as e:
                LOG.debug("Social report unavailable: %s", e)
