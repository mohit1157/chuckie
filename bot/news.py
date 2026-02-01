"""
News Filter - Blocks trading during high-impact events.

Integrates with:
1. Economic Calendar (NFP, FOMC, ECB, etc.)
2. NewsAPI.org for financial news sentiment
3. Reddit sentiment (r/Forex, r/forextrading)

Configure API keys in .env file.
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
    - "full": Calendar + news sentiment + Reddit

    API Keys (set in .env):
    - NEWSAPI_KEY: From https://newsapi.org
    - REDDIT_CLIENT_ID: From https://reddit.com/prefs/apps
    - REDDIT_CLIENT_SECRET: From Reddit app settings
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._provider = None
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

    def block_new_entries(self) -> bool:
        """
        Check if new entries should be blocked.

        Returns:
            True if trading should be blocked.
        """
        if self.cfg.news_filter.provider == "stub":
            return False

        if self._provider is None:
            return False

        try:
            blocked, reason = self._provider.should_block_trading(self.cfg.symbol)
            if blocked:
                LOG.warning("Trading blocked: %s", reason)
            return blocked
        except Exception as e:
            LOG.warning("Error checking news: %s", e)
            return False

    def get_sentiment(self) -> Optional[Dict]:
        """
        Get market sentiment for the configured symbol.

        Returns:
            Dict with sentiment scores, or None if unavailable.
        """
        if self._provider is None:
            return None

        try:
            return self._provider.get_market_sentiment(self.cfg.symbol)
        except Exception as e:
            LOG.warning("Error getting sentiment: %s", e)
            return None

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
        if self._provider is None:
            LOG.info("News filter: Disabled or not configured")
            return

        try:
            self._provider.log_sentiment_report(self.cfg.symbol)
        except Exception as e:
            LOG.warning("Error logging sentiment: %s", e)
