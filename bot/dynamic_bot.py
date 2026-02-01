"""
Dynamic Trading Bot - Fully autonomous pair selection and trading.

This bot:
1. Scans all major pairs for the best opportunity
2. Uses currency strength + sentiment to pick the pair
3. Only trades when sentiment aligns with technical signals
4. Automatically adapts to market conditions

Usage:
    python -m bot.dynamic_bot --config configs/dynamic.yaml
"""
import argparse
import time
import logging
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

from .config import AppConfig
from .monitoring import setup_logging, log_health
from .mt5_client import MT5Client
from .execution import ExecutionEngine, Signal
from .strategy import StrategyEngine
from .news import NewsFilter
from .risk import RiskManager
from .dynamic_selector import DynamicSymbolSelector
from .market_intel import MarketIntelligence

LOG = logging.getLogger("bot.dynamic")


class DynamicTradingBot:
    """
    Fully autonomous trading bot with dynamic pair selection.

    Features:
    - Scans multiple pairs for best opportunity
    - Currency strength analysis
    - News/social sentiment integration
    - Volatility-based position sizing
    - Automatic session filtering
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

        # Core components
        self.mt5 = MT5Client()
        self.mt5.connect()

        self.risk = RiskManager(cfg, self.mt5)
        self.news_filter = NewsFilter(cfg)
        self.exec_engine = ExecutionEngine(cfg, self.mt5, self.risk)

        # Market intelligence
        self.intel = MarketIntelligence(self.mt5)

        # Dynamic selector
        self.selector = DynamicSymbolSelector(
            self.mt5,
            cfg,
            news_filter=self.news_filter,
            market_intel=self.intel,
        )

        # Strategy engines per symbol (lazily created)
        self._strategies: dict = {}

        # Tracking
        self._current_symbol: str = cfg.symbol  # Default
        self._current_direction: str = ""
        self._last_scan_time: datetime = datetime.min.replace(tzinfo=timezone.utc)
        self._scan_interval_seconds = 300  # Re-scan every 5 minutes

    def _get_strategy(self, symbol: str) -> StrategyEngine:
        """Get or create strategy engine for a symbol."""
        if symbol not in self._strategies:
            # Create a config copy with the new symbol
            cfg_copy = self.cfg
            cfg_copy.symbol = symbol
            self._strategies[symbol] = StrategyEngine(cfg_copy, self.mt5)
        return self._strategies[symbol]

    def _should_rescan(self) -> bool:
        """Check if we should rescan for a new pair."""
        now = datetime.now(timezone.utc)
        seconds_since = (now - self._last_scan_time).total_seconds()
        return seconds_since >= self._scan_interval_seconds

    def _scan_for_best_pair(self) -> bool:
        """
        Scan market and select the best pair to trade.

        Returns:
            True if a tradeable pair was found
        """
        self._last_scan_time = datetime.now(timezone.utc)

        result = self.selector.get_best_pair(min_score=60.0)

        if not result:
            LOG.info("No pairs meet trading criteria")
            return False

        symbol, direction, score = result

        # Check if pair changed
        if symbol != self._current_symbol:
            LOG.info(
                "Switching from %s to %s %s (score: %.1f)",
                self._current_symbol, symbol, direction, score.total_score
            )
            self._current_symbol = symbol
            self._current_direction = direction

            # Update config symbol for risk/execution
            self.cfg.symbol = symbol
        else:
            LOG.debug("Keeping %s %s (score: %.1f)",
                      symbol, direction, score.total_score)

        return True

    def _check_sentiment_alignment(self, signal: Signal) -> bool:
        """
        Check if the technical signal aligns with market sentiment.

        Returns:
            True if aligned (OK to trade), False if conflicting
        """
        if not self.news_filter:
            return True

        try:
            allow_buy, allow_sell, reason = self.news_filter.should_trade_direction()

            if signal.side == "BUY" and not allow_buy:
                LOG.warning("Signal rejected: BUY blocked by sentiment (%s)", reason)
                return False

            if signal.side == "SELL" and not allow_sell:
                LOG.warning("Signal rejected: SELL blocked by sentiment (%s)", reason)
                return False

            return True

        except Exception as e:
            LOG.warning("Error checking sentiment alignment: %s", e)
            return True  # Allow on error

    def _check_strength_alignment(self, signal: Signal) -> bool:
        """
        Check if the technical signal aligns with currency strength.

        Returns:
            True if aligned, False if conflicting
        """
        # If we have a preferred direction from the selector, check alignment
        if self._current_direction:
            if signal.side != self._current_direction:
                LOG.info(
                    "Signal %s conflicts with strength direction %s - skipping",
                    signal.side, self._current_direction
                )
                return False

        return True

    def run_iteration(self) -> None:
        """Run a single iteration of the trading loop."""
        # Refresh risk state
        self.risk.refresh_daily_circuit_breaker()
        self.exec_engine.sync_open_positions()
        self.exec_engine.manage_positions()

        # Check if we can trade
        if not self.risk.can_trade_now():
            return

        # Periodically rescan for best pair
        if self._should_rescan():
            self._scan_for_best_pair()

        # Check news filter
        if self.cfg.news_filter.enabled and self.news_filter.block_new_entries():
            return

        # Get technical signal for current symbol
        strategy = self._get_strategy(self._current_symbol)
        signal = strategy.get_signal()

        if signal is None:
            return

        # Check alignment with market context
        if not self._check_sentiment_alignment(signal):
            return

        if not self._check_strength_alignment(signal):
            return

        # Execute the trade
        LOG.info(
            "EXECUTING: %s %s - %s",
            self._current_symbol, signal.side, signal.reason
        )
        self.exec_engine.execute_signal(signal)

    def run(self) -> None:
        """Run the bot continuously."""
        log_health(self.mt5, self.cfg, "DYNAMIC_BOT_STARTED")

        # Initial market scan and report
        LOG.info("=" * 60)
        LOG.info("DYNAMIC TRADING BOT STARTED")
        LOG.info("=" * 60)

        self.intel.log_market_report()
        self.selector.log_market_scan()

        if self.news_filter:
            self.news_filter.log_status()

        try:
            while True:
                self.run_iteration()
                time.sleep(1.0)

        except KeyboardInterrupt:
            log_health(self.mt5, self.cfg, "STOPPED: KeyboardInterrupt")
        except Exception as e:
            log_health(self.mt5, self.cfg, f"CRASHED: {e}")
            raise
        finally:
            self.mt5.shutdown()

    def scan_and_report(self) -> None:
        """One-time market scan and report (no trading)."""
        LOG.info("=" * 60)
        LOG.info("MARKET SCAN REPORT")
        LOG.info("=" * 60)

        self.intel.log_market_report()
        self.selector.log_market_scan()

        if self.news_filter:
            self.news_filter.log_status()

        self.mt5.shutdown()


def main():
    load_dotenv()

    ap = argparse.ArgumentParser(description="Dynamic Trading Bot")
    ap.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (e.g., configs/dynamic.yaml)"
    )
    ap.add_argument(
        "--scan-only",
        action="store_true",
        help="Just scan and report, don't trade"
    )
    args = ap.parse_args()

    cfg = AppConfig.from_yaml(Path(args.config).resolve())
    setup_logging(cfg.logging.level)

    bot = DynamicTradingBot(cfg)

    if args.scan_only:
        bot.scan_and_report()
    else:
        bot.run()


if __name__ == "__main__":
    main()
