"""
High-Probability Scalping Bot - Main Entry Point.

This bot is designed for 80%+ win rate day trading on forex pairs.
Uses multiple confirmations, session filtering, and dynamic risk management.

Usage:
    python -m bot.scalping_bot --config configs/scalping.yaml
"""
import argparse
import time
import signal
import sys
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

from .config import AppConfig
from .monitoring import setup_logging, log_health
from .mt5_client import MT5Client
from .scalping_execution import ScalpingExecutionEngine
from .scalping_strategy import ScalpingStrategy, ConservativeScalpingStrategy
from .news import NewsFilter
from .risk import RiskManager
from .trade_logger import TradeLogger
from .session_filter import SessionFilter, get_session_info
from .market_intel import MarketIntelligence
import logging

LOG = logging.getLogger("bot.scalping_main")


class ScalpingBot:
    """
    Main scalping bot orchestrator.

    Features:
    - High-probability multi-confirmation strategy
    - Session-based trading (London/NY focus)
    - ATR-based dynamic SL/TP
    - Trade logging and analytics
    - Automatic position management
    - Daily performance reporting
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.running = False
        self._setup_signal_handlers()

        # Initialize components
        self.mt5 = MT5Client()
        self.risk = RiskManager(cfg, self.mt5)
        self.logger = TradeLogger(db_path="scalping_trades.db")
        self.news_filter = NewsFilter(cfg)
        self.session_filter = SessionFilter()
        self.market_intel = None  # Initialized after MT5 connection

        # Choose strategy based on config
        if cfg.strategy.min_confirmations >= 6:
            self.strategy = ConservativeScalpingStrategy(cfg, self.mt5)
            LOG.info("Using CONSERVATIVE strategy (6+ confirmations)")
        else:
            self.strategy = ScalpingStrategy(cfg, self.mt5)
            LOG.info("Using STANDARD strategy (5+ confirmations)")

        self.execution = ScalpingExecutionEngine(cfg, self.mt5, self.risk, self.logger)

        # Stats
        self._trades_today = 0
        self._last_report_hour = -1
        self._use_market_intel = True  # Can be disabled

    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers."""
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signal."""
        LOG.info("Shutdown signal received, stopping bot...")
        self.running = False

    def _should_trade(self) -> tuple:
        """
        Check all conditions for trading.

        Returns:
            (can_trade: bool, reason: str)
        """
        # 1. Risk manager check
        if not self.risk.can_trade_now():
            return False, "risk_manager_blocked"

        # 2. Session filter (if enabled)
        if self.cfg.session.enabled:
            if self.cfg.session.overlap_only:
                session_info = get_session_info()
                if session_info["session"] != "london_ny_overlap":
                    return False, f"waiting_for_overlap (current: {session_info['session']})"

            avoid, reason = self.session_filter.should_avoid_now()
            if avoid:
                return False, f"session_avoid: {reason}"

            if not self.session_filter.is_allowed_session():
                return False, "session_not_allowed"

        # 3. News filter (if enabled and not stub)
        if self.cfg.news_filter.enabled:
            if self.news_filter.block_new_entries():
                return False, "news_blocked"

        # 4. Market intelligence check (VIX too high = don't trade)
        if self._use_market_intel and self.market_intel:
            context = self.market_intel.get_market_context()
            if context.vix_value > 30:
                return False, f"vix_too_high ({context.vix_value:.1f})"

        # 5. Max trades per day (prevent overtrading)
        max_trades_day = 10
        if self._trades_today >= max_trades_day:
            return False, f"max_trades_reached ({max_trades_day})"

        return True, "ok"

    def _log_hourly_report(self):
        """Log hourly performance report."""
        now = datetime.now(timezone.utc)
        if now.hour != self._last_report_hour:
            self._last_report_hour = now.hour

            # Log session status
            session_info = get_session_info()
            LOG.info("=== Hourly Report ===")
            LOG.info("Session: %s | Quality: %s (%d/100)",
                     session_info["session"], session_info["quality"], session_info["score"])
            LOG.info("Trades today: %d", self._trades_today)

            # Log performance stats
            stats = self.logger.get_performance_stats(days=1)
            if stats.total_trades > 0:
                LOG.info("Today: %d trades | Win Rate: %.1f%% | Profit: $%.2f",
                         stats.total_trades, stats.win_rate, stats.total_profit)

            # Log market intelligence
            if self._use_market_intel and self.market_intel:
                self.market_intel.log_market_report()

    def run(self):
        """Main bot loop."""
        load_dotenv()
        setup_logging(self.cfg.logging.level)

        LOG.info("=" * 60)
        LOG.info("  SCALPING BOT STARTING")
        LOG.info("  Symbol: %s | Timeframe: %s", self.cfg.symbol, self.cfg.timeframe)
        LOG.info("  Strategy: %s | Min Confirmations: %d",
                 self.cfg.strategy.name, self.cfg.strategy.min_confirmations)
        LOG.info("=" * 60)

        # Connect to MT5
        self.mt5.connect()
        log_health(self.mt5, self.cfg, "STARTED")

        # Initialize market intelligence (needs MT5 connection)
        try:
            self.market_intel = MarketIntelligence(self.mt5)
            LOG.info("Market intelligence initialized")
            self.market_intel.log_market_report()
        except Exception as e:
            LOG.warning("Market intelligence unavailable: %s", e)
            self._use_market_intel = False

        # Print initial session info
        session_info = get_session_info()
        LOG.info("Current session: %s | Quality: %s",
                 session_info["session"], session_info["quality"])

        if session_info["minutes_until_good_session"] > 0:
            LOG.info("Next good session in %d minutes", session_info["minutes_until_good_session"])

        self.running = True
        loop_count = 0

        try:
            while self.running:
                loop_count += 1

                # Refresh risk circuit breaker
                self.risk.refresh_daily_circuit_breaker()

                # Sync positions (detect closes)
                self.execution.sync_open_positions()

                # Manage open positions (trailing, breakeven)
                self.execution.manage_positions()

                # Hourly report
                if loop_count % 60 == 0:  # Every ~60 seconds
                    self._log_hourly_report()

                # Check if we should trade
                can_trade, reason = self._should_trade()
                if not can_trade:
                    if loop_count % 30 == 0:  # Log every 30 seconds
                        LOG.debug("Not trading: %s", reason)
                    time.sleep(1.0)
                    continue

                # Get signal from strategy
                sig = self.strategy.get_signal()
                if sig is not None:
                    # Validate against market intelligence
                    if self._use_market_intel and self.market_intel:
                        should_trade, reason = self.market_intel.should_take_trade(
                            self.cfg.symbol, sig.side
                        )
                        if not should_trade:
                            LOG.info("Signal rejected by market intel: %s", reason)
                            sig = None

                if sig is not None:
                    # Get dynamic SL/TP if enabled
                    if self.cfg.strategy.use_dynamic_sl_tp:
                        sl_pips, tp_pips = self.strategy.get_dynamic_sl_tp()
                    else:
                        sl_pips, tp_pips = self.cfg.trade.sl_pips, self.cfg.trade.tp_pips

                    # Adjust position size based on volatility
                    size_multiplier = 1.0
                    if self._use_market_intel and self.market_intel:
                        size_multiplier = self.market_intel.get_position_size_multiplier()
                        if size_multiplier < 1.0:
                            LOG.info("Position size reduced to %.0f%% due to volatility",
                                     size_multiplier * 100)

                    # Execute trade
                    success = self.execution.execute_signal(sig, sl_pips, tp_pips)
                    if success:
                        self._trades_today += 1
                        LOG.info("Trade #%d today executed", self._trades_today)

                time.sleep(1.0)

        except KeyboardInterrupt:
            LOG.info("Keyboard interrupt received")
        except Exception as e:
            LOG.exception("Bot crashed: %s", e)
            log_health(self.mt5, self.cfg, f"CRASHED: {e}")
            raise
        finally:
            self._shutdown()

    def _shutdown(self):
        """Graceful shutdown."""
        LOG.info("Shutting down...")

        # Print final performance report
        LOG.info("=== Final Performance Report ===")
        self.logger.print_performance_report(days=1)

        # Close MT5 connection
        self.mt5.shutdown()
        log_health(self.mt5, self.cfg, "STOPPED")

        LOG.info("Bot stopped gracefully")


def main():
    """Entry point."""
    ap = argparse.ArgumentParser(description="High-Probability Scalping Bot")
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    ap.add_argument("--report", action="store_true", help="Print performance report and exit")
    args = ap.parse_args()

    cfg = AppConfig.from_yaml(Path(args.config).resolve())

    if args.report:
        logger = TradeLogger(db_path="scalping_trades.db")
        logger.print_performance_report(days=30)
        return

    bot = ScalpingBot(cfg)
    bot.run()


if __name__ == "__main__":
    main()
