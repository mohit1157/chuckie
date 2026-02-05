"""
High-Probability Scalping Bot - Main Entry Point.

This bot is designed for 80%+ win rate day trading on forex pairs.
Uses multiple confirmations, session filtering, and dynamic risk management.

Features:
- Dynamic pair selection based on currency strength analysis
- Automatic re-evaluation of best pair periodically
- Only trades pairs available in your MT5 account

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
from typing import List, Optional, Tuple

from .config import AppConfig
from .monitoring import setup_logging, log_health
from .mt5_client import MT5Client
from .scalping_execution import ScalpingExecutionEngine
from .scalping_strategy import ScalpingStrategy, ConservativeScalpingStrategy
from .price_action_strategy import PriceActionStrategy, Signal
from .news import NewsFilter
from .risk import RiskManager
from .trade_logger import TradeLogger
from .session_filter import SessionFilter, get_session_info
from .market_intel import MarketIntelligence
from .trade_manager import SmartTradeManager
import logging
import MetaTrader5 as mt5

LOG = logging.getLogger("bot.scalping_main")


class DynamicPairSelector:
    """
    Dynamically selects the best forex pair to trade based on:
    1. Currency strength analysis (35%)
    2. Sentiment alignment (30%)
    3. Volatility suitability (20%)
    4. Spread conditions (15%)
    """

    # Major currencies we analyze
    CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"]

    # Minimum score to trade (0-100)
    MIN_SCORE_TO_TRADE = 60.0

    def __init__(self, mt5_client: MT5Client, market_intel: MarketIntelligence,
                 news_filter: Optional['NewsFilter'] = None, config: Optional[AppConfig] = None):
        self.mt5 = mt5_client
        self.market_intel = market_intel
        self.news_filter = news_filter
        self.cfg = config
        self._available_pairs: List[str] = []
        self._current_pair: Optional[str] = None
        self._current_direction: Optional[str] = None
        self._last_evaluation: Optional[datetime] = None
        self._evaluation_interval_minutes = 5  # Re-evaluate every 5 minutes
        self._has_valid_pair: bool = False  # Track if current pair meets threshold

    def discover_available_pairs(self) -> List[str]:
        """Discover forex pairs available in the MT5 account."""
        symbols = mt5.symbols_get()
        if not symbols:
            LOG.warning("Could not get symbols from MT5")
            return []

        forex_pairs = []
        for s in symbols:
            name = s.name
            # Check if it's a forex pair (6 characters, both parts are currencies)
            if len(name) == 6:
                base = name[:3]
                quote = name[3:]
                if base in self.CURRENCIES and quote in self.CURRENCIES:
                    # Check if symbol exists (visible or has valid spread)
                    # trade_mode may be 0 on weekends, so also check if symbol is valid
                    if s.visible or s.spread > 0 or s.trade_mode > 0:
                        forex_pairs.append(name)

        # If still no pairs found (weekend), use known pairs from your account
        if not forex_pairs:
            LOG.info("Market likely closed - using known forex pairs")
            forex_pairs = ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]

        self._available_pairs = sorted(forex_pairs)
        LOG.info("Discovered %d tradeable forex pairs: %s",
                 len(self._available_pairs), ", ".join(self._available_pairs))
        return self._available_pairs

    def _get_sentiment_score(self, symbol: str, direction: str) -> Tuple[float, str]:
        """Get sentiment score for a pair in given direction."""
        if not self.news_filter:
            return 50.0, "No sentiment data"

        try:
            overall, recommendation = self.news_filter.get_pair_sentiment(symbol)

            if direction == "BUY":
                if recommendation == "bullish":
                    return 100.0, f"Sentiment aligned (bullish {overall:.2f})"
                elif recommendation == "bearish":
                    return 0.0, f"Sentiment disagrees (bearish {overall:.2f})"
                else:
                    return 50.0, f"Sentiment neutral ({overall:.2f})"
            else:  # SELL
                if recommendation == "bearish":
                    return 100.0, f"Sentiment aligned (bearish {overall:.2f})"
                elif recommendation == "bullish":
                    return 0.0, f"Sentiment disagrees (bullish {overall:.2f})"
                else:
                    return 50.0, f"Sentiment neutral ({overall:.2f})"
        except Exception as e:
            LOG.debug("Sentiment error for %s: %s", symbol, e)
            return 50.0, "Sentiment unavailable"

    def _get_volatility_score(self, symbol: str) -> Tuple[float, str]:
        """Get volatility score - ideal is 3-8 pips ATR for scalping."""
        try:
            state, atr = self.market_intel.forex_volatility.get_volatility_state(symbol)
            if state == "normal":
                return 100.0, f"Ideal volatility ({atr:.1f} pips)"
            elif state == "low":
                return 60.0, f"Low volatility ({atr:.1f} pips)"
            elif state == "elevated":
                return 70.0, f"Elevated volatility ({atr:.1f} pips)"
            else:
                return 20.0, f"High volatility ({atr:.1f} pips)"
        except Exception:
            return 50.0, "Volatility unknown"

    def _get_spread_score(self, symbol: str) -> Tuple[float, str, int]:
        """Get spread score based on current spread vs max allowed.

        Returns:
            (score, reason, actual_spread_points)
        """
        try:
            info = self.mt5.symbol_info(symbol)
            if not info:
                return 50.0, "Spread unknown", 0

            spread_points = info.spread
            max_spread = self.cfg.risk.max_spread_points if self.cfg else 30

            if spread_points <= max_spread * 0.5:
                return 100.0, f"Tight spread ({spread_points} pts)", spread_points
            elif spread_points <= max_spread:
                return 70.0, f"Acceptable spread ({spread_points} pts)", spread_points
            else:
                return 0.0, f"Wide spread ({spread_points} pts)", spread_points
        except Exception:
            return 50.0, "Spread unknown", 0

    def get_best_pair(self, force_refresh: bool = False) -> Tuple[Optional[str], Optional[str], str]:
        """
        Get the best pair to trade based on weighted scoring:
        - Strength: 35%
        - Sentiment: 30%
        - Volatility: 20%
        - Spread: 15%

        Returns:
            (pair, direction, reason)
        """
        now = datetime.now(timezone.utc)

        # Check if we need to re-evaluate
        if not force_refresh and self._last_evaluation:
            minutes_since = (now - self._last_evaluation).total_seconds() / 60
            if minutes_since < self._evaluation_interval_minutes and self._current_pair:
                if self._has_valid_pair:
                    return self._current_pair, self._current_direction, "Using cached selection"
                else:
                    return None, None, "No pairs meet threshold (cached)"

        self._last_evaluation = now

        if not self._available_pairs:
            self.discover_available_pairs()

        if not self._available_pairs:
            self._has_valid_pair = False
            return None, None, "No tradeable pairs found"

        # Get currency strength from market intelligence
        context = self.market_intel.get_market_context(force_refresh=True)
        strengths = context.currency_strengths

        if not strengths:
            self._has_valid_pair = False
            return None, None, "Currency strength data unavailable"

        # Log currency strength ranking
        sorted_currencies = sorted(strengths.values(), key=lambda x: x.strength, reverse=True)
        LOG.info("Currency Strength Ranking:")
        for cs in sorted_currencies:
            LOG.info("  %d. %s: %.1f (%s)", cs.rank, cs.currency, cs.strength, cs.trend)

        # Score all pairs
        pair_scores = []
        LOG.info("=" * 60)
        LOG.info("PAIR SCORING (min threshold: %.0f)", self.MIN_SCORE_TO_TRADE)
        LOG.info("%-8s %-4s %6s %6s %6s %6s %6s %6s", "PAIR", "DIR", "TOTAL", "STR", "SENT", "VOL", "SPR_SC", "SPR_PT")
        LOG.info("-" * 60)

        for pair in self._available_pairs:
            base = pair[:3]
            quote = pair[3:]

            base_strength = strengths.get(base)
            quote_strength = strengths.get(quote)

            if not base_strength or not quote_strength:
                continue

            # Calculate strength differential
            diff = base_strength.strength - quote_strength.strength
            abs_diff = abs(diff)

            # Skip if no clear direction
            if abs_diff < 5:
                continue

            # Determine direction
            if diff > 0:
                direction = "BUY"
                strength_score = min(100, abs_diff * 10)  # Scale to 0-100
            else:
                direction = "SELL"
                strength_score = min(100, abs_diff * 10)

            # Get other scores
            sentiment_score, sent_reason = self._get_sentiment_score(pair, direction)
            volatility_score, vol_reason = self._get_volatility_score(pair)
            spread_score, spr_reason, actual_spread = self._get_spread_score(pair)

            # Note: Don't skip wide spread pairs - just give them low score
            # This way we still see them in the log

            # Calculate weighted total
            total_score = (
                strength_score * 0.35 +
                sentiment_score * 0.30 +
                volatility_score * 0.20 +
                spread_score * 0.15
            )

            pair_scores.append({
                'pair': pair,
                'direction': direction,
                'total': total_score,
                'strength': strength_score,
                'sentiment': sentiment_score,
                'volatility': volatility_score,
                'spread': spread_score,
                'actual_spread': actual_spread,
                'reasons': [sent_reason, vol_reason, spr_reason]
            })

        # Sort by total score
        pair_scores.sort(key=lambda x: x['total'], reverse=True)

        # Log top pairs
        for ps in pair_scores[:6]:
            LOG.info("%-8s %-4s %6.1f %6.0f %6.0f %6.0f %6.0f %6d",
                     ps['pair'], ps['direction'], ps['total'],
                     ps['strength'], ps['sentiment'], ps['volatility'], ps['spread'],
                     ps['actual_spread'])
        LOG.info("=" * 60)

        # Select best pair if it meets threshold
        if pair_scores and pair_scores[0]['total'] >= self.MIN_SCORE_TO_TRADE:
            best = pair_scores[0]
            self._current_pair = best['pair']
            self._current_direction = best['direction']
            self._has_valid_pair = True

            reason = f"Score {best['total']:.1f} | " + " | ".join(best['reasons'][:2])
            LOG.info("SELECTED: %s %s (score: %.1f)", best['direction'], best['pair'], best['total'])
            return best['pair'], best['direction'], reason
        else:
            self._has_valid_pair = False
            if pair_scores:
                LOG.info("No pairs meet minimum score of %.0f (best: %s %.1f)",
                         self.MIN_SCORE_TO_TRADE, pair_scores[0]['pair'], pair_scores[0]['total'])
            return None, None, f"No pairs meet threshold ({self.MIN_SCORE_TO_TRADE})"

            return None, None, "No suitable pair found"

    def should_switch_pair(self) -> Tuple[bool, Optional[str], Optional[str], str]:
        """
        Check if we should switch to a different pair.

        Returns:
            (should_switch, new_pair, new_direction, reason)
        """
        old_pair = self._current_pair
        new_pair, new_direction, reason = self.get_best_pair(force_refresh=True)

        if new_pair and new_pair != old_pair:
            return True, new_pair, new_direction, f"Switching from {old_pair} to {new_pair}: {reason}"

        return False, old_pair, new_direction, "Keeping current pair"


class ScalpingBot:
    """
    Main scalping bot orchestrator with dynamic pair selection.

    Features:
    - High-probability multi-confirmation strategy
    - Dynamic pair selection based on currency strength
    - Session-based trading (London/NY focus)
    - ATR-based dynamic SL/TP
    - Trade logging and analytics
    - Automatic position management
    - Daily performance reporting
    """

    def __init__(self, cfg: AppConfig):
        load_dotenv()  # Load env vars early for MT5 credentials
        self.cfg = cfg
        self.running = False
        self._setup_signal_handlers()

        # Initialize components
        self.mt5 = MT5Client()
        self.mt5.connect()  # Connect BEFORE creating RiskManager
        self.risk = RiskManager(cfg, self.mt5)
        self.logger = TradeLogger(db_path="scalping_trades.db")
        self.news_filter = NewsFilter(cfg)
        self.session_filter = SessionFilter()
        self.market_intel = None  # Initialized after MT5 connection
        self.pair_selector = None  # Initialized after market_intel

        # Choose strategy based on config
        strategy_name = getattr(cfg.strategy, 'name', 'high_probability_scalping')

        if strategy_name == "price_action":
            self.strategy = PriceActionStrategy(cfg, self.mt5)
            LOG.info("Using PRICE ACTION strategy (S/R + candle patterns)")
        elif getattr(cfg.strategy, "high_precision_mode", False) or cfg.strategy.min_confirmations >= 6:
            self.strategy = ConservativeScalpingStrategy(cfg, self.mt5)
            LOG.info("Using CONSERVATIVE strategy (6+ confirmations)")
        else:
            self.strategy = ScalpingStrategy(cfg, self.mt5)
            LOG.info("Using STANDARD strategy (5+ confirmations)")

        self.execution = ScalpingExecutionEngine(cfg, self.mt5, self.risk, self.logger)

        # Smart trade manager - manages trades like a professional
        self.trade_manager = SmartTradeManager(self.mt5, cfg)

        # Stats
        self._trades_today = 0
        self._last_report_hour = -1
        self._use_market_intel = True  # Can be disabled
        self._last_pair_check = None
        self._pair_check_interval = 300  # Check every 5 minutes (seconds)

        # Momentum scalping parameters
        self._last_tick_price = None
        self._tick_move_accumulator = 0.0  # Accumulates price movement
        self._momentum_threshold_pips = 1.5  # Minimum momentum to trigger trade
        self._last_candle_time = None

    def _get_dynamic_sl_tp(self, symbol: str) -> tuple:
        """
        Get dynamic SL/TP based on current spread.

        MOMENTUM CATCHING STRATEGY: Tight SL, Big TP
        - Catch big moves early, cut losses quick if wrong
        - Risk/Reward = 1:3 or better (3 pip SL, 10 pip TP)

        Pepperstone has zero/low stop levels - allows tight scalping.
        Tight spread (<1.5 pip): SL=3, TP=10
        Wide spread (>1.5 pip):  SL=4, TP=10
        """
        info = self.mt5.symbol_info(symbol)
        if info is None:
            return self.cfg.trade.sl_pips, self.cfg.trade.tp_pips

        # Calculate spread in pips
        spread_points = info.spread
        spread_pips = spread_points / 10.0  # Convert points to pips

        # MOMENTUM CATCHING: Tight SL, Big TP to catch big moves
        if spread_pips < 1.5:
            # Tight spread - ultra tight stop, big target
            sl_pips = 3.0
            tp_pips = 10.0  # Let winners run!
        else:
            # Wide spread - slightly bigger stop, same target
            sl_pips = 4.0
            tp_pips = 10.0  # Still aim for big moves

        LOG.info("Spread=%.1f pips -> SL=%.1f, TP=%.1f (R:R = 1:%.1f)",
                 spread_pips, sl_pips, tp_pips, tp_pips/sl_pips)

        return sl_pips, tp_pips

    def _detect_momentum_candle(self, symbol: str) -> tuple:
        """
        Detect EARLY momentum - catch moves at the START, not after they complete.

        Key insight: Enter when momentum is BUILDING (0.8-1.5 pips), not after
        a big move completes (2+ pips) when pullback is imminent.

        Returns:
            (has_momentum, direction, candle_size_pips)
        """
        tick = self.mt5.symbol_info_tick(symbol)
        info = self.mt5.symbol_info(symbol)
        if tick is None or info is None:
            return False, None, 0.0

        current_price = (tick.bid + tick.ask) / 2
        pip_value = info.point * 10

        # Get current forming candle AND previous candle
        bars = self.mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 3)
        if bars is None or len(bars) < 3:
            return False, None, 0.0

        current_bar = bars[-1]
        prev_bar = bars[-2]

        candle_open = current_bar['open']
        candle_high = current_bar['high']
        candle_low = current_bar['low']

        prev_close = prev_bar['close']
        prev_range = (prev_bar['high'] - prev_bar['low']) / pip_value

        # Calculate current candle metrics
        current_move_pips = abs(current_price - candle_open) / pip_value
        candle_range_pips = (candle_high - candle_low) / pip_value

        # Determine direction
        if current_price > candle_open:
            direction = "BUY"
        elif current_price < candle_open:
            direction = "SELL"
        else:
            direction = None

        spread_pips = info.spread / 10.0

        # EARLY MOMENTUM DETECTION:
        # - Enter when move is 0.8-1.5 pips (catching early momentum)
        # - Price should be moving AWAY from open (not retracing)
        # - Previous candle should be smaller (this one is accelerating)

        min_early_momentum = 0.8 if spread_pips < 1.5 else 1.2
        max_early_momentum = 2.0 if spread_pips < 1.5 else 2.5  # Don't enter late

        # Check for EARLY momentum (before the big move completes)
        is_early_momentum = (
            current_move_pips >= min_early_momentum and
            current_move_pips <= max_early_momentum and  # Not too late!
            direction is not None and
            candle_range_pips > prev_range * 0.8  # Current candle is expanding
        )

        # Also check: is price at extreme of candle? (momentum still going)
        if direction == "BUY":
            at_extreme = (current_price >= candle_high - (0.3 * pip_value))
        elif direction == "SELL":
            at_extreme = (current_price <= candle_low + (0.3 * pip_value))
        else:
            at_extreme = False

        has_momentum = is_early_momentum and at_extreme

        if has_momentum:
            LOG.info("EARLY MOMENTUM DETECTED: %s %.1f pips (range: 0.8-2.0) - catching move early!",
                     direction, current_move_pips)
        elif current_move_pips > max_early_momentum:
            LOG.debug("Momentum move too late (%.1f pips > %.1f max) - waiting for pullback",
                     current_move_pips, max_early_momentum)

        return has_momentum, direction, current_move_pips

    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers."""
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signal."""
        LOG.info("Shutdown signal received, stopping bot...")
        self.running = False

    def _update_trading_symbol(self, new_symbol: str):
        """Update the trading symbol across all components."""
        old_symbol = self.cfg.symbol
        if new_symbol == old_symbol:
            return

        self.cfg.symbol = new_symbol

        # Recreate strategy with new symbol
        strategy_name = getattr(self.cfg.strategy, 'name', 'high_probability_scalping')
        if strategy_name == "price_action":
            self.strategy = PriceActionStrategy(self.cfg, self.mt5)
        elif getattr(self.cfg.strategy, "high_precision_mode", False) or self.cfg.strategy.min_confirmations >= 6:
            self.strategy = ConservativeScalpingStrategy(self.cfg, self.mt5)
        else:
            self.strategy = ScalpingStrategy(self.cfg, self.mt5)

        # Update execution engine
        self.execution = ScalpingExecutionEngine(self.cfg, self.mt5, self.risk, self.logger)

        # Enable the symbol in MT5 (True = enable in Market Watch)
        import MetaTrader5 as mt5
        if not mt5.symbol_select(new_symbol, True):
            LOG.warning("Failed to enable symbol %s in Market Watch", new_symbol)
        else:
            LOG.info("Symbol %s enabled in Market Watch", new_symbol)

        LOG.info("*** SWITCHED TRADING SYMBOL: %s -> %s ***", old_symbol, new_symbol)

    def _check_and_update_pair(self) -> Tuple[str, Optional[str]]:
        """
        Check if we should switch pairs and update if needed.

        Returns:
            (current_pair, preferred_direction)
        """
        now = time.time()

        # Only check periodically
        if self._last_pair_check and (now - self._last_pair_check) < self._pair_check_interval:
            if self.pair_selector and self.pair_selector._current_pair:
                return self.pair_selector._current_pair, self.pair_selector._current_direction
            return self.cfg.symbol, None

        self._last_pair_check = now

        if not self.pair_selector:
            return self.cfg.symbol, None

        # Check if we have open positions - don't switch if we do
        positions = self.mt5.positions_get(symbol=self.cfg.symbol)
        if positions and len(positions) > 0:
            LOG.debug("Have open positions on %s - not switching pairs", self.cfg.symbol)
            return self.cfg.symbol, self.pair_selector._current_direction

        # Evaluate best pair
        should_switch, new_pair, direction, reason = self.pair_selector.should_switch_pair()

        if should_switch and new_pair:
            LOG.info("Pair selector: %s", reason)
            self._update_trading_symbol(new_pair)

        return new_pair or self.cfg.symbol, direction

    def _should_trade(self) -> tuple:
        """
        Check all conditions for trading.

        Returns:
            (can_trade: bool, reason: str)
        """
        # 0. Check if pair selector has a valid pair (meets threshold)
        if self.pair_selector and not self.pair_selector._has_valid_pair:
            return False, "no_pairs_meet_threshold"

        # 1. Risk manager check
        if not self.risk.can_trade_now():
            return False, "risk_manager_blocked"

        # 2. Session filter - DISABLED per user feedback
        # Good traders can profit in any session with proper price action
        # The real problem was bad entries, not the session
        # session_info = get_session_info()
        # if session_info["score"] < 40:
        #     return False, f"session_quality_low ({session_info['session']})"

        # 3. News filter (if enabled and not stub)
        if self.cfg.news_filter.enabled:
            if self.news_filter.block_new_entries():
                return False, "news_blocked"

        # 4. Market intelligence check (forex volatility too high = don't trade)
        if self._use_market_intel and self.market_intel:
            can_trade, reason = self.market_intel.forex_volatility.should_trade(self.cfg.symbol)
            if not can_trade:
                return False, f"forex_volatility: {reason}"

            # Also check for high market fear
            context = self.market_intel.get_market_context()
            if context.vix_value > 30:
                return False, f"high_vix ({context.vix_value:.1f})"

        # 5. Max trades per day (prevent overtrading)
        max_trades_day = 10
        if self._trades_today >= max_trades_day:
            return False, f"max_trades_reached ({max_trades_day})"

        return True, "ok"

    def _manage_trades_like_pro(self):
        """
        Manage open trades like a professional trader would.

        Uses the SmartTradeManager to:
        - Move to breakeven after 3 pips profit
        - Trail stop aggressively
        - Exit on reversal candles
        - Exit if momentum fades
        - Exit if trade chops too long
        """
        actions = self.trade_manager.manage_all_trades()

        for ticket, action in actions:
            if action.startswith("close_"):
                # Smart manager wants to close this trade
                reason = action.replace("close_", "")
                success = self.trade_manager.close_trade(ticket, reason)
                if success:
                    LOG.info("Smart exit: ticket=%d reason=%s", ticket, reason)
            elif action == "update_sl":
                # SL was updated - already logged by trade manager
                pass

    def _log_hourly_report(self):
        """Log hourly performance report."""
        now = datetime.now(timezone.utc)
        if now.hour != self._last_report_hour:
            self._last_report_hour = now.hour

            # Log session status
            session_info = get_session_info()
            LOG.info("=== Hourly Report ===")
            LOG.info("Current Symbol: %s", self.cfg.symbol)
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
        LOG.info("  SCALPING BOT STARTING (DYNAMIC PAIR SELECTION)")
        LOG.info("  Initial Symbol: %s | Timeframe: %s", self.cfg.symbol, self.cfg.timeframe)
        LOG.info("  Strategy: %s | Min Confirmations: %d",
                 self.cfg.strategy.name, self.cfg.strategy.min_confirmations)
        # FIX 16: Log critical settings to confirm they loaded
        LOG.info("  High Precision Mode: %s", getattr(self.cfg.strategy, 'high_precision_mode', 'NOT SET'))
        LOG.info("  Max Lot Size: %.1f", getattr(self.cfg.risk, 'max_lot_size', 5.0))
        LOG.info("  Strong Rejection: %s", getattr(self.cfg.strategy, 'strong_rejection_enabled', 'NOT SET'))
        LOG.info("=" * 60)

        # Connect to MT5
        self.mt5.connect()
        log_health(self.mt5, self.cfg, "STARTED")

        # Initialize market intelligence (needs MT5 connection)
        try:
            self.market_intel = MarketIntelligence(self.mt5)
            LOG.info("Market intelligence initialized")

            # Initialize dynamic pair selector with sentiment integration
            self.pair_selector = DynamicPairSelector(
                self.mt5, self.market_intel,
                news_filter=self.news_filter,
                config=self.cfg
            )
            self.pair_selector.discover_available_pairs()

            # Get initial best pair
            best_pair, direction, reason = self.pair_selector.get_best_pair()
            if best_pair:
                LOG.info("Initial pair selection: %s %s - %s",
                         direction or "ANALYZE", best_pair, reason)
                self._update_trading_symbol(best_pair)

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

                # Smart trade management - like a professional trader
                self._manage_trades_like_pro()

                # Manage open positions (trailing, breakeven) - legacy
                self.execution.manage_positions()

                # Check and update best pair periodically
                current_pair, preferred_direction = self._check_and_update_pair()

                # Hourly report
                if loop_count % 60 == 0:  # Every ~60 seconds
                    self._log_hourly_report()

                # Check if we should trade
                can_trade, reason = self._should_trade()
                if not can_trade:
                    if loop_count % 30 == 0:  # Log every 30 seconds
                        LOG.info("Not trading: %s", reason)  # Changed to INFO for visibility
                    time.sleep(1.0)
                    continue

                # Set currency bias on strategy before getting signal
                if self._use_market_intel and self.market_intel:
                    bias_direction, bias_strength = self.market_intel.get_currency_bias(self.cfg.symbol)
                    if hasattr(self.strategy, 'set_currency_bias'):
                        self.strategy.set_currency_bias(bias_direction, bias_strength)

                # MOMENTUM SCALPING: First check for momentum candle
                has_momentum, momentum_dir, candle_size = self._detect_momentum_candle(self.cfg.symbol)

                # Get signal from strategy
                sig = self.strategy.get_signal()

                # If strategy gives signal, check if momentum confirms it
                if sig is not None:
                    # Log trade type info
                    trade_type = getattr(sig, 'trade_type', 'unknown')
                    size_mult = getattr(sig, 'size_multiplier', 1.0)
                    LOG.info("Signal: %s %s [%s] - size multiplier: %.0f%%",
                             sig.side, self.cfg.symbol, trade_type.upper(), size_mult * 100)

                    # Check if momentum confirms the signal direction
                    if has_momentum and momentum_dir == sig.side:
                        LOG.info("MOMENTUM CONFIRMS SIGNAL: %s (candle: %.1f pips)", sig.side, candle_size)
                    elif has_momentum and momentum_dir != sig.side:
                        LOG.info("Signal direction (%s) conflicts with momentum (%s) - skipping",
                                 sig.side, momentum_dir)
                        sig = None  # Don't trade against momentum

                    # Validate against market intelligence (for volatility/VIX checks only)
                    if sig and self._use_market_intel and self.market_intel:
                        should_trade, reason = self.market_intel.should_take_trade(
                            self.cfg.symbol, sig.side
                        )
                        if not should_trade:
                            LOG.info("Signal rejected by market intel: %s", reason)
                            sig = None

                # If no strategy signal but strong momentum detected, create momentum signal
                if sig is None and has_momentum:
                    sig = Signal(
                        side=momentum_dir,
                        reason=f"momentum_scalp|size={candle_size:.1f}pips",
                        confidence=0.7,
                        trade_type="momentum_scalp",
                        size_multiplier=0.8  # Slightly smaller for pure momentum trades
                    )
                    LOG.info("PURE MOMENTUM TRADE: %s %s (%.1f pips move)",
                             momentum_dir, self.cfg.symbol, candle_size)

                if sig is not None:
                    # Use dynamic SL/TP based on spread
                    sl_pips, tp_pips = self._get_dynamic_sl_tp(self.cfg.symbol)
                    LOG.info("Dynamic SL/TP: SL=%.1f pips, TP=%.1f pips", sl_pips, tp_pips)

                    # Adjust position size based on volatility
                    size_multiplier = 1.0
                    if self._use_market_intel and self.market_intel:
                        size_multiplier = self.market_intel.get_position_size_multiplier()
                        if size_multiplier < 1.0:
                            LOG.info("Position size reduced to %.0f%% due to volatility",
                                     size_multiplier * 100)

                    # Execute trade
                    success, ticket = self.execution.execute_signal_with_ticket(sig, sl_pips, tp_pips)
                    if success and ticket:
                        self._trades_today += 1
                        LOG.info("Trade #%d today executed on %s", self._trades_today, self.cfg.symbol)

                        # Register with smart trade manager
                        tick = self.mt5.symbol_info_tick(self.cfg.symbol)
                        info = self.mt5.symbol_info(self.cfg.symbol)
                        if tick and info:
                            pip_value = info.point * 10
                            entry_price = tick.ask if sig.side == "BUY" else tick.bid
                            sl_dist = sl_pips * pip_value
                            tp_dist = tp_pips * pip_value
                            sl = entry_price - sl_dist if sig.side == "BUY" else entry_price + sl_dist
                            tp = entry_price + tp_dist if sig.side == "BUY" else entry_price - tp_dist
                            lots = self.risk.calc_lot_size(sl_pips)

                            self.trade_manager.register_trade(
                                ticket=ticket,
                                symbol=self.cfg.symbol,
                                side=sig.side,
                                entry_price=entry_price,
                                sl=sl,
                                tp=tp,
                                volume=lots
                            )

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
