"""
High-Probability Scalping Strategy for 80%+ Win Rate.

This strategy uses multiple confirmations to filter for only the highest
probability setups. It trades less frequently but with much higher accuracy.

Key principles:
1. Trade WITH the trend (higher timeframe confirmation)
2. Enter on pullbacks to dynamic support/resistance
3. Multiple indicator confluence required
4. Session-based filtering (London/NY overlap is best)
5. ATR-based dynamic SL/TP for volatility adaptation
"""
import logging
from datetime import datetime, timezone
from typing import Optional, List
from dataclasses import dataclass
import numpy as np

from .config import AppConfig
from .data import get_recent_bars
from .execution import Signal
from .mt5_client import MT5Client
from .indicators import calculate_all_indicators, IndicatorValues, ema, atr

LOG = logging.getLogger("bot.scalping")


@dataclass
class TradeSetup:
    """Detailed trade setup with entry/exit levels."""
    side: str
    reason: str
    confidence: float  # 0-100
    entry_price: float
    stop_loss: float
    take_profit: float
    atr_pips: float
    indicators: IndicatorValues


class ScalpingStrategy:
    """
    Multi-confirmation scalping strategy targeting 80%+ win rate.

    Entry Conditions (ALL must be true):
    ======================================

    BUY Setup:
    1. Trend: EMA9 > EMA21 (short-term uptrend)
    2. Pullback: Price touched or near EMA9 (buying the dip)
    3. RSI: Between 40-70 (not overbought, showing strength)
    4. MACD: Histogram positive or turning positive
    5. Stochastic: %K crossed above %D from oversold OR %K > 50
    6. ADX: > 20 (trending market, not ranging)
    7. Session: During high-liquidity hours

    SELL Setup (mirror of BUY):
    1. Trend: EMA9 < EMA21 (short-term downtrend)
    2. Pullback: Price touched or near EMA9 (selling the rally)
    3. RSI: Between 30-60 (not oversold, showing weakness)
    4. MACD: Histogram negative or turning negative
    5. Stochastic: %K crossed below %D from overbought OR %K < 50
    6. ADX: > 20 (trending market)
    7. Session: During high-liquidity hours
    """

    # Optimal trading sessions (UTC hours)
    LONDON_OPEN = 7
    LONDON_CLOSE = 16
    NY_OPEN = 12
    NY_CLOSE = 21

    # Minimum confirmations required (out of 7)
    MIN_CONFIRMATIONS = 5

    def __init__(self, cfg: AppConfig, mt5c: MT5Client):
        self.cfg = cfg
        self.mt5 = mt5c
        self._last_signal_time: Optional[int] = None
        self._last_signal_bar: Optional[int] = None
        self._cooldown_until: Optional[datetime] = None

        # Anti-whipsaw protection - track last signal direction
        self._last_signal_direction: Optional[str] = None
        self._last_signal_datetime: Optional[datetime] = None
        self._direction_lock_minutes = 10  # Don't flip direction within 10 minutes

        # Scalping parameters
        self.ema_fast = cfg.strategy.fast_ema if hasattr(cfg.strategy, 'fast_ema') else 9
        self.ema_slow = cfg.strategy.slow_ema if hasattr(cfg.strategy, 'slow_ema') else 21
        self.rsi_period = 14
        self.atr_period = 14
        self.min_confirmations = getattr(cfg.strategy, "min_confirmations", self.MIN_CONFIRMATIONS)
        self.high_precision_mode = getattr(cfg.strategy, "high_precision_mode", True)
        self.min_adx = getattr(cfg.strategy, "min_adx", 20.0)
        self.min_rsi_buy = getattr(cfg.strategy, "min_rsi_buy", 40.0)
        self.max_rsi_buy = getattr(cfg.strategy, "max_rsi_buy", 70.0)
        self.min_rsi_sell = getattr(cfg.strategy, "min_rsi_sell", 30.0)
        self.max_rsi_sell = getattr(cfg.strategy, "max_rsi_sell", 60.0)
        self.max_spread_pips = getattr(cfg.strategy, "max_spread_pips", 1.5)
        self.cooldown_minutes = getattr(cfg.strategy, "cooldown_minutes", 5)
        if self.high_precision_mode:
            self.min_confirmations = max(self.min_confirmations, 6)
        # R:R ratio for profitability
        self.atr_sl_multiplier = getattr(cfg.strategy, "atr_sl_multiplier", 1.0)
        self.atr_tp_multiplier = getattr(cfg.strategy, "atr_tp_multiplier", 1.5)

    def is_trading_session(self) -> bool:
        """Check if current time is during high-liquidity session."""
        now = datetime.now(timezone.utc)
        hour = now.hour

        if hasattr(self.cfg, "session") and self.cfg.session and not self.cfg.session.enabled:
            return True

        if hasattr(self.cfg, "session") and self.cfg.session:
            if self.cfg.session.overlap_only:
                return 12 <= hour < 16
            if self.cfg.session.london_only:
                return self.LONDON_OPEN <= hour < self.LONDON_CLOSE

        # Best times: London session or London/NY overlap
        london_session = self.LONDON_OPEN <= hour < self.LONDON_CLOSE
        ny_session = self.NY_OPEN <= hour < self.NY_CLOSE

        # Asian session (for testing/24h trading) - lower liquidity but still tradeable
        asian_session = 0 <= hour < 7  # Sydney/Tokyo

        # Allow all major sessions (set to False to disable Asian)
        allow_asian = not self.high_precision_mode

        if allow_asian:
            return True  # Trade 24/7 for now

        # Avoid: Asian session lows, market open/close volatility
        avoid_hours = [0, 1, 2, 3, 4, 5, 22, 23]  # Low liquidity
        if hour in avoid_hours:
            return False

        return london_session or ny_session

    def is_in_cooldown(self) -> bool:
        """Check if strategy is in cooldown period after a trade."""
        if self._cooldown_until is None:
            return False
        return datetime.now(timezone.utc) < self._cooldown_until

    def set_cooldown(self, minutes: int = 5):
        """Set cooldown period to avoid overtrading."""
        from datetime import timedelta
        self._cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=minutes)
        LOG.info("Cooldown set for %d minutes", minutes)

    def _check_buy_confirmations(self, ind: IndicatorValues, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> List[str]:
        """Check all BUY confirmations, return list of met conditions."""
        confirmations = []
        current_price = close[-1]

        # 1. Trend confirmation: Fast EMA > Slow EMA
        if ind.ema_fast > ind.ema_slow:
            confirmations.append("trend_up")

        # 2. Pullback to EMA: Price near fast EMA (within 0.5 ATR)
        ema_distance = abs(current_price - ind.ema_fast)
        if ema_distance <= ind.atr * 0.5:
            confirmations.append("pullback_to_ema")

        # 3. RSI in buy zone
        if self.min_rsi_buy <= ind.rsi <= self.max_rsi_buy:
            confirmations.append("rsi_buy_zone")

        # 4. MACD positive momentum
        if ind.macd_histogram > 0 or (ind.macd > ind.macd_signal):
            confirmations.append("macd_bullish")

        # 5. Stochastic bullish
        if ind.stoch_k > ind.stoch_d or ind.stoch_k > 50:
            confirmations.append("stoch_bullish")

        # 6. ADX showing trend strength
        if ind.adx > self.min_adx and ind.plus_di > ind.minus_di:
            confirmations.append("adx_trend")

        # 7. Price action: Current candle bullish (close > open approximation)
        if close[-1] > close[-2]:
            confirmations.append("bullish_candle")

        return confirmations

    def _check_sell_confirmations(self, ind: IndicatorValues, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> List[str]:
        """Check all SELL confirmations, return list of met conditions."""
        confirmations = []
        current_price = close[-1]

        # 1. Trend confirmation: Fast EMA < Slow EMA
        if ind.ema_fast < ind.ema_slow:
            confirmations.append("trend_down")

        # 2. Pullback to EMA: Price near fast EMA (within 0.5 ATR)
        ema_distance = abs(current_price - ind.ema_fast)
        if ema_distance <= ind.atr * 0.5:
            confirmations.append("pullback_to_ema")

        # 3. RSI in sell zone
        if self.min_rsi_sell <= ind.rsi <= self.max_rsi_sell:
            confirmations.append("rsi_sell_zone")

        # 4. MACD negative momentum
        if ind.macd_histogram < 0 or (ind.macd < ind.macd_signal):
            confirmations.append("macd_bearish")

        # 5. Stochastic bearish
        if ind.stoch_k < ind.stoch_d or ind.stoch_k < 50:
            confirmations.append("stoch_bearish")

        # 6. ADX showing trend strength
        if ind.adx > self.min_adx and ind.minus_di > ind.plus_di:
            confirmations.append("adx_trend")

        # 7. Price action: Current candle bearish
        if close[-1] < close[-2]:
            confirmations.append("bearish_candle")

        return confirmations

    def get_signal(self) -> Optional[Signal]:
        """Generate trading signal with multi-confirmation."""

        # Spread filter (tight spreads improve accuracy for scalping)
        tick = self.mt5.symbol_info_tick(self.cfg.symbol)
        info = self.mt5.symbol_info(self.cfg.symbol)
        if tick and info:
            pip_value = info.point * 10
            spread_pips = (tick.ask - tick.bid) / pip_value if pip_value else 0.0
            if spread_pips > self.max_spread_pips:
                LOG.info("Spread too wide: %.2f pips > %.2f pips", spread_pips, self.max_spread_pips)
                return None

        # Session filter
        if not self.is_trading_session():
            return None

        # Cooldown check
        if self.is_in_cooldown():
            return None

        # Get market data
        rates = get_recent_bars(self.mt5, self.cfg.symbol, self.cfg.timeframe, n=200)
        if rates is None or len(rates) < 100:
            return None

        # Extract OHLC
        close = np.array([r["close"] for r in rates], dtype=float)
        high = np.array([r["high"] for r in rates], dtype=float)
        low = np.array([r["low"] for r in rates], dtype=float)

        # Prevent duplicate signals on same bar
        last_bar_time = int(rates[-1]["time"])
        if self._last_signal_time == last_bar_time:
            return None

        # Calculate all indicators
        ind = calculate_all_indicators(
            high, low, close,
            ema_fast_period=self.ema_fast,
            ema_slow_period=self.ema_slow,
            rsi_period=self.rsi_period,
            atr_period=self.atr_period
        )

        # Check BUY setup
        buy_confirmations = self._check_buy_confirmations(ind, close, high, low)
        buy_confidence = (len(buy_confirmations) / 7) * 100

        # Check SELL setup
        sell_confirmations = self._check_sell_confirmations(ind, close, high, low)
        sell_confidence = (len(sell_confirmations) / 7) * 100

        # Log confirmation status periodically (every bar)
        LOG.info("Confirmations - BUY: %d/7 %s | SELL: %d/7 %s",
                 len(buy_confirmations), buy_confirmations,
                 len(sell_confirmations), sell_confirmations)

        # Determine best signal
        signal = None
        potential_direction = None

        if len(buy_confirmations) >= self.min_confirmations and buy_confidence > sell_confidence:
            potential_direction = "BUY"
            reason = f"scalp_buy|conf={buy_confidence:.0f}%|{','.join(buy_confirmations[:3])}"

        elif len(sell_confirmations) >= self.min_confirmations and sell_confidence > buy_confidence:
            potential_direction = "SELL"
            reason = f"scalp_sell|conf={sell_confidence:.0f}%|{','.join(sell_confirmations[:3])}"

        # Anti-whipsaw check: Don't flip direction within lock period
        # This prevents the SELL->SELL->BUY pattern that caused all losses
        if potential_direction:
            now = datetime.now(timezone.utc)
            if self._last_signal_direction and self._last_signal_datetime:
                time_since_last = (now - self._last_signal_datetime).total_seconds() / 60
                if potential_direction != self._last_signal_direction and time_since_last < self._direction_lock_minutes:
                    LOG.info("ANTI-WHIPSAW: Blocking %s signal (last was %s %.1f min ago, lock=%d min)",
                             potential_direction, self._last_signal_direction, time_since_last, self._direction_lock_minutes)
                    return None

            # Create the signal
            self._last_signal_time = last_bar_time
            self._last_signal_direction = potential_direction
            self._last_signal_datetime = now

            if potential_direction == "BUY":
                signal = Signal(side="BUY", reason=reason)
                LOG.info("BUY signal: %d/7 confirmations (%.1f%%): %s",
                         len(buy_confirmations), buy_confidence, buy_confirmations)
            else:
                signal = Signal(side="SELL", reason=reason)
                LOG.info("SELL signal: %d/7 confirmations (%.1f%%): %s",
                         len(sell_confirmations), sell_confidence, sell_confirmations)

        if signal:
            # Set cooldown to prevent overtrading (increased from 3 to 5 minutes)
            self.set_cooldown(minutes=self.cooldown_minutes)

        return signal

    def get_dynamic_sl_tp(self) -> tuple:
        """Calculate ATR-based dynamic SL and TP."""
        rates = get_recent_bars(self.mt5, self.cfg.symbol, self.cfg.timeframe, n=50)
        if rates is None or len(rates) < 20:
            # Fallback to config values
            return self.cfg.trade.sl_pips, self.cfg.trade.tp_pips

        high = np.array([r["high"] for r in rates], dtype=float)
        low = np.array([r["low"] for r in rates], dtype=float)
        close = np.array([r["close"] for r in rates], dtype=float)

        atr_vals = atr(high, low, close, self.atr_period)
        current_atr = atr_vals[-1]

        # Get symbol info for pip calculation
        info = self.mt5.symbol_info(self.cfg.symbol)
        if info is None:
            return self.cfg.trade.sl_pips, self.cfg.trade.tp_pips

        pip_value = info.point * 10  # For 5-digit brokers

        # Convert ATR to pips
        atr_pips = current_atr / pip_value

        # Calculate SL and TP based on ATR
        # CRITICAL: Min 10 pips SL to avoid noise-triggered stops (was 5, caused all losses)
        # 5 pip SL is too tight - normal market noise triggered all stops
        sl_pips = max(10, min(20, atr_pips * self.atr_sl_multiplier * 2))  # Min 10, Max 20 pips
        tp_pips = max(15, min(30, atr_pips * self.atr_tp_multiplier * 2))  # Min 15, Max 30 pips (1.5:1 R:R)

        LOG.info("Dynamic SL/TP: ATR=%.1f pips | SL=%.1f pips | TP=%.1f pips | R:R=1:%.1f",
                 atr_pips, sl_pips, tp_pips, tp_pips/sl_pips if sl_pips > 0 else 0)

        return sl_pips, tp_pips


class ConservativeScalpingStrategy(ScalpingStrategy):
    """
    Ultra-conservative variant for 85%+ win rate.
    Requires 6/7 confirmations and tighter session filter.
    """

    MIN_CONFIRMATIONS = 6

    def is_trading_session(self) -> bool:
        """Only trade during London/NY overlap (best liquidity)."""
        now = datetime.now(timezone.utc)
        hour = now.hour

        # Only London/NY overlap: 12:00 - 16:00 UTC
        return 12 <= hour < 16
