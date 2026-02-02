"""
Price Action Trading Strategy.

This strategy thinks like a real trader:
1. Identify key support/resistance levels
2. Wait for price to reach those levels
3. Look for rejection patterns (wicks, engulfing candles)
4. Trade WITH the trend structure, not against it
5. Enter on confirmation, not prediction

No lagging indicators - pure price action.
"""
import logging
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional, List, Tuple

from .config import AppConfig
from .mt5_client import MT5Client
from .data import get_recent_bars
from .pattern_recognition import PatternRecognition, DetectedPattern
from .level_memory import LevelMemory

LOG = logging.getLogger("bot.price_action")


@dataclass
class Signal:
    side: str  # "BUY" or "SELL"
    reason: str
    entry_price: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    confidence: float = 0.0
    trade_type: str = "with_trend"  # "with_trend" or "counter_trend"
    size_multiplier: float = 1.0  # 1.0 for with_trend, 0.4 for counter_trend


@dataclass
class SwingPoint:
    """A swing high or swing low point."""
    price: float
    index: int
    type: str  # "high" or "low"
    strength: int  # How many candles confirm this swing


@dataclass
class CandlePattern:
    """Detected candle pattern."""
    name: str
    direction: str  # "bullish" or "bearish"
    strength: float  # 0-1


class PriceActionStrategy:
    """
    Price Action Strategy - Trade like a professional.

    Core Principles:
    1. BUY at support with bullish rejection
    2. SELL at resistance with bearish rejection
    3. Trade WITH the trend structure
    4. Wait for confirmation, don't predict
    """

    def __init__(self, cfg: AppConfig, mt5c: MT5Client):
        self.cfg = cfg
        self.mt5 = mt5c
        self._cooldown_until: Optional[datetime] = None
        self._last_signal_bar: Optional[int] = None

        # Currency strength bias (set externally by bot)
        self._currency_bias: Optional[str] = None  # "BUY", "SELL", or None
        self._bias_strength: float = 0.0  # How strong is the bias (0-100)

        # Strategy parameters
        self.swing_lookback = 5  # Candles to confirm a swing point
        self.sr_touch_tolerance_pips = 3.0  # How close price must be to S/R
        self.min_wick_ratio = 0.5  # Minimum wick/body ratio for rejection
        self.min_rr_ratio = 1.5  # Minimum risk:reward ratio

        # Adaptive trade parameters
        self.counter_trend_size_multiplier = 0.4  # 40% size for counter-trend
        self.counter_trend_tp_multiplier = 0.5  # Smaller TP for counter-trend (quick scalp)
        self.with_trend_tp_multiplier = 1.5  # Larger TP for with-trend (let it run)

        # Momentum entry parameters (Fix 1)
        self.momentum_entry_enabled = True
        self.momentum_min_bias_strength = 6.0  # Minimum currency bias strength
        self.momentum_min_pattern_confidence = 0.5  # Minimum pattern confidence
        self.momentum_size_multiplier = 0.7  # 70% size for momentum trades (slightly conservative)

        # Pattern invalidation parameters (Fix 4)
        self.pattern_invalidation_pips = 10.0  # Invalidate pattern if price moves X pips beyond it

        # MA Touch Entry parameters (Fix 2)
        self.ma_touch_enabled = True
        self.ma_period = 20  # EMA period for trend identification
        self.ma_touch_tolerance_pips = 3.0  # How close to MA counts as "touch"
        self.ma_touch_size_multiplier = 0.8  # 80% size for MA touch trades

        # Strong Rejection Entry parameters (Fix 3)
        self.strong_rejection_enabled = True
        self.strong_rejection_size_multiplier = 0.5  # 50% size for counter-trend rejections

        # ============================================================
        # FIX 5: Price Action Trend Filter - NEVER fight the obvious trend
        # ============================================================
        # This overrides sentiment/currency strength when chart clearly shows a trend
        self.trend_filter_enabled = True
        self.trend_ema_period = 50  # Use 50 EMA for trend direction
        self.min_ema_distance_pips = 5.0  # Price must be X pips away from EMA to confirm trend
        self.require_higher_lows_for_buy = True  # Must see higher lows to allow BUY
        self.require_lower_highs_for_sell = True  # Must see lower highs to allow SELL

        # Tick-level monitoring (watch forming candle)
        self._forming_candle_open: Optional[float] = None
        self._forming_candle_high: float = 0.0
        self._forming_candle_low: float = float('inf')
        self._tick_count: int = 0
        self._buyer_pressure: float = 0.5  # 0=sellers, 1=buyers
        self._last_tick_price: Optional[float] = None

        # Pattern recognition and level memory
        info = self.mt5.symbol_info(cfg.symbol)
        pip_value = info.point * 10 if info else 0.0001
        self.pattern_recognition = PatternRecognition(pip_value=pip_value)
        self.level_memory = LevelMemory(symbol=cfg.symbol, pip_value=pip_value)
        self._detected_patterns: List[DetectedPattern] = []

    def set_currency_bias(self, direction: Optional[str], strength: float = 0.0):
        """Set the currency strength bias for trade classification."""
        self._currency_bias = direction
        self._bias_strength = strength
        LOG.info("Currency bias set: %s (strength: %.1f)", direction or "NEUTRAL", strength)

    def is_in_cooldown(self) -> bool:
        """Check if strategy is in cooldown."""
        if self._cooldown_until is None:
            return False
        return datetime.now(timezone.utc) < self._cooldown_until

    def set_cooldown(self, minutes: int = 5):
        """Set cooldown period."""
        from datetime import timedelta
        self._cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=minutes)
        LOG.info("Cooldown set for %d minutes", minutes)

    def update_tick(self, price: float):
        """
        Update tick-level analysis for the forming candle.

        This watches how the candle is forming in real-time:
        - Track buyer/seller pressure
        - Detect if price is being rejected at a level
        - See momentum building before candle closes

        Good traders watch the tape, not just the closed candles.
        """
        # Initialize forming candle on first tick
        if self._forming_candle_open is None:
            self._forming_candle_open = price
            self._forming_candle_high = price
            self._forming_candle_low = price

        # Update high/low
        self._forming_candle_high = max(self._forming_candle_high, price)
        self._forming_candle_low = min(self._forming_candle_low, price)
        self._tick_count += 1

        # Calculate buyer/seller pressure based on tick direction
        if self._last_tick_price is not None:
            if price > self._last_tick_price:
                # Uptick - buyers
                self._buyer_pressure = self._buyer_pressure * 0.9 + 0.1
            elif price < self._last_tick_price:
                # Downtick - sellers
                self._buyer_pressure = self._buyer_pressure * 0.9

        self._last_tick_price = price

    def reset_forming_candle(self):
        """Reset forming candle tracking when new candle starts."""
        self._forming_candle_open = None
        self._forming_candle_high = 0.0
        self._forming_candle_low = float('inf')
        self._tick_count = 0
        self._buyer_pressure = 0.5
        self._last_tick_price = None

    def get_forming_candle_analysis(self) -> dict:
        """
        Get analysis of the currently forming candle.

        Returns dict with:
        - direction: "bullish", "bearish", or "doji"
        - wick_developing: "upper", "lower", "both", or "none"
        - buyer_pressure: 0.0 to 1.0
        - tick_count: number of ticks
        """
        if self._forming_candle_open is None:
            return {"direction": "unknown", "wick_developing": "none",
                    "buyer_pressure": 0.5, "tick_count": 0}

        # Current price is last tick
        current_price = self._last_tick_price or self._forming_candle_open
        candle_range = self._forming_candle_high - self._forming_candle_low

        if candle_range == 0:
            return {"direction": "doji", "wick_developing": "none",
                    "buyer_pressure": self._buyer_pressure, "tick_count": self._tick_count}

        # Calculate body
        body_top = max(self._forming_candle_open, current_price)
        body_bottom = min(self._forming_candle_open, current_price)
        body_size = body_top - body_bottom

        # Calculate wicks
        upper_wick = self._forming_candle_high - body_top
        lower_wick = body_bottom - self._forming_candle_low

        # Determine direction
        if current_price > self._forming_candle_open:
            direction = "bullish"
        elif current_price < self._forming_candle_open:
            direction = "bearish"
        else:
            direction = "doji"

        # Determine wick developing
        upper_wick_ratio = upper_wick / candle_range if candle_range > 0 else 0
        lower_wick_ratio = lower_wick / candle_range if candle_range > 0 else 0

        if upper_wick_ratio > 0.3 and lower_wick_ratio > 0.3:
            wick_developing = "both"
        elif upper_wick_ratio > 0.3:
            wick_developing = "upper"
        elif lower_wick_ratio > 0.3:
            wick_developing = "lower"
        else:
            wick_developing = "none"

        return {
            "direction": direction,
            "wick_developing": wick_developing,
            "buyer_pressure": self._buyer_pressure,
            "tick_count": self._tick_count,
            "upper_wick_ratio": upper_wick_ratio,
            "lower_wick_ratio": lower_wick_ratio
        }

    def get_signal(self) -> Optional[Signal]:
        """
        Analyze price action and generate trading signal.

        Logic:
        1. Find recent swing highs/lows (support/resistance)
        2. Check if price is at a key level
        3. Look for rejection pattern (wick rejection, engulfing)
        4. Confirm with trend structure
        5. Generate signal with proper S/L and T/P
        """
        if self.is_in_cooldown():
            return None

        # Get recent price data
        rates = get_recent_bars(self.mt5, self.cfg.symbol, self.cfg.timeframe, n=100)
        if rates is None or len(rates) < 50:
            LOG.warning("Not enough data for price action analysis")
            return None

        # Extract OHLC
        opens = np.array([r["open"] for r in rates], dtype=float)
        highs = np.array([r["high"] for r in rates], dtype=float)
        lows = np.array([r["low"] for r in rates], dtype=float)
        closes = np.array([r["close"] for r in rates], dtype=float)
        times = [r["time"] for r in rates]

        # Get pip value for this symbol
        info = self.mt5.symbol_info(self.cfg.symbol)
        if info is None:
            return None
        pip_value = info.point * 10  # For 5-digit brokers

        # Current price info
        current_close = closes[-1]
        current_high = highs[-1]
        current_low = lows[-1]
        current_open = opens[-1]

        # Step 1: Find swing points (support/resistance levels)
        swing_highs = self._find_swing_highs(highs, lows, closes)
        swing_lows = self._find_swing_lows(highs, lows, closes)

        # Step 2: Identify trend structure
        trend = self._analyze_trend_structure(swing_highs, swing_lows)

        # Step 3: Find nearest support and resistance
        nearest_support = self._find_nearest_support(current_close, swing_lows, pip_value)
        nearest_resistance = self._find_nearest_resistance(current_close, swing_highs, pip_value)

        # Step 4: Check for price at key level
        at_support = self._is_at_level(current_low, nearest_support, pip_value)
        at_resistance = self._is_at_level(current_high, nearest_resistance, pip_value)

        # Step 5: Look for rejection patterns on current/recent candles
        bullish_rejection = self._check_bullish_rejection(opens, highs, lows, closes)
        bearish_rejection = self._check_bearish_rejection(opens, highs, lows, closes)

        # Step 6: Check for engulfing patterns
        bullish_engulfing = self._check_bullish_engulfing(opens, highs, lows, closes)
        bearish_engulfing = self._check_bearish_engulfing(opens, highs, lows, closes)

        # Step 7: Check momentum (are recent candles supporting the move?)
        recent_momentum = self._analyze_momentum(opens, closes)

        # Step 8: Detect chart patterns (double tops, channels, etc.)
        self._detected_patterns = self.pattern_recognition.detect_all_patterns(
            opens, highs, lows, closes
        )

        # FIX 4: Invalidate broken patterns
        self._detected_patterns = self._invalidate_broken_patterns(
            self._detected_patterns, current_close, pip_value
        )

        pattern_bias, pattern_confidence = self.pattern_recognition.get_pattern_bias(
            self._detected_patterns
        )

        # Step 9: Check level history (how did price react here before?)
        support_prediction = None
        resistance_prediction = None
        if nearest_support:
            support_prediction = self.level_memory.predict_reaction(nearest_support, "support")
        if nearest_resistance:
            resistance_prediction = self.level_memory.predict_reaction(nearest_resistance, "resistance")

        # Log analysis
        LOG.info("Price Action Analysis:")
        LOG.info("  Trend: %s | Momentum: %s | Currency Bias: %s", trend, recent_momentum, self._currency_bias or "NEUTRAL")
        LOG.info("  Nearest Support: %.5f (at_support=%s) | Resistance: %.5f (at_resistance=%s)",
                 nearest_support or 0, at_support, nearest_resistance or 0, at_resistance)
        LOG.info("  Bullish Rejection: %s | Bearish Rejection: %s", bullish_rejection, bearish_rejection)
        LOG.info("  Bullish Engulfing: %s | Bearish Engulfing: %s", bullish_engulfing, bearish_engulfing)

        # Log patterns if detected
        if self._detected_patterns:
            for p in self._detected_patterns[:2]:  # Log top 2 patterns
                LOG.info("  Pattern: %s (%s) | strength: %.2f", p.pattern_type.value, p.direction, p.strength)
        if pattern_bias:
            LOG.info("  Pattern Bias: %s (confidence: %.2f)", pattern_bias, pattern_confidence)

        # Log level history if available
        if support_prediction and support_prediction["touch_count"] > 0:
            LOG.info("  Support History: %d touches | likely: %s (%.0f%% conf)",
                     support_prediction["touch_count"], support_prediction["likely_reaction"],
                     support_prediction["confidence"] * 100)
        if resistance_prediction and resistance_prediction["touch_count"] > 0:
            LOG.info("  Resistance History: %d touches | likely: %s (%.0f%% conf)",
                     resistance_prediction["touch_count"], resistance_prediction["likely_reaction"],
                     resistance_prediction["confidence"] * 100)

        # ============================================================
        # FIX 5: TREND FILTER - Don't fight the obvious chart trend!
        # ============================================================
        allowed_direction, chart_trend, trend_reason = self._get_chart_trend_filter(
            closes, highs, lows, pip_value
        )
        LOG.info("  CHART TREND: %s | Allowed: %s | %s", chart_trend, allowed_direction, trend_reason)

        # Generate signal based on confluence
        signal = None

        # BUY Setup: At support + bullish rejection
        # FIX 5: Check trend filter - don't BUY if chart is clearly bearish
        if at_support and (bullish_rejection or bullish_engulfing) and allowed_direction in ["BUY", "BOTH"]:
            # Determine if this is with or against currency bias
            is_with_trend = self._currency_bias in ["BUY", None]

            # For counter-trend, require stronger confirmation
            min_confidence = 0.6 if is_with_trend else 0.7

            confidence = self._calculate_confidence(
                at_level=True,
                rejection=bullish_rejection,
                engulfing=bullish_engulfing,
                trend_aligned=(trend in ["uptrend", "strong_uptrend", "ranging"]),
                momentum_aligned=(recent_momentum in ["bullish", "neutral"])
            )

            # Boost confidence if level history supports bounce
            if support_prediction and support_prediction["likely_reaction"] == "bounce":
                confidence += support_prediction["confidence"] * 0.1
            # Boost if pattern supports direction
            if pattern_bias == "BUY":
                confidence += pattern_confidence * 0.1

            if confidence >= min_confidence:
                sl_price = nearest_support - (self.sr_touch_tolerance_pips * 2 * pip_value)
                risk_pips = (current_close - sl_price) / pip_value

                # Adaptive TP based on trade type
                if is_with_trend:
                    tp_multiplier = self.with_trend_tp_multiplier
                    size_mult = 1.0
                    trade_type = "with_trend"
                else:
                    tp_multiplier = self.counter_trend_tp_multiplier
                    size_mult = self.counter_trend_size_multiplier
                    trade_type = "counter_trend"

                tp_price = current_close + (risk_pips * tp_multiplier * pip_value)

                reasons = ["at_support"]
                if bullish_rejection:
                    reasons.append("wick_rejection")
                if bullish_engulfing:
                    reasons.append("engulfing")
                if is_with_trend:
                    reasons.append("WITH_TREND")
                else:
                    reasons.append("COUNTER_TREND")

                signal = Signal(
                    side="BUY",
                    reason=f"pa_buy|{'+'.join(reasons)}|conf={confidence:.0%}",
                    entry_price=current_close,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    confidence=confidence,
                    trade_type=trade_type,
                    size_multiplier=size_mult
                )
                LOG.info("BUY SIGNAL [%s]: %s (confidence: %.0f%%, size: %.0f%%)",
                         trade_type.upper(), signal.reason, confidence * 100, size_mult * 100)

        # SELL Setup: At resistance + bearish rejection
        # FIX 5: Check trend filter - don't SELL if chart is clearly bullish
        elif at_resistance and (bearish_rejection or bearish_engulfing) and allowed_direction in ["SELL", "BOTH"]:
            # Determine if this is with or against currency bias
            is_with_trend = self._currency_bias in ["SELL", None]

            # For counter-trend, require stronger confirmation
            min_confidence = 0.6 if is_with_trend else 0.7

            confidence = self._calculate_confidence(
                at_level=True,
                rejection=bearish_rejection,
                engulfing=bearish_engulfing,
                trend_aligned=(trend in ["downtrend", "strong_downtrend", "ranging"]),
                momentum_aligned=(recent_momentum in ["bearish", "neutral"])
            )

            # Boost confidence if level history supports bounce
            if resistance_prediction and resistance_prediction["likely_reaction"] == "bounce":
                confidence += resistance_prediction["confidence"] * 0.1
            # Boost if pattern supports direction
            if pattern_bias == "SELL":
                confidence += pattern_confidence * 0.1

            if confidence >= min_confidence:
                sl_price = nearest_resistance + (self.sr_touch_tolerance_pips * 2 * pip_value)
                risk_pips = (sl_price - current_close) / pip_value

                # Adaptive TP based on trade type
                if is_with_trend:
                    tp_multiplier = self.with_trend_tp_multiplier
                    size_mult = 1.0
                    trade_type = "with_trend"
                else:
                    tp_multiplier = self.counter_trend_tp_multiplier
                    size_mult = self.counter_trend_size_multiplier
                    trade_type = "counter_trend"

                tp_price = current_close - (risk_pips * tp_multiplier * pip_value)

                reasons = ["at_resistance"]
                if bearish_rejection:
                    reasons.append("wick_rejection")
                if bearish_engulfing:
                    reasons.append("engulfing")
                if is_with_trend:
                    reasons.append("WITH_TREND")
                else:
                    reasons.append("COUNTER_TREND")

                signal = Signal(
                    side="SELL",
                    reason=f"pa_sell|{'+'.join(reasons)}|conf={confidence:.0%}",
                    entry_price=current_close,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    confidence=confidence,
                    trade_type=trade_type,
                    size_multiplier=size_mult
                )
                LOG.info("SELL SIGNAL [%s]: %s (confidence: %.0f%%, size: %.0f%%)",
                         trade_type.upper(), signal.reason, confidence * 100, size_mult * 100)

        # ============================================================
        # FIX 1: MOMENTUM ENTRY - Trade strong trends without S/R
        # ============================================================
        # If no S/R signal, check for momentum entry
        # FIX 6: Don't momentum BUY at resistance, don't momentum SELL at support
        if signal is None and self.momentum_entry_enabled:
            signal = self._check_momentum_entry(
                trend=trend,
                momentum=recent_momentum,
                pattern_bias=pattern_bias,
                pattern_confidence=pattern_confidence,
                current_close=current_close,
                pip_value=pip_value,
                highs=highs,
                lows=lows,
                at_support=at_support,
                at_resistance=at_resistance
            )

        # ============================================================
        # FIX 2: MA TOUCH ENTRY - Trade pullbacks to moving average
        # ============================================================
        if signal is None and self.ma_touch_enabled:
            signal = self._check_ma_touch_entry(
                closes=closes,
                highs=highs,
                lows=lows,
                current_close=current_close,
                pip_value=pip_value,
                trend=trend,
                momentum=recent_momentum
            )

        # ============================================================
        # FIX 3: STRONG REJECTION ENTRY - Trade V-bottoms/tops
        # ============================================================
        if signal is None and self.strong_rejection_enabled:
            signal = self._check_strong_rejection_entry(
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                current_close=current_close,
                pip_value=pip_value
            )

        # ============================================================
        # FIX 5: FINAL TREND FILTER CHECK - Block any signal that fights the trend
        # ============================================================
        if signal is not None:
            if allowed_direction == "BUY" and signal.side == "SELL":
                LOG.warning("SIGNAL BLOCKED: SELL signal rejected - chart is BULLISH (only BUY allowed)")
                signal = None
            elif allowed_direction == "SELL" and signal.side == "BUY":
                LOG.warning("SIGNAL BLOCKED: BUY signal rejected - chart is BEARISH (only SELL allowed)")
                signal = None

        if signal:
            self.set_cooldown(minutes=5)

            # Record this level touch for future reference
            if signal.side == "BUY" and nearest_support:
                self.level_memory.record_level_touch(
                    price=nearest_support,
                    level_type="support",
                    reaction_type="bounce" if bullish_rejection else "reject",
                    reaction_pips=risk_pips if 'risk_pips' in dir() else 5.0,
                    candles_at_level=1
                )
            elif signal.side == "SELL" and nearest_resistance:
                self.level_memory.record_level_touch(
                    price=nearest_resistance,
                    level_type="resistance",
                    reaction_type="bounce" if bearish_rejection else "reject",
                    reaction_pips=risk_pips if 'risk_pips' in dir() else 5.0,
                    candles_at_level=1
                )

        return signal

    def get_detected_patterns(self) -> List[DetectedPattern]:
        """Get the most recently detected patterns."""
        return self._detected_patterns

    def get_level_report(self):
        """Log a report of known levels."""
        self.level_memory.log_level_report()

    def _find_swing_highs(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> List[SwingPoint]:
        """Find swing high points (resistance levels)."""
        swing_highs = []
        lookback = self.swing_lookback

        for i in range(lookback, len(highs) - lookback):
            # A swing high has lower highs on both sides
            is_swing = True
            for j in range(1, lookback + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing = False
                    break

            if is_swing:
                # Calculate strength based on how much higher this point is
                strength = sum(1 for j in range(1, lookback + 1)
                              if highs[i] > highs[i - j] and highs[i] > highs[i + j])
                swing_highs.append(SwingPoint(price=highs[i], index=i, type="high", strength=strength))

        return swing_highs

    def _find_swing_lows(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> List[SwingPoint]:
        """Find swing low points (support levels)."""
        swing_lows = []
        lookback = self.swing_lookback

        for i in range(lookback, len(lows) - lookback):
            # A swing low has higher lows on both sides
            is_swing = True
            for j in range(1, lookback + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing = False
                    break

            if is_swing:
                strength = sum(1 for j in range(1, lookback + 1)
                              if lows[i] < lows[i - j] and lows[i] < lows[i + j])
                swing_lows.append(SwingPoint(price=lows[i], index=i, type="low", strength=strength))

        return swing_lows

    def _analyze_trend_structure(self, swing_highs: List[SwingPoint], swing_lows: List[SwingPoint]) -> str:
        """
        Analyze trend based on swing structure.

        Uptrend: Higher Highs + Higher Lows
        Downtrend: Lower Highs + Lower Lows
        Ranging: Mixed structure
        """
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return "unknown"

        # Get recent swings (last 4 of each)
        recent_highs = sorted(swing_highs, key=lambda x: x.index)[-4:]
        recent_lows = sorted(swing_lows, key=lambda x: x.index)[-4:]

        # Check for higher highs
        higher_highs = 0
        for i in range(1, len(recent_highs)):
            if recent_highs[i].price > recent_highs[i-1].price:
                higher_highs += 1

        # Check for higher lows
        higher_lows = 0
        for i in range(1, len(recent_lows)):
            if recent_lows[i].price > recent_lows[i-1].price:
                higher_lows += 1

        # Check for lower highs
        lower_highs = 0
        for i in range(1, len(recent_highs)):
            if recent_highs[i].price < recent_highs[i-1].price:
                lower_highs += 1

        # Check for lower lows
        lower_lows = 0
        for i in range(1, len(recent_lows)):
            if recent_lows[i].price < recent_lows[i-1].price:
                lower_lows += 1

        # Determine trend
        hh_hl_score = higher_highs + higher_lows
        lh_ll_score = lower_highs + lower_lows

        if hh_hl_score >= 4:
            return "strong_uptrend"
        elif hh_hl_score >= 2 and lh_ll_score <= 1:
            return "uptrend"
        elif lh_ll_score >= 4:
            return "strong_downtrend"
        elif lh_ll_score >= 2 and hh_hl_score <= 1:
            return "downtrend"
        else:
            return "ranging"

    def _find_nearest_support(self, current_price: float, swing_lows: List[SwingPoint], pip_value: float) -> Optional[float]:
        """Find the nearest support level below current price."""
        supports_below = [s.price for s in swing_lows if s.price < current_price]
        if not supports_below:
            return None
        return max(supports_below)  # Nearest one below

    def _find_nearest_resistance(self, current_price: float, swing_highs: List[SwingPoint], pip_value: float) -> Optional[float]:
        """Find the nearest resistance level above current price."""
        resistances_above = [s.price for s in swing_highs if s.price > current_price]
        if not resistances_above:
            return None
        return min(resistances_above)  # Nearest one above

    def _is_at_level(self, price: float, level: Optional[float], pip_value: float) -> bool:
        """Check if price is at a key level (within tolerance)."""
        if level is None:
            return False
        distance_pips = abs(price - level) / pip_value
        return distance_pips <= self.sr_touch_tolerance_pips

    def _check_bullish_rejection(self, opens: np.ndarray, highs: np.ndarray,
                                  lows: np.ndarray, closes: np.ndarray) -> bool:
        """
        Check for STRONG bullish rejection (long lower wick).

        A good scalper looks for:
        1. Long lower wick (at least 2x the body)
        2. Close near the high of the candle (buyers in control at close)
        3. The candle should be complete (we only check closed candles)

        Bullish rejection = price went down but buyers pushed it back up DECISIVELY
        """
        # Only check the LAST CLOSED candle (index -2, since -1 is forming)
        # This ensures we're not entering on a half-formed wick
        i = -2

        body = abs(closes[i] - opens[i])
        candle_range = highs[i] - lows[i]
        lower_wick = min(opens[i], closes[i]) - lows[i]
        upper_wick = highs[i] - max(opens[i], closes[i])

        if candle_range == 0:
            return False

        # STRICT criteria for a quality rejection:
        # 1. Lower wick must be at least 60% of total candle range
        wick_ratio = lower_wick / candle_range
        if wick_ratio < 0.6:
            return False

        # 2. Close must be in upper 40% of candle (buyers won)
        close_position = (closes[i] - lows[i]) / candle_range
        if close_position < 0.6:
            return False

        # 3. Upper wick should be small (less than 20% of range)
        upper_wick_ratio = upper_wick / candle_range
        if upper_wick_ratio > 0.2:
            return False

        return True

    def _check_bearish_rejection(self, opens: np.ndarray, highs: np.ndarray,
                                  lows: np.ndarray, closes: np.ndarray) -> bool:
        """
        Check for STRONG bearish rejection (long upper wick).

        A good scalper looks for:
        1. Long upper wick (at least 60% of candle range)
        2. Close near the low of the candle (sellers in control at close)
        3. The candle should be complete (we only check closed candles)

        Bearish rejection = price went up but sellers pushed it back down DECISIVELY
        """
        # Only check the LAST CLOSED candle (index -2, since -1 is forming)
        i = -2

        body = abs(closes[i] - opens[i])
        candle_range = highs[i] - lows[i]
        upper_wick = highs[i] - max(opens[i], closes[i])
        lower_wick = min(opens[i], closes[i]) - lows[i]

        if candle_range == 0:
            return False

        # STRICT criteria for a quality rejection:
        # 1. Upper wick must be at least 60% of total candle range
        wick_ratio = upper_wick / candle_range
        if wick_ratio < 0.6:
            return False

        # 2. Close must be in lower 40% of candle (sellers won)
        close_position = (closes[i] - lows[i]) / candle_range
        if close_position > 0.4:
            return False

        # 3. Lower wick should be small (less than 20% of range)
        lower_wick_ratio = lower_wick / candle_range
        if lower_wick_ratio > 0.2:
            return False

        return True

    def _check_bullish_engulfing(self, opens: np.ndarray, highs: np.ndarray,
                                  lows: np.ndarray, closes: np.ndarray) -> bool:
        """
        Check for STRONG bullish engulfing pattern.

        Good scalper criteria:
        1. Previous candle: bearish (red)
        2. Current candle (CLOSED - index -2): bullish and body FULLY engulfs previous
        3. Current candle should be significantly larger than previous
        4. Close should be near the high
        """
        # Check candles -3 (previous) and -2 (engulfing, closed)
        # We use -2 because -1 is still forming
        prev_idx = -3
        curr_idx = -2

        if len(closes) < 4:
            return False

        # Previous candle must be bearish
        prev_bearish = closes[prev_idx] < opens[prev_idx]
        # Current candle must be bullish
        curr_bullish = closes[curr_idx] > opens[curr_idx]

        if not (prev_bearish and curr_bullish):
            return False

        # Calculate bodies
        curr_body = abs(closes[curr_idx] - opens[curr_idx])
        prev_body = abs(closes[prev_idx] - opens[prev_idx])
        curr_range = highs[curr_idx] - lows[curr_idx]

        if prev_body == 0 or curr_range == 0:
            return False

        # Current body must be at least 1.5x previous body (decisive move)
        if curr_body < prev_body * 1.5:
            return False

        # Current body must engulf previous body
        curr_body_low = min(opens[curr_idx], closes[curr_idx])
        curr_body_high = max(opens[curr_idx], closes[curr_idx])
        prev_body_low = min(opens[prev_idx], closes[prev_idx])
        prev_body_high = max(opens[prev_idx], closes[prev_idx])

        engulfs = curr_body_low <= prev_body_low and curr_body_high >= prev_body_high

        if not engulfs:
            return False

        # Close should be in upper 70% of candle range
        close_position = (closes[curr_idx] - lows[curr_idx]) / curr_range
        if close_position < 0.7:
            return False

        return True

    def _check_bearish_engulfing(self, opens: np.ndarray, highs: np.ndarray,
                                  lows: np.ndarray, closes: np.ndarray) -> bool:
        """
        Check for STRONG bearish engulfing pattern.

        Good scalper criteria:
        1. Previous candle: bullish (green)
        2. Current candle (CLOSED - index -2): bearish and body FULLY engulfs previous
        3. Current candle should be significantly larger than previous
        4. Close should be near the low
        """
        # Check candles -3 (previous) and -2 (engulfing, closed)
        # We use -2 because -1 is still forming
        prev_idx = -3
        curr_idx = -2

        if len(closes) < 4:
            return False

        # Previous candle must be bullish
        prev_bullish = closes[prev_idx] > opens[prev_idx]
        # Current candle must be bearish
        curr_bearish = closes[curr_idx] < opens[curr_idx]

        if not (prev_bullish and curr_bearish):
            return False

        # Calculate bodies
        curr_body = abs(closes[curr_idx] - opens[curr_idx])
        prev_body = abs(closes[prev_idx] - opens[prev_idx])
        curr_range = highs[curr_idx] - lows[curr_idx]

        if prev_body == 0 or curr_range == 0:
            return False

        # Current body must be at least 1.5x previous body (decisive move)
        if curr_body < prev_body * 1.5:
            return False

        # Current body must engulf previous body
        curr_body_low = min(opens[curr_idx], closes[curr_idx])
        curr_body_high = max(opens[curr_idx], closes[curr_idx])
        prev_body_low = min(opens[prev_idx], closes[prev_idx])
        prev_body_high = max(opens[prev_idx], closes[prev_idx])

        engulfs = curr_body_low <= prev_body_low and curr_body_high >= prev_body_high

        if not engulfs:
            return False

        # Close should be in lower 30% of candle range (sellers in control)
        close_position = (closes[curr_idx] - lows[curr_idx]) / curr_range
        if close_position > 0.3:
            return False

        return True

    def _analyze_momentum(self, opens: np.ndarray, closes: np.ndarray) -> str:
        """
        Analyze recent momentum based on last 5 candles.

        Returns: "bullish", "bearish", or "neutral"
        """
        # Count bullish vs bearish candles in last 5
        recent = 5
        bullish_count = sum(1 for i in range(-recent, 0) if closes[i] > opens[i])
        bearish_count = sum(1 for i in range(-recent, 0) if closes[i] < opens[i])

        # Also check cumulative move
        cumulative_move = closes[-1] - closes[-recent]

        if bullish_count >= 4 or (bullish_count >= 3 and cumulative_move > 0):
            return "bullish"
        elif bearish_count >= 4 or (bearish_count >= 3 and cumulative_move < 0):
            return "bearish"
        else:
            return "neutral"

    def _calculate_confidence(self, at_level: bool, rejection: bool, engulfing: bool,
                              trend_aligned: bool, momentum_aligned: bool) -> float:
        """Calculate signal confidence based on confluence factors."""
        confidence = 0.0

        if at_level:
            confidence += 0.25  # At key S/R level
        if rejection:
            confidence += 0.25  # Wick rejection pattern
        if engulfing:
            confidence += 0.20  # Engulfing pattern
        if trend_aligned:
            confidence += 0.20  # Trading with trend
        if momentum_aligned:
            confidence += 0.10  # Momentum supports direction

        return min(1.0, confidence)

    def _check_momentum_entry(self, trend: str, momentum: str, pattern_bias: Optional[str],
                               pattern_confidence: float, current_close: float, pip_value: float,
                               highs: np.ndarray, lows: np.ndarray,
                               at_support: bool = False, at_resistance: bool = False) -> Optional[Signal]:
        """
        FIX 1: Momentum Entry - Enter strong trends without requiring S/R level.

        This catches breakdowns/breakouts where:
        - Strong trend is established
        - Currency bias strongly aligns
        - Momentum confirms direction
        - Pattern analysis supports direction

        A good trader sees "everything is bearish" and sells, without waiting for
        price to touch a specific level.

        FIX 6: Don't BUY at resistance, don't SELL at support - these are reversal zones!
        """
        # Check if we have strong enough currency bias
        if self._bias_strength < self.momentum_min_bias_strength:
            return None

        # SELL momentum entry
        # FIX 6: Block SELL at support (support is where price bounces UP)
        if (trend in ["strong_downtrend", "downtrend"] and
            self._currency_bias == "SELL" and
            momentum == "bearish" and
            pattern_bias == "SELL" and
            pattern_confidence >= self.momentum_min_pattern_confidence and
            not at_support):  # FIX 6: Don't sell at support!

            # Calculate ATR-based stops (use recent volatility)
            atr = self._calculate_atr(highs, lows, period=14)
            atr_pips = atr / pip_value

            # Tighter stops for momentum trades (1.5x ATR)
            sl_pips = max(8, min(20, atr_pips * 1.5))
            sl_price = current_close + (sl_pips * pip_value)

            # TP = 2x SL for momentum trades (ride the trend)
            tp_pips = sl_pips * 2.0
            tp_price = current_close - (tp_pips * pip_value)

            confidence = 0.65 + (pattern_confidence * 0.2) + (min(self._bias_strength, 10) / 100)

            LOG.info("MOMENTUM SELL: trend=%s, bias=%.1f, momentum=%s, pattern=%s(%.2f)",
                     trend, self._bias_strength, momentum, pattern_bias, pattern_confidence)

            return Signal(
                side="SELL",
                reason=f"momentum_sell|trend={trend}|bias={self._bias_strength:.1f}|conf={confidence:.0%}",
                entry_price=current_close,
                sl_price=sl_price,
                tp_price=tp_price,
                confidence=confidence,
                trade_type="momentum",
                size_multiplier=self.momentum_size_multiplier
            )

        # BUY momentum entry
        # FIX 6: Block BUY at resistance (resistance is where price bounces DOWN)
        elif (trend in ["strong_uptrend", "uptrend"] and
              self._currency_bias == "BUY" and
              momentum == "bullish" and
              pattern_bias == "BUY" and
              pattern_confidence >= self.momentum_min_pattern_confidence and
              not at_resistance):  # FIX 6: Don't buy at resistance!

            # Calculate ATR-based stops
            atr = self._calculate_atr(highs, lows, period=14)
            atr_pips = atr / pip_value

            # Tighter stops for momentum trades (1.5x ATR)
            sl_pips = max(8, min(20, atr_pips * 1.5))
            sl_price = current_close - (sl_pips * pip_value)

            # TP = 2x SL for momentum trades
            tp_pips = sl_pips * 2.0
            tp_price = current_close + (tp_pips * pip_value)

            confidence = 0.65 + (pattern_confidence * 0.2) + (min(self._bias_strength, 10) / 100)

            LOG.info("MOMENTUM BUY: trend=%s, bias=%.1f, momentum=%s, pattern=%s(%.2f)",
                     trend, self._bias_strength, momentum, pattern_bias, pattern_confidence)

            return Signal(
                side="BUY",
                reason=f"momentum_buy|trend={trend}|bias={self._bias_strength:.1f}|conf={confidence:.0%}",
                entry_price=current_close,
                sl_price=sl_price,
                tp_price=tp_price,
                confidence=confidence,
                trade_type="momentum",
                size_multiplier=self.momentum_size_multiplier
            )

        # FIX 6: Log when momentum entry is blocked due to S/R level
        if at_resistance and self._currency_bias == "BUY" and trend in ["strong_uptrend", "uptrend"]:
            LOG.warning("MOMENTUM BUY BLOCKED: Price at resistance - waiting for breakout or pullback")
        if at_support and self._currency_bias == "SELL" and trend in ["strong_downtrend", "downtrend"]:
            LOG.warning("MOMENTUM SELL BLOCKED: Price at support - waiting for breakdown or pullback")

        return None

    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range for volatility-based stops."""
        if len(highs) < period:
            return (highs[-1] - lows[-1])  # Fallback to current range

        ranges = highs[-period:] - lows[-period:]
        return np.mean(ranges)

    def _invalidate_broken_patterns(self, patterns: List, current_price: float, pip_value: float) -> List:
        """
        FIX 4: Remove patterns that have been broken/invalidated.

        If a double bottom was at 1.1850 but price is now at 1.1830 (20 pips below),
        that double bottom is broken and should not influence signals.
        """
        from .pattern_recognition import PatternType

        valid_patterns = []
        invalidation_distance = self.pattern_invalidation_pips * pip_value

        for pattern in patterns:
            should_keep = True

            # Check double bottom invalidation
            if pattern.pattern_type == PatternType.DOUBLE_BOTTOM:
                # Double bottom is invalidated if price is significantly below the troughs
                if hasattr(pattern, 'key_level') and pattern.key_level:
                    if current_price < pattern.key_level - invalidation_distance:
                        LOG.info("Invalidating broken double bottom at %.5f (price=%.5f)",
                                 pattern.key_level, current_price)
                        should_keep = False

            # Check double top invalidation
            elif pattern.pattern_type == PatternType.DOUBLE_TOP:
                # Double top is invalidated if price is significantly above the peaks
                if hasattr(pattern, 'key_level') and pattern.key_level:
                    if current_price > pattern.key_level + invalidation_distance:
                        LOG.info("Invalidating broken double top at %.5f (price=%.5f)",
                                 pattern.key_level, current_price)
                        should_keep = False

            if should_keep:
                valid_patterns.append(pattern)

        return valid_patterns

    def _get_chart_trend_filter(self, closes: np.ndarray, highs: np.ndarray,
                                 lows: np.ndarray, pip_value: float) -> tuple:
        """
        FIX 5: Price Action Trend Filter - Determine allowed trade direction from ACTUAL chart.

        This is the MOST IMPORTANT filter. It looks at:
        1. Price position relative to 50 EMA (above = bullish, below = bearish)
        2. Recent swing structure (higher lows = bullish, lower highs = bearish)
        3. Recent candle direction (are we making new highs or new lows?)

        Returns:
            (allowed_direction, chart_trend, reason)
            - allowed_direction: "BUY", "SELL", or "BOTH"
            - chart_trend: "bullish", "bearish", or "neutral"
            - reason: explanation string
        """
        if not self.trend_filter_enabled:
            return "BOTH", "neutral", "trend_filter_disabled"

        if len(closes) < self.trend_ema_period + 10:
            return "BOTH", "neutral", "insufficient_data"

        # Calculate 50 EMA for trend direction
        ema50 = self._calculate_ema(closes, self.trend_ema_period)
        current_ema = ema50[-1]
        current_close = closes[-1]

        # Check price position relative to EMA
        ema_distance_pips = (current_close - current_ema) / pip_value
        price_above_ema = ema_distance_pips > self.min_ema_distance_pips
        price_below_ema = ema_distance_pips < -self.min_ema_distance_pips

        # Check recent swing structure (last 20 candles)
        recent_lows = lows[-20:]
        recent_highs = highs[-20:]

        # Count higher lows (bullish structure)
        higher_lows = 0
        for i in range(5, len(recent_lows), 5):  # Check every 5 candles
            if recent_lows[i:i+5].min() > recent_lows[i-5:i].min():
                higher_lows += 1

        # Count lower highs (bearish structure)
        lower_highs = 0
        for i in range(5, len(recent_highs), 5):
            if recent_highs[i:i+5].max() < recent_highs[i-5:i].max():
                lower_highs += 1

        # Check last 5 candles direction
        recent_closes = closes[-5:]
        bullish_candles = sum(1 for i in range(1, len(recent_closes)) if recent_closes[i] > recent_closes[i-1])
        bearish_candles = 5 - bullish_candles

        # Determine chart trend
        bullish_signals = 0
        bearish_signals = 0
        reasons = []

        if price_above_ema:
            bullish_signals += 2
            reasons.append(f"price_above_EMA50(+{ema_distance_pips:.1f}pips)")
        elif price_below_ema:
            bearish_signals += 2
            reasons.append(f"price_below_EMA50({ema_distance_pips:.1f}pips)")

        if higher_lows >= 2:
            bullish_signals += 2
            reasons.append(f"higher_lows({higher_lows})")
        if lower_highs >= 2:
            bearish_signals += 2
            reasons.append(f"lower_highs({lower_highs})")

        if bullish_candles >= 4:
            bullish_signals += 1
            reasons.append(f"recent_bullish({bullish_candles}/5)")
        elif bearish_candles >= 4:
            bearish_signals += 1
            reasons.append(f"recent_bearish({bearish_candles}/5)")

        # Determine allowed direction
        reason_str = " | ".join(reasons) if reasons else "neutral_chart"

        if bullish_signals >= 3 and bearish_signals <= 1:
            LOG.info("TREND FILTER: BULLISH chart - only BUY allowed | %s", reason_str)
            return "BUY", "bullish", reason_str
        elif bearish_signals >= 3 and bullish_signals <= 1:
            LOG.info("TREND FILTER: BEARISH chart - only SELL allowed | %s", reason_str)
            return "SELL", "bearish", reason_str
        else:
            LOG.info("TREND FILTER: NEUTRAL chart - both directions allowed | %s", reason_str)
            return "BOTH", "neutral", reason_str

    def _calculate_ema(self, closes: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        ema = np.zeros_like(closes)
        multiplier = 2.0 / (period + 1)

        # Start with SMA for first value
        ema[period - 1] = np.mean(closes[:period])

        # Calculate EMA for remaining values
        for i in range(period, len(closes)):
            ema[i] = (closes[i] - ema[i - 1]) * multiplier + ema[i - 1]

        return ema

    def _check_ma_touch_entry(self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                               current_close: float, pip_value: float,
                               trend: str, momentum: str) -> Optional[Signal]:
        """
        FIX 2: MA Touch Entry - Enter when price pulls back to moving average in a trend.

        This catches "rally then sell" or "dip then buy" setups where:
        - Price is trending (established by trend structure)
        - Price pulls back to touch the moving average
        - MA acts as dynamic support/resistance

        A good trader sees "price touched the 20 EMA in a downtrend" and sells.
        """
        if len(closes) < self.ma_period + 5:
            return None

        # Calculate EMA
        ema = self._calculate_ema(closes, self.ma_period)
        current_ema = ema[-1]

        # Check if price is touching EMA (within tolerance)
        ema_distance_pips = abs(current_close - current_ema) / pip_value

        if ema_distance_pips > self.ma_touch_tolerance_pips:
            return None  # Not touching EMA

        # SELL: Downtrend + price rallied up to touch EMA from below
        if (trend in ["strong_downtrend", "downtrend"] and
            self._currency_bias == "SELL" and
            current_close <= current_ema):  # At or just below EMA

            # Confirm: Recent candles came UP to EMA (rally into resistance)
            recent_lows = lows[-5:]
            if not all(recent_lows[i] <= recent_lows[i + 1] for i in range(len(recent_lows) - 1)):
                # Price was moving up (higher lows = rally)
                atr = self._calculate_atr(highs, lows, period=14)
                atr_pips = atr / pip_value

                sl_pips = max(6, min(15, atr_pips * 1.2))
                sl_price = current_ema + (sl_pips * pip_value)

                tp_pips = sl_pips * 1.5  # Slightly tighter TP for pullback trades
                tp_price = current_close - (tp_pips * pip_value)

                confidence = 0.60
                if self._bias_strength >= 5.0:
                    confidence += 0.1
                if momentum == "bearish":
                    confidence += 0.05

                LOG.info("MA TOUCH SELL: EMA=%.5f, price=%.5f, distance=%.1f pips",
                         current_ema, current_close, ema_distance_pips)

                return Signal(
                    side="SELL",
                    reason=f"ma_touch_sell|ema={current_ema:.5f}|trend={trend}|conf={confidence:.0%}",
                    entry_price=current_close,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    confidence=confidence,
                    trade_type="ma_touch",
                    size_multiplier=self.ma_touch_size_multiplier
                )

        # BUY: Uptrend + price dipped down to touch EMA from above
        elif (trend in ["strong_uptrend", "uptrend"] and
              self._currency_bias == "BUY" and
              current_close >= current_ema):  # At or just above EMA

            # Confirm: Recent candles came DOWN to EMA (dip into support)
            recent_highs = highs[-5:]
            if not all(recent_highs[i] >= recent_highs[i + 1] for i in range(len(recent_highs) - 1)):
                # Price was moving down (lower highs = dip)
                atr = self._calculate_atr(highs, lows, period=14)
                atr_pips = atr / pip_value

                sl_pips = max(6, min(15, atr_pips * 1.2))
                sl_price = current_ema - (sl_pips * pip_value)

                tp_pips = sl_pips * 1.5
                tp_price = current_close + (tp_pips * pip_value)

                confidence = 0.60
                if self._bias_strength >= 5.0:
                    confidence += 0.1
                if momentum == "bullish":
                    confidence += 0.05

                LOG.info("MA TOUCH BUY: EMA=%.5f, price=%.5f, distance=%.1f pips",
                         current_ema, current_close, ema_distance_pips)

                return Signal(
                    side="BUY",
                    reason=f"ma_touch_buy|ema={current_ema:.5f}|trend={trend}|conf={confidence:.0%}",
                    entry_price=current_close,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    confidence=confidence,
                    trade_type="ma_touch",
                    size_multiplier=self.ma_touch_size_multiplier
                )

        return None

    def _check_strong_rejection_entry(self, opens: np.ndarray, highs: np.ndarray,
                                       lows: np.ndarray, closes: np.ndarray,
                                       current_close: float, pip_value: float) -> Optional[Signal]:
        """
        FIX 3: Strong Rejection Entry - Enter on very strong rejection candles (V-bottoms/tops).

        This catches sharp reversals where:
        - Price makes a sudden spike low/high
        - Very aggressive rejection (wick > 70% of candle)
        - Even without being at a known S/R level

        A good trader sees "massive rejection wick" and knows it's a potential reversal.
        This is counter-trend so we use smaller size.
        """
        # Check the last closed candle
        i = -2

        body = abs(closes[i] - opens[i])
        candle_range = highs[i] - lows[i]
        lower_wick = min(opens[i], closes[i]) - lows[i]
        upper_wick = highs[i] - max(opens[i], closes[i])

        if candle_range == 0:
            return None

        # Calculate ATR for context
        atr = self._calculate_atr(highs, lows, period=14)
        atr_pips = atr / pip_value

        # The rejection candle should be larger than average (significant move)
        if candle_range < atr * 0.8:
            return None  # Not significant enough

        # ============================================================
        # BULLISH STRONG REJECTION (V-bottom)
        # ============================================================
        lower_wick_ratio = lower_wick / candle_range
        if lower_wick_ratio >= 0.70:  # Very strong rejection
            # Close should be in upper portion
            close_position = (closes[i] - lows[i]) / candle_range
            if close_position >= 0.65:
                # Calculate SL below the wick low
                sl_price = lows[i] - (3 * pip_value)  # 3 pip buffer below wick

                risk_pips = (current_close - sl_price) / pip_value
                if risk_pips > 25:
                    return None  # Risk too high

                tp_pips = risk_pips * 1.2  # Conservative TP for counter-trend
                tp_price = current_close + (tp_pips * pip_value)

                confidence = 0.55 + (lower_wick_ratio - 0.70) * 0.5  # Bonus for extra strong wick

                LOG.info("STRONG REJECTION BUY: wick_ratio=%.0f%%, candle_range=%.1f pips",
                         lower_wick_ratio * 100, candle_range / pip_value)

                return Signal(
                    side="BUY",
                    reason=f"strong_rejection_buy|wick={lower_wick_ratio:.0%}|conf={confidence:.0%}",
                    entry_price=current_close,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    confidence=confidence,
                    trade_type="strong_rejection",
                    size_multiplier=self.strong_rejection_size_multiplier
                )

        # ============================================================
        # BEARISH STRONG REJECTION (Inverted V-top)
        # ============================================================
        upper_wick_ratio = upper_wick / candle_range
        if upper_wick_ratio >= 0.70:  # Very strong rejection
            # Close should be in lower portion
            close_position = (closes[i] - lows[i]) / candle_range
            if close_position <= 0.35:
                # Calculate SL above the wick high
                sl_price = highs[i] + (3 * pip_value)  # 3 pip buffer above wick

                risk_pips = (sl_price - current_close) / pip_value
                if risk_pips > 25:
                    return None  # Risk too high

                tp_pips = risk_pips * 1.2  # Conservative TP for counter-trend
                tp_price = current_close - (tp_pips * pip_value)

                confidence = 0.55 + (upper_wick_ratio - 0.70) * 0.5  # Bonus for extra strong wick

                LOG.info("STRONG REJECTION SELL: wick_ratio=%.0f%%, candle_range=%.1f pips",
                         upper_wick_ratio * 100, candle_range / pip_value)

                return Signal(
                    side="SELL",
                    reason=f"strong_rejection_sell|wick={upper_wick_ratio:.0%}|conf={confidence:.0%}",
                    entry_price=current_close,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    confidence=confidence,
                    trade_type="strong_rejection",
                    size_multiplier=self.strong_rejection_size_multiplier
                )

        return None

    def get_dynamic_sl_tp(self) -> Tuple[float, float]:
        """Get SL and TP in pips based on recent volatility."""
        rates = get_recent_bars(self.mt5, self.cfg.symbol, self.cfg.timeframe, n=20)
        if rates is None or len(rates) < 10:
            return 15.0, 22.5  # Default fallback

        highs = np.array([r["high"] for r in rates], dtype=float)
        lows = np.array([r["low"] for r in rates], dtype=float)

        # Calculate average range
        ranges = highs - lows
        avg_range = np.mean(ranges)

        info = self.mt5.symbol_info(self.cfg.symbol)
        if info is None:
            return 15.0, 22.5

        pip_value = info.point * 10
        avg_range_pips = avg_range / pip_value

        # SL = 2x average range (give room to breathe)
        # TP = 1.5x SL (maintain good R:R)
        sl_pips = max(10, min(25, avg_range_pips * 2))
        tp_pips = sl_pips * self.min_rr_ratio

        LOG.info("Dynamic SL/TP: Avg Range=%.1f pips | SL=%.1f pips | TP=%.1f pips",
                 avg_range_pips, sl_pips, tp_pips)

        return sl_pips, tp_pips
