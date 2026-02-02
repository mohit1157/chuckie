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

LOG = logging.getLogger("bot.price_action")


@dataclass
class Signal:
    side: str  # "BUY" or "SELL"
    reason: str
    entry_price: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    confidence: float = 0.0


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

        # Strategy parameters
        self.swing_lookback = 5  # Candles to confirm a swing point
        self.sr_touch_tolerance_pips = 3.0  # How close price must be to S/R
        self.min_wick_ratio = 0.5  # Minimum wick/body ratio for rejection
        self.min_rr_ratio = 1.5  # Minimum risk:reward ratio

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

        # Log analysis
        LOG.info("Price Action Analysis:")
        LOG.info("  Trend: %s | Momentum: %s", trend, recent_momentum)
        LOG.info("  Nearest Support: %.5f (at_support=%s) | Resistance: %.5f (at_resistance=%s)",
                 nearest_support or 0, at_support, nearest_resistance or 0, at_resistance)
        LOG.info("  Bullish Rejection: %s | Bearish Rejection: %s", bullish_rejection, bearish_rejection)
        LOG.info("  Bullish Engulfing: %s | Bearish Engulfing: %s", bullish_engulfing, bearish_engulfing)

        # Generate signal based on confluence
        signal = None

        # BUY Setup: At support + bullish rejection + trend not strongly bearish
        if at_support and (bullish_rejection or bullish_engulfing):
            if trend != "strong_downtrend":  # Don't buy in strong downtrend
                confidence = self._calculate_confidence(
                    at_level=True,
                    rejection=bullish_rejection,
                    engulfing=bullish_engulfing,
                    trend_aligned=(trend in ["uptrend", "strong_uptrend", "ranging"]),
                    momentum_aligned=(recent_momentum in ["bullish", "neutral"])
                )

                if confidence >= 0.6:  # Minimum 60% confidence
                    sl_price = nearest_support - (self.sr_touch_tolerance_pips * 2 * pip_value)
                    risk_pips = (current_close - sl_price) / pip_value
                    tp_price = current_close + (risk_pips * self.min_rr_ratio * pip_value)

                    reasons = []
                    if at_support:
                        reasons.append("at_support")
                    if bullish_rejection:
                        reasons.append("wick_rejection")
                    if bullish_engulfing:
                        reasons.append("engulfing")
                    if trend in ["uptrend", "strong_uptrend"]:
                        reasons.append("trend_aligned")

                    signal = Signal(
                        side="BUY",
                        reason=f"pa_buy|{'+'.join(reasons)}|conf={confidence:.0%}",
                        entry_price=current_close,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        confidence=confidence
                    )
                    LOG.info("BUY SIGNAL: %s (confidence: %.0f%%)", signal.reason, confidence * 100)

        # SELL Setup: At resistance + bearish rejection + trend not strongly bullish
        elif at_resistance and (bearish_rejection or bearish_engulfing):
            if trend != "strong_uptrend":  # Don't sell in strong uptrend
                confidence = self._calculate_confidence(
                    at_level=True,
                    rejection=bearish_rejection,
                    engulfing=bearish_engulfing,
                    trend_aligned=(trend in ["downtrend", "strong_downtrend", "ranging"]),
                    momentum_aligned=(recent_momentum in ["bearish", "neutral"])
                )

                if confidence >= 0.6:  # Minimum 60% confidence
                    sl_price = nearest_resistance + (self.sr_touch_tolerance_pips * 2 * pip_value)
                    risk_pips = (sl_price - current_close) / pip_value
                    tp_price = current_close - (risk_pips * self.min_rr_ratio * pip_value)

                    reasons = []
                    if at_resistance:
                        reasons.append("at_resistance")
                    if bearish_rejection:
                        reasons.append("wick_rejection")
                    if bearish_engulfing:
                        reasons.append("engulfing")
                    if trend in ["downtrend", "strong_downtrend"]:
                        reasons.append("trend_aligned")

                    signal = Signal(
                        side="SELL",
                        reason=f"pa_sell|{'+'.join(reasons)}|conf={confidence:.0%}",
                        entry_price=current_close,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        confidence=confidence
                    )
                    LOG.info("SELL SIGNAL: %s (confidence: %.0f%%)", signal.reason, confidence * 100)

        # Trend continuation trades (not at S/R but with strong momentum)
        elif trend in ["strong_uptrend"] and recent_momentum == "bullish" and bullish_rejection:
            # Pullback buy in uptrend
            confidence = 0.65
            sl_pips = 15
            tp_pips = sl_pips * self.min_rr_ratio

            signal = Signal(
                side="BUY",
                reason=f"pa_trend_buy|pullback+rejection|conf={confidence:.0%}",
                entry_price=current_close,
                sl_price=current_close - (sl_pips * pip_value),
                tp_price=current_close + (tp_pips * pip_value),
                confidence=confidence
            )
            LOG.info("TREND BUY SIGNAL: %s", signal.reason)

        elif trend in ["strong_downtrend"] and recent_momentum == "bearish" and bearish_rejection:
            # Pullback sell in downtrend
            confidence = 0.65
            sl_pips = 15
            tp_pips = sl_pips * self.min_rr_ratio

            signal = Signal(
                side="SELL",
                reason=f"pa_trend_sell|pullback+rejection|conf={confidence:.0%}",
                entry_price=current_close,
                sl_price=current_close + (sl_pips * pip_value),
                tp_price=current_close - (tp_pips * pip_value),
                confidence=confidence
            )
            LOG.info("TREND SELL SIGNAL: %s", signal.reason)

        if signal:
            self.set_cooldown(minutes=5)

        return signal

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
        Check for bullish rejection (long lower wick).

        Bullish rejection = price went down but buyers pushed it back up
        Shows as candle with long lower wick relative to body.
        """
        # Check last 2 candles
        for i in [-1, -2]:
            body = abs(closes[i] - opens[i])
            lower_wick = min(opens[i], closes[i]) - lows[i]
            upper_wick = highs[i] - max(opens[i], closes[i])

            if body == 0:
                body = 0.00001  # Avoid division by zero

            # Long lower wick + small upper wick = bullish rejection
            if lower_wick > body * self.min_wick_ratio and lower_wick > upper_wick * 1.5:
                return True

        return False

    def _check_bearish_rejection(self, opens: np.ndarray, highs: np.ndarray,
                                  lows: np.ndarray, closes: np.ndarray) -> bool:
        """
        Check for bearish rejection (long upper wick).

        Bearish rejection = price went up but sellers pushed it back down
        Shows as candle with long upper wick relative to body.
        """
        # Check last 2 candles
        for i in [-1, -2]:
            body = abs(closes[i] - opens[i])
            upper_wick = highs[i] - max(opens[i], closes[i])
            lower_wick = min(opens[i], closes[i]) - lows[i]

            if body == 0:
                body = 0.00001

            # Long upper wick + small lower wick = bearish rejection
            if upper_wick > body * self.min_wick_ratio and upper_wick > lower_wick * 1.5:
                return True

        return False

    def _check_bullish_engulfing(self, opens: np.ndarray, highs: np.ndarray,
                                  lows: np.ndarray, closes: np.ndarray) -> bool:
        """
        Check for bullish engulfing pattern.

        Previous candle: bearish (red)
        Current candle: bullish (green) and body engulfs previous body
        """
        # Previous candle must be bearish
        prev_bearish = closes[-2] < opens[-2]
        # Current candle must be bullish
        curr_bullish = closes[-1] > opens[-1]

        if not (prev_bearish and curr_bullish):
            return False

        # Current body must engulf previous body
        curr_body_low = min(opens[-1], closes[-1])
        curr_body_high = max(opens[-1], closes[-1])
        prev_body_low = min(opens[-2], closes[-2])
        prev_body_high = max(opens[-2], closes[-2])

        engulfs = curr_body_low <= prev_body_low and curr_body_high >= prev_body_high

        return engulfs

    def _check_bearish_engulfing(self, opens: np.ndarray, highs: np.ndarray,
                                  lows: np.ndarray, closes: np.ndarray) -> bool:
        """
        Check for bearish engulfing pattern.

        Previous candle: bullish (green)
        Current candle: bearish (red) and body engulfs previous body
        """
        # Previous candle must be bullish
        prev_bullish = closes[-2] > opens[-2]
        # Current candle must be bearish
        curr_bearish = closes[-1] < opens[-1]

        if not (prev_bullish and curr_bearish):
            return False

        # Current body must engulf previous body
        curr_body_low = min(opens[-1], closes[-1])
        curr_body_high = max(opens[-1], closes[-1])
        prev_body_low = min(opens[-2], closes[-2])
        prev_body_high = max(opens[-2], closes[-2])

        engulfs = curr_body_low <= prev_body_low and curr_body_high >= prev_body_high

        return engulfs

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
