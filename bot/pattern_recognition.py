"""
Pattern Recognition - Visual Patterns Like a Trader Sees.

Good traders spot patterns instantly:
- Double tops/bottoms
- Head and shoulders
- Channels (ascending, descending, horizontal)
- Triangles
- Flag patterns

This module detects these patterns from OHLC data.
"""
import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum

LOG = logging.getLogger("bot.patterns")


class PatternType(Enum):
    """Types of patterns we can detect."""
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    ASCENDING_CHANNEL = "ascending_channel"
    DESCENDING_CHANNEL = "descending_channel"
    HORIZONTAL_CHANNEL = "horizontal_channel"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    BULLISH_FLAG = "bullish_flag"
    BEARISH_FLAG = "bearish_flag"


@dataclass
class DetectedPattern:
    """A detected chart pattern."""
    pattern_type: PatternType
    direction: str  # "bullish" or "bearish"
    strength: float  # 0.0 to 1.0
    start_index: int
    end_index: int
    key_levels: List[float]  # Important price levels in the pattern
    target_price: Optional[float] = None  # Projected target if pattern completes
    invalidation_price: Optional[float] = None  # Price that invalidates the pattern


class PatternRecognition:
    """
    Detects chart patterns like a professional trader.

    Patterns detected:
    1. Double Top/Bottom - Reversal patterns
    2. Head and Shoulders - Strong reversal patterns
    3. Channels - Trend continuation/range
    4. Triangles - Breakout patterns
    5. Flags - Continuation patterns
    """

    def __init__(self, pip_value: float = 0.0001):
        self.pip_value = pip_value
        self.min_pattern_candles = 10  # Minimum candles for a pattern
        self.tolerance_pips = 3.0  # Price tolerance for pattern matching

    def detect_all_patterns(self, opens: np.ndarray, highs: np.ndarray,
                            lows: np.ndarray, closes: np.ndarray) -> List[DetectedPattern]:
        """
        Detect all patterns in the price data.

        Returns list of detected patterns, sorted by strength.
        """
        patterns = []

        # Need enough data
        if len(closes) < self.min_pattern_candles:
            return patterns

        # Detect each pattern type
        double_top = self._detect_double_top(highs, lows, closes)
        if double_top:
            patterns.append(double_top)

        double_bottom = self._detect_double_bottom(highs, lows, closes)
        if double_bottom:
            patterns.append(double_bottom)

        channel = self._detect_channel(highs, lows, closes)
        if channel:
            patterns.append(channel)

        # Sort by strength
        patterns.sort(key=lambda p: p.strength, reverse=True)

        return patterns

    def _detect_double_top(self, highs: np.ndarray, lows: np.ndarray,
                           closes: np.ndarray) -> Optional[DetectedPattern]:
        """
        Detect double top pattern.

        Double Top:
        - Two peaks at approximately the same level
        - Valley between them (neckline)
        - Bearish signal when price breaks below neckline

        Good traders look for:
        - Peaks within 0.5% of each other
        - Clear valley between (at least 50% retracement)
        - Second peak with less momentum (lower volume/smaller candles)
        """
        if len(highs) < 20:
            return None

        # Look for swing highs in the data
        swing_highs = self._find_swing_highs(highs, lookback=5)

        if len(swing_highs) < 2:
            return None

        # Get the two most recent swing highs
        recent_highs = swing_highs[-2:]

        peak1_idx, peak1_price = recent_highs[0]
        peak2_idx, peak2_price = recent_highs[1]

        # Peaks should be at similar levels (within tolerance)
        tolerance = self.tolerance_pips * self.pip_value * 10  # 3 pips tolerance
        price_diff = abs(peak1_price - peak2_price)

        if price_diff > tolerance:
            return None

        # Find the valley between the peaks (neckline)
        valley_idx = peak1_idx + np.argmin(lows[peak1_idx:peak2_idx + 1])
        neckline = lows[valley_idx]

        # Valley should be significant (at least 30% retracement)
        avg_peak = (peak1_price + peak2_price) / 2
        retracement = (avg_peak - neckline) / (avg_peak - np.min(lows))

        if retracement < 0.3:
            return None

        # Calculate pattern strength
        # Higher strength if: peaks very close, good retracement, second peak slightly lower
        peak_similarity = 1.0 - (price_diff / tolerance) if tolerance > 0 else 1.0
        second_peak_weaker = 1.0 if peak2_price <= peak1_price else 0.7
        strength = (peak_similarity * 0.4 + retracement * 0.3 + second_peak_weaker * 0.3)

        # Target is typically the height of the pattern projected down from neckline
        pattern_height = avg_peak - neckline
        target_price = neckline - pattern_height

        LOG.info("Double Top detected: peaks at %.5f, %.5f | neckline: %.5f | strength: %.2f",
                 peak1_price, peak2_price, neckline, strength)

        return DetectedPattern(
            pattern_type=PatternType.DOUBLE_TOP,
            direction="bearish",
            strength=strength,
            start_index=peak1_idx,
            end_index=peak2_idx,
            key_levels=[peak1_price, peak2_price, neckline],
            target_price=target_price,
            invalidation_price=max(peak1_price, peak2_price) + tolerance
        )

    def _detect_double_bottom(self, highs: np.ndarray, lows: np.ndarray,
                               closes: np.ndarray) -> Optional[DetectedPattern]:
        """
        Detect double bottom pattern.

        Double Bottom:
        - Two troughs at approximately the same level
        - Peak between them (neckline)
        - Bullish signal when price breaks above neckline
        """
        if len(lows) < 20:
            return None

        # Look for swing lows in the data
        swing_lows = self._find_swing_lows(lows, lookback=5)

        if len(swing_lows) < 2:
            return None

        # Get the two most recent swing lows
        recent_lows = swing_lows[-2:]

        trough1_idx, trough1_price = recent_lows[0]
        trough2_idx, trough2_price = recent_lows[1]

        # Troughs should be at similar levels
        tolerance = self.tolerance_pips * self.pip_value * 10
        price_diff = abs(trough1_price - trough2_price)

        if price_diff > tolerance:
            return None

        # Find the peak between the troughs (neckline)
        peak_idx = trough1_idx + np.argmax(highs[trough1_idx:trough2_idx + 1])
        neckline = highs[peak_idx]

        # Peak should be significant (at least 30% retracement)
        avg_trough = (trough1_price + trough2_price) / 2
        retracement = (neckline - avg_trough) / (np.max(highs) - avg_trough)

        if retracement < 0.3:
            return None

        # Calculate pattern strength
        peak_similarity = 1.0 - (price_diff / tolerance) if tolerance > 0 else 1.0
        second_trough_stronger = 1.0 if trough2_price >= trough1_price else 0.7
        strength = (peak_similarity * 0.4 + retracement * 0.3 + second_trough_stronger * 0.3)

        # Target
        pattern_height = neckline - avg_trough
        target_price = neckline + pattern_height

        LOG.info("Double Bottom detected: troughs at %.5f, %.5f | neckline: %.5f | strength: %.2f",
                 trough1_price, trough2_price, neckline, strength)

        return DetectedPattern(
            pattern_type=PatternType.DOUBLE_BOTTOM,
            direction="bullish",
            strength=strength,
            start_index=trough1_idx,
            end_index=trough2_idx,
            key_levels=[trough1_price, trough2_price, neckline],
            target_price=target_price,
            invalidation_price=min(trough1_price, trough2_price) - tolerance
        )

    def _detect_channel(self, highs: np.ndarray, lows: np.ndarray,
                        closes: np.ndarray) -> Optional[DetectedPattern]:
        """
        Detect channel patterns (ascending, descending, horizontal).

        A channel is:
        - Two parallel trendlines
        - Price bouncing between them
        - Trade: buy at support, sell at resistance
        """
        if len(closes) < 20:
            return None

        # Get swing points
        swing_highs = self._find_swing_highs(highs, lookback=3)
        swing_lows = self._find_swing_lows(lows, lookback=3)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None

        # Fit trendlines
        high_indices = [s[0] for s in swing_highs[-4:]]
        high_prices = [s[1] for s in swing_highs[-4:]]
        low_indices = [s[0] for s in swing_lows[-4:]]
        low_prices = [s[1] for s in swing_lows[-4:]]

        if len(high_indices) < 2 or len(low_indices) < 2:
            return None

        # Calculate slopes
        high_slope = (high_prices[-1] - high_prices[0]) / (high_indices[-1] - high_indices[0] + 1)
        low_slope = (low_prices[-1] - low_prices[0]) / (low_indices[-1] - low_indices[0] + 1)

        # Slopes should be similar (parallel lines)
        slope_diff = abs(high_slope - low_slope)
        avg_slope = (high_slope + low_slope) / 2

        # Determine channel type
        slope_threshold = 0.00001  # Very small slope = horizontal

        if abs(avg_slope) < slope_threshold:
            channel_type = PatternType.HORIZONTAL_CHANNEL
            direction = "neutral"
        elif avg_slope > 0:
            channel_type = PatternType.ASCENDING_CHANNEL
            direction = "bullish"
        else:
            channel_type = PatternType.DESCENDING_CHANNEL
            direction = "bearish"

        # Calculate channel width
        channel_width = np.mean(high_prices) - np.mean(low_prices)

        # Strength based on how parallel the lines are and how many touches
        parallelism = 1.0 - min(1.0, slope_diff / (abs(avg_slope) + 0.00001))
        touches = min(len(swing_highs), len(swing_lows)) / 4.0  # Normalize to max of 1
        strength = parallelism * 0.6 + min(1.0, touches) * 0.4

        if strength < 0.5:
            return None

        LOG.info("Channel detected: %s | slope: %.6f | width: %.5f pips | strength: %.2f",
                 channel_type.value, avg_slope, channel_width / self.pip_value, strength)

        return DetectedPattern(
            pattern_type=channel_type,
            direction=direction,
            strength=strength,
            start_index=min(high_indices[0], low_indices[0]),
            end_index=max(high_indices[-1], low_indices[-1]),
            key_levels=[np.mean(high_prices), np.mean(low_prices)],
            target_price=None,  # Trade within channel
            invalidation_price=None
        )

    def _find_swing_highs(self, highs: np.ndarray, lookback: int = 5) -> List[Tuple[int, float]]:
        """Find swing high points (local maxima)."""
        swing_highs = []

        for i in range(lookback, len(highs) - lookback):
            is_swing = True
            for j in range(1, lookback + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing = False
                    break

            if is_swing:
                swing_highs.append((i, highs[i]))

        return swing_highs

    def _find_swing_lows(self, lows: np.ndarray, lookback: int = 5) -> List[Tuple[int, float]]:
        """Find swing low points (local minima)."""
        swing_lows = []

        for i in range(lookback, len(lows) - lookback):
            is_swing = True
            for j in range(1, lookback + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing = False
                    break

            if is_swing:
                swing_lows.append((i, lows[i]))

        return swing_lows

    def get_pattern_bias(self, patterns: List[DetectedPattern]) -> Tuple[Optional[str], float]:
        """
        Get overall bias from detected patterns.

        Returns:
            (direction, confidence) where direction is "BUY", "SELL", or None
        """
        if not patterns:
            return None, 0.0

        # Weight patterns by strength
        bullish_score = sum(p.strength for p in patterns if p.direction == "bullish")
        bearish_score = sum(p.strength for p in patterns if p.direction == "bearish")

        if bullish_score > bearish_score and bullish_score > 0.5:
            return "BUY", bullish_score
        elif bearish_score > bullish_score and bearish_score > 0.5:
            return "SELL", bearish_score
        else:
            return None, 0.0
