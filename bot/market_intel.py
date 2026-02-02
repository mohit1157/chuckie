"""
Market Intelligence Module.

Gathers broader market context to improve trade filtering:
1. Dollar Index (DXY) - USD strength/weakness
2. VIX - Market volatility/fear
3. Currency Strength - Relative strength of currencies
4. Intermarket Correlations - Gold, bonds, indices
5. COT Data - Institutional positioning
6. Market Regime - Trending vs ranging

This adds VALUE by:
- Filtering out trades against major market forces
- Avoiding trading during high volatility/uncertainty
- Trading in direction of institutional money flow
- Confirming technical signals with fundamental context
"""
import logging
import json
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import urllib.request
import urllib.error

LOG = logging.getLogger("bot.market_intel")


class MarketRegime(Enum):
    """Current market regime."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    RISK_ON = "risk_on"      # Stocks up, safe havens down
    RISK_OFF = "risk_off"    # Stocks down, safe havens up


@dataclass
class CurrencyStrength:
    """Relative strength of a currency."""
    currency: str
    strength: float  # -100 to +100
    trend: str       # "bullish", "bearish", "neutral"
    rank: int        # 1 = strongest, 8 = weakest


@dataclass
class MarketContext:
    """Complete market context for trading decisions."""
    # Dollar Index
    dxy_value: float
    dxy_trend: str  # "bullish", "bearish", "neutral"

    # Volatility
    vix_value: float
    volatility_regime: str  # "low", "normal", "high", "extreme"

    # Market sentiment
    risk_sentiment: str  # "risk_on", "risk_off", "neutral"

    # Currency strengths
    currency_strengths: Dict[str, CurrencyStrength]

    # Overall regime
    regime: MarketRegime

    # Trading recommendation
    favorable_pairs: List[str]
    avoid_pairs: List[str]

    # Timestamp
    updated_at: datetime


class CurrencyStrengthMeter:
    """
    Calculate relative strength of major currencies.

    Compares each currency against all others to determine
    which are strongest and weakest.

    Trading Logic:
    - BUY: Strong currency vs Weak currency
    - SELL: Weak currency vs Strong currency
    - AVOID: Two currencies with similar strength
    """

    CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"]

    # Major pairs for strength calculation
    PAIRS = {
        "USD": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"],
        "EUR": ["EURUSD", "EURGBP", "EURJPY", "EURAUD", "EURNZD", "EURCAD", "EURCHF"],
        "GBP": ["GBPUSD", "EURGBP", "GBPJPY", "GBPAUD", "GBPNZD", "GBPCAD", "GBPCHF"],
        "JPY": ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"],
        "AUD": ["AUDUSD", "EURAUD", "GBPAUD", "AUDJPY", "AUDNZD", "AUDCAD", "AUDCHF"],
        "NZD": ["NZDUSD", "EURNZD", "GBPNZD", "NZDJPY", "AUDNZD", "NZDCAD", "NZDCHF"],
        "CAD": ["USDCAD", "EURCAD", "GBPCAD", "CADJPY", "AUDCAD", "NZDCAD", "CADCHF"],
        "CHF": ["USDCHF", "EURCHF", "GBPCHF", "CHFJPY", "AUDCHF", "NZDCHF", "CADCHF"],
    }

    def __init__(self, mt5_client=None):
        self.mt5 = mt5_client
        self._strengths: Dict[str, CurrencyStrength] = {}
        self._last_update: Optional[datetime] = None

    def calculate_strength(self) -> Dict[str, CurrencyStrength]:
        """
        Calculate currency strength from MT5 price data.

        Returns dict of currency -> CurrencyStrength
        """
        if self.mt5 is None:
            LOG.debug("MT5 client not available for currency strength")
            return self._get_default_strengths()

        try:
            # Get price changes for all pairs
            pair_changes = {}
            for currency, pairs in self.PAIRS.items():
                for pair in pairs:
                    if pair not in pair_changes:
                        change = self._get_pair_change(pair)
                        if change is not None:
                            pair_changes[pair] = change

            # Calculate strength for each currency
            strengths = {}
            for currency in self.CURRENCIES:
                strength = self._calculate_currency_strength(currency, pair_changes)
                trend = "bullish" if strength > 20 else "bearish" if strength < -20 else "neutral"
                strengths[currency] = CurrencyStrength(
                    currency=currency,
                    strength=strength,
                    trend=trend,
                    rank=0  # Will be set after sorting
                )

            # Rank currencies
            sorted_currencies = sorted(strengths.values(), key=lambda x: x.strength, reverse=True)
            for i, cs in enumerate(sorted_currencies):
                strengths[cs.currency].rank = i + 1

            self._strengths = strengths
            self._last_update = datetime.now(timezone.utc)

            return strengths

        except Exception as e:
            LOG.warning("Failed to calculate currency strength: %s", e)
            return self._get_default_strengths()

    def _get_pair_change(self, pair: str, periods: int = 20) -> Optional[float]:
        """Get percentage change for a pair over N periods."""
        try:
            import MetaTrader5 as mt5
            rates = self.mt5.copy_rates_from_pos(pair, mt5.TIMEFRAME_H1, 0, periods + 1)
            if rates is None or len(rates) < periods:
                return None

            old_close = rates[0]["close"]
            new_close = rates[-1]["close"]

            if old_close == 0:
                return None

            return ((new_close - old_close) / old_close) * 100
        except:
            return None

    def _calculate_currency_strength(self, currency: str, pair_changes: Dict[str, float]) -> float:
        """Calculate strength score for a single currency."""
        total_strength = 0
        count = 0

        for pair in self.PAIRS.get(currency, []):
            if pair not in pair_changes:
                continue

            change = pair_changes[pair]

            # Determine if currency is base or quote
            if pair.startswith(currency):
                # Currency is base - positive change = currency strength
                total_strength += change
            else:
                # Currency is quote - negative change = currency strength
                total_strength -= change

            count += 1

        if count == 0:
            return 0

        # Normalize to -100 to +100 scale
        avg = total_strength / count
        return max(-100, min(100, avg * 10))

    def _get_default_strengths(self) -> Dict[str, CurrencyStrength]:
        """Return neutral strengths when calculation fails."""
        return {
            c: CurrencyStrength(currency=c, strength=0, trend="neutral", rank=4)
            for c in self.CURRENCIES
        }

    def get_best_pair(self) -> Tuple[str, str, str]:
        """
        Get the best pair to trade based on currency strength.

        Returns:
            (pair, direction, reason)
            e.g., ("EURUSD", "SELL", "EUR weak, USD strong")
        """
        if not self._strengths:
            self.calculate_strength()

        if not self._strengths:
            return "", "", "No data"

        # Find strongest and weakest
        sorted_strengths = sorted(self._strengths.values(), key=lambda x: x.strength, reverse=True)
        strongest = sorted_strengths[0]
        weakest = sorted_strengths[-1]

        # Minimum strength difference required
        if abs(strongest.strength - weakest.strength) < 30:
            return "", "", "No clear strength differential"

        # Construct pair
        pair = f"{strongest.currency}{weakest.currency}"
        reverse_pair = f"{weakest.currency}{strongest.currency}"

        # Check which pair exists (simplified - in practice check MT5)
        common_pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
                        "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURAUD", "GBPAUD"]

        if pair in common_pairs:
            return pair, "BUY", f"{strongest.currency} strong ({strongest.strength:.0f}), {weakest.currency} weak ({weakest.strength:.0f})"
        elif reverse_pair in common_pairs:
            return reverse_pair, "SELL", f"{weakest.currency} weak, {strongest.currency} strong"

        return "", "", "No tradeable pair found"


class VIXMonitor:
    """
    Monitor VIX (Volatility Index) for market sentiment.

    NOTE: VIX measures S&P 500 volatility, not forex directly.
    Use ForexVolatilityMonitor for pair-specific volatility.

    VIX correlation with forex:
    - High VIX = Risk-off = USD/JPY/CHF strengthen
    - Low VIX = Risk-on = AUD/NZD/EM currencies strengthen

    VIX Levels:
    - 0-15:  Low volatility (risk-on environment)
    - 15-20: Normal
    - 20-25: Elevated (caution, wider spreads)
    - 25-30: High fear (reduce positions)
    - 30+:   Very high fear (avoid scalping)
    """

    def __init__(self, finnhub_key: str = None):
        self.api_key = finnhub_key or os.getenv("FINNHUB_API_KEY")
        self._vix: float = 15.0  # Default normal level
        self._last_fetch: Optional[datetime] = None

    def fetch_vix(self) -> float:
        """Fetch current VIX value."""
        # Check cache (update every 5 minutes)
        now = datetime.now(timezone.utc)
        if self._last_fetch and (now - self._last_fetch).seconds < 300:
            return self._vix

        if not self.api_key:
            LOG.debug("Finnhub API key not set, using default VIX")
            return self._vix

        try:
            url = f"https://finnhub.io/api/v1/quote?symbol=VIX&token={self.api_key}"
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read().decode())
                self._vix = float(data.get("c", 15.0))
                self._last_fetch = now
                LOG.info("VIX updated: %.2f", self._vix)
        except Exception as e:
            LOG.warning("Failed to fetch VIX: %s", e)

        return self._vix

    def get_volatility_regime(self) -> Tuple[str, bool]:
        """
        Get current volatility regime and trading recommendation.

        Returns:
            (regime, should_trade)
        """
        vix = self.fetch_vix()

        if vix < 15:
            return "low", True
        elif vix < 20:
            return "normal", True
        elif vix < 25:
            return "elevated", True  # Trade with caution
        elif vix < 30:
            return "high", True  # Reduce position size
        else:
            return "very_high", False  # Avoid scalping

    def get_risk_sentiment(self) -> str:
        """
        Get risk sentiment based on VIX.

        Returns:
            "risk_on", "neutral", or "risk_off"
        """
        vix = self.fetch_vix()

        if vix < 15:
            return "risk_on"   # Favor AUD, NZD, risk currencies
        elif vix < 25:
            return "neutral"
        else:
            return "risk_off"  # Favor USD, JPY, CHF

    def should_reduce_size(self) -> Tuple[bool, float]:
        """
        Check if position size should be reduced based on VIX.

        Returns:
            (should_reduce, multiplier)
        """
        vix = self.fetch_vix()

        if vix < 20:
            return False, 1.0
        elif vix < 25:
            return True, 0.75  # 75% of normal size
        elif vix < 30:
            return True, 0.5   # 50% of normal size
        else:
            return True, 0.0   # Don't trade


class ForexVolatilityMonitor:
    """
    Forex-specific volatility monitor using ATR.

    This is MORE relevant than VIX for forex trading because:
    1. Measures actual pip movement of the pair
    2. Adapts to each currency pair's characteristics
    3. Real-time from MT5 data

    ATR Interpretation for EURUSD (M5 timeframe):
    - < 3 pips: Very low volatility (tight range)
    - 3-6 pips: Normal volatility (good for scalping)
    - 6-10 pips: Elevated volatility (widen SL/TP)
    - > 10 pips: High volatility (reduce size or skip)

    Note: These thresholds vary by pair and timeframe.
    """

    # Normal ATR ranges for major pairs (M5 timeframe, in pips)
    NORMAL_ATR_RANGES = {
        "EURUSD": (3, 8),
        "GBPUSD": (4, 10),
        "USDJPY": (3, 8),
        "USDCHF": (3, 7),
        "AUDUSD": (3, 7),
        "USDCAD": (3, 7),
        "NZDUSD": (2, 6),
        "EURGBP": (2, 5),
        "EURJPY": (4, 10),
        "GBPJPY": (6, 15),  # Very volatile pair
    }

    def __init__(self, mt5_client=None):
        self.mt5 = mt5_client
        self._atr_cache: Dict[str, Tuple[float, datetime]] = {}

    def get_pair_atr(self, symbol: str, timeframe: str = "M5", period: int = 14) -> float:
        """
        Get ATR for a specific pair in pips.

        Returns:
            ATR value in pips
        """
        if self.mt5 is None:
            return 5.0  # Default normal value

        # Check cache (valid for 5 minutes)
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self._atr_cache:
            atr, cached_time = self._atr_cache[cache_key]
            if (datetime.now(timezone.utc) - cached_time).seconds < 300:
                return atr

        try:
            import MetaTrader5 as mt5

            # Map timeframe string to MT5 constant
            tf_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "H1": mt5.TIMEFRAME_H1,
            }
            tf = tf_map.get(timeframe, mt5.TIMEFRAME_M5)

            rates = self.mt5.copy_rates_from_pos(symbol, tf, 0, period + 10)
            if rates is None or len(rates) < period:
                return 5.0

            # Calculate ATR
            import numpy as np
            high = np.array([r["high"] for r in rates], dtype=float)
            low = np.array([r["low"] for r in rates], dtype=float)
            close = np.array([r["close"] for r in rates], dtype=float)

            tr = np.zeros(len(close))
            tr[0] = high[0] - low[0]
            for i in range(1, len(close)):
                hl = high[i] - low[i]
                hc = abs(high[i] - close[i-1])
                lc = abs(low[i] - close[i-1])
                tr[i] = max(hl, hc, lc)

            atr = np.mean(tr[-period:])

            # Convert to pips
            info = self.mt5.symbol_info(symbol)
            if info:
                pip_value = info.point * 10  # 5-digit broker
                atr_pips = atr / pip_value
            else:
                atr_pips = atr * 10000  # Assume 4-digit

            # Cache result
            self._atr_cache[cache_key] = (atr_pips, datetime.now(timezone.utc))

            return atr_pips

        except Exception as e:
            LOG.warning("Failed to calculate ATR for %s: %s", symbol, e)
            return 5.0

    def get_volatility_state(self, symbol: str) -> Tuple[str, float]:
        """
        Get volatility state for a pair.

        Returns:
            (state, atr_pips)
            state: "low", "normal", "elevated", "high"
        """
        atr = self.get_pair_atr(symbol)

        # Get normal range for this pair
        normal_range = self.NORMAL_ATR_RANGES.get(symbol, (3, 8))
        low_threshold, high_threshold = normal_range

        if atr < low_threshold * 0.7:
            return "low", atr
        elif atr < high_threshold:
            return "normal", atr
        elif atr < high_threshold * 1.5:
            return "elevated", atr
        else:
            return "high", atr

    def should_trade(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if volatility is suitable for scalping.

        Returns:
            (should_trade, reason)
        """
        state, atr = self.get_volatility_state(symbol)

        if state == "low":
            return True, f"Low volatility ({atr:.1f} pips) - tight scalps OK"
        elif state == "normal":
            return True, f"Normal volatility ({atr:.1f} pips) - ideal for scalping"
        elif state == "elevated":
            return True, f"Elevated volatility ({atr:.1f} pips) - widen SL/TP"
        else:
            return False, f"High volatility ({atr:.1f} pips) - skip scalping"

    def get_recommended_sl_tp(self, symbol: str) -> Tuple[float, float]:
        """
        Get recommended SL/TP based on current volatility.

        Returns:
            (sl_pips, tp_pips)
        """
        atr = self.get_pair_atr(symbol)

        # SL = 1.5 * ATR, TP = 1.0 * ATR (for high win rate)
        sl_pips = max(5, min(15, atr * 1.5))
        tp_pips = max(3, min(10, atr * 1.0))

        return sl_pips, tp_pips


class DollarIndexTracker:
    """
    Track DXY (Dollar Index) for USD strength.

    DXY measures USD against basket of currencies:
    - EUR (57.6%), JPY (13.6%), GBP (11.9%), CAD (9.1%), SEK (4.2%), CHF (3.6%)

    Trading Logic:
    - DXY rising: Favor USD longs (short EURUSD, long USDJPY)
    - DXY falling: Favor USD shorts (long EURUSD, short USDJPY)
    - DXY ranging: Use other confirmations
    """

    def __init__(self, mt5_client=None):
        self.mt5 = mt5_client
        self._dxy: float = 100.0
        self._dxy_change: float = 0.0
        self._last_fetch: Optional[datetime] = None

    def get_dxy_bias(self) -> Tuple[str, float]:
        """
        Get DXY trend and bias.

        Returns:
            (trend, change_percent)
            trend: "bullish", "bearish", "neutral"
        """
        # In practice, you'd fetch DXY from a data source
        # Here we calculate a synthetic DXY from EURUSD

        if self.mt5 is None:
            return "neutral", 0.0

        try:
            import MetaTrader5 as mt5
            rates = self.mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H4, 0, 20)
            if rates is None or len(rates) < 20:
                return "neutral", 0.0

            # EURUSD inverse ≈ DXY direction (simplified)
            old_close = rates[0]["close"]
            new_close = rates[-1]["close"]

            if old_close == 0:
                return "neutral", 0.0

            eur_change = ((new_close - old_close) / old_close) * 100

            # DXY moves opposite to EURUSD (since EUR is 57.6% of DXY)
            dxy_change = -eur_change * 0.576

            if dxy_change > 0.3:
                return "bullish", dxy_change
            elif dxy_change < -0.3:
                return "bearish", dxy_change
            else:
                return "neutral", dxy_change

        except Exception as e:
            LOG.warning("Failed to calculate DXY bias: %s", e)
            return "neutral", 0.0


class MarketIntelligence:
    """
    Central market intelligence hub.

    Combines all market data sources to provide:
    1. Trade filtering (should we trade at all?)
    2. Direction bias (favor longs or shorts?)
    3. Pair selection (which pair has best setup?)
    4. Position sizing (full size or reduced?)

    Volatility monitoring:
    - VIX: General market fear (affects risk sentiment)
    - Forex ATR: Pair-specific volatility (more important for forex)

    Usage:
        intel = MarketIntelligence(mt5_client)

        # Before taking a trade
        context = intel.get_market_context()

        if not context.should_trade:
            return  # Skip trade

        if signal.side == "BUY" and context.usd_bias == "bullish" and "USD" in pair[3:]:
            # Buying a pair where USD is quote, but USD is strong
            return  # Skip - trading against USD strength

        # Adjust position size based on volatility
        lots = lots * context.size_multiplier
    """

    def __init__(self, mt5_client=None, finnhub_key: str = None):
        self.mt5 = mt5_client
        self.currency_strength = CurrencyStrengthMeter(mt5_client)
        self.vix_monitor = VIXMonitor(finnhub_key)
        self.forex_volatility = ForexVolatilityMonitor(mt5_client)
        self.dxy_tracker = DollarIndexTracker(mt5_client)
        self._context: Optional[MarketContext] = None
        self._last_update: Optional[datetime] = None

    def get_market_context(self, force_refresh: bool = False) -> MarketContext:
        """Get current market context."""
        now = datetime.now(timezone.utc)

        # Refresh every 15 minutes unless forced
        if not force_refresh and self._context and self._last_update:
            if (now - self._last_update).seconds < 900:
                return self._context

        # Gather all data
        currency_strengths = self.currency_strength.calculate_strength()
        vix = self.vix_monitor.fetch_vix()
        vol_regime, can_trade_vol = self.vix_monitor.get_volatility_regime()
        dxy_trend, dxy_change = self.dxy_tracker.get_dxy_bias()

        # Determine overall regime
        if vix > 30:
            regime = MarketRegime.HIGH_VOLATILITY
        elif vix < 12:
            regime = MarketRegime.LOW_VOLATILITY
        elif dxy_trend == "bullish":
            regime = MarketRegime.TRENDING_UP  # Risk-off usually
        elif dxy_trend == "bearish":
            regime = MarketRegime.TRENDING_DOWN  # Risk-on usually
        else:
            regime = MarketRegime.RANGING

        # Determine risk sentiment
        if dxy_trend == "bearish" and vix < 20:
            risk_sentiment = "risk_on"
        elif dxy_trend == "bullish" and vix > 20:
            risk_sentiment = "risk_off"
        else:
            risk_sentiment = "neutral"

        # Find favorable and unfavorable pairs
        favorable = []
        avoid = []

        if dxy_trend == "bullish":
            favorable.extend(["USDJPY", "USDCAD", "USDCHF"])
            avoid.extend(["EURUSD", "GBPUSD", "AUDUSD"])
        elif dxy_trend == "bearish":
            favorable.extend(["EURUSD", "GBPUSD", "AUDUSD"])
            avoid.extend(["USDJPY", "USDCAD"])

        self._context = MarketContext(
            dxy_value=100 + dxy_change,  # Approximate
            dxy_trend=dxy_trend,
            vix_value=vix,
            volatility_regime=vol_regime,
            risk_sentiment=risk_sentiment,
            currency_strengths=currency_strengths,
            regime=regime,
            favorable_pairs=favorable,
            avoid_pairs=avoid,
            updated_at=now,
        )
        self._last_update = now

        return self._context

    def should_take_trade(self, symbol: str, direction: str) -> Tuple[bool, str]:
        """
        Validate a trade against market context.

        Args:
            symbol: e.g., "EURUSD"
            direction: "BUY" or "SELL"

        Returns:
            (should_trade, reason)
        """
        context = self.get_market_context()

        # Check FOREX volatility first (more relevant than VIX)
        can_trade_vol, vol_reason = self.forex_volatility.should_trade(symbol)
        if not can_trade_vol:
            return False, vol_reason

        # Check VIX for high fear (> 30)
        if context.vix_value > 30:
            return False, f"High market fear (VIX={context.vix_value:.1f}) - avoid scalping"

        # CRITICAL: Re-enable avoid_pairs check - EURUSD was in avoid list and we lost!
        # The pair selector score can be fooled by sentiment, but avoid_pairs is based on
        # actual currency strength analysis which proved more reliable.
        if symbol in context.avoid_pairs:
            return False, f"{symbol} in avoid list (currency strength unfavorable)"

        return True, "Trade aligns with market context"

    def get_recommended_sl_tp(self, symbol: str) -> Tuple[float, float]:
        """
        Get ATR-based SL/TP recommendations.

        Returns:
            (sl_pips, tp_pips)
        """
        return self.forex_volatility.get_recommended_sl_tp(symbol)

    def get_position_size_multiplier(self) -> float:
        """Get position size multiplier based on volatility."""
        _, multiplier = self.vix_monitor.should_reduce_size()
        return multiplier

    def log_market_report(self):
        """Log current market intelligence report."""
        context = self.get_market_context()

        LOG.info("=" * 50)
        LOG.info("MARKET INTELLIGENCE REPORT")
        LOG.info("=" * 50)
        LOG.info("DXY: %.2f (%s)", context.dxy_value, context.dxy_trend.upper())
        LOG.info("VIX: %.2f (%s)", context.vix_value, context.volatility_regime.upper())
        LOG.info("Risk: %s", context.risk_sentiment.upper())
        LOG.info("Regime: %s", context.regime.value.upper())
        LOG.info("-" * 50)
        LOG.info("Currency Strength (1=strongest, 8=weakest):")
        for cs in sorted(context.currency_strengths.values(), key=lambda x: x.rank):
            LOG.info("  %d. %s: %.1f (%s)", cs.rank, cs.currency, cs.strength, cs.trend)
        LOG.info("-" * 50)
        LOG.info("Favorable pairs: %s", ", ".join(context.favorable_pairs) or "None")
        LOG.info("Avoid pairs: %s", ", ".join(context.avoid_pairs) or "None")
        LOG.info("=" * 50)


# Convenience function
def get_market_intel(mt5_client=None) -> MarketIntelligence:
    """Get a market intelligence instance."""
    return MarketIntelligence(mt5_client, os.getenv("FINNHUB_API_KEY"))
