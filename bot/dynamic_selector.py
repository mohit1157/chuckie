"""
Dynamic Symbol Selector - Chooses the best pair to trade based on market conditions.

Selection criteria (in order of importance):
1. Currency Strength - Trade strongest vs weakest currency
2. Sentiment Alignment - News + social sentiment agreeing on direction
3. Volatility - Suitable ATR for scalping (not too low, not too high)
4. Spread - Within acceptable limits
5. Session - Pair is active during current session

This module makes the bot fully autonomous in selecting WHAT to trade,
not just WHEN to trade.
"""
import logging
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from datetime import datetime, timezone
from .mt5_client import MT5Client
from .market_intel import MarketIntelligence, CurrencyStrength
from .news import NewsFilter
from .config import AppConfig

LOG = logging.getLogger("bot.dynamic_selector")


@dataclass
class PairScore:
    """Score for a currency pair."""
    symbol: str
    total_score: float
    strength_score: float      # Based on currency strength differential
    sentiment_score: float     # Based on news/social sentiment alignment
    volatility_score: float    # Based on ATR suitability
    spread_score: float        # Based on current spread
    direction: str             # "BUY" or "SELL"
    reason: str                # Human-readable explanation


class DynamicSymbolSelector:
    """
    Dynamically selects the best currency pair to trade.

    Trading Logic:
    1. Calculate strength for all 8 major currencies
    2. Find pairs with largest strength differential
    3. Check if sentiment aligns with strength-based direction
    4. Verify volatility and spread are suitable
    5. Return the best pair + direction to trade

    Usage:
        selector = DynamicSymbolSelector(mt5_client, config)

        # Get best pair to trade right now
        result = selector.get_best_pair()
        if result:
            symbol, direction, score = result
            print(f"Trade {symbol} {direction} (score: {score})")
    """

    # Major pairs to consider
    TRADEABLE_PAIRS = [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
        "AUDUSD", "USDCAD", "NZDUSD",
        "EURGBP", "EURJPY", "GBPJPY",
        "EURAUD", "GBPAUD", "AUDJPY",
        "EURCHF", "GBPCHF", "CADJPY",
    ]

    # Session-appropriate pairs
    SESSION_PAIRS = {
        "london": ["EURUSD", "GBPUSD", "EURGBP", "EURJPY", "GBPJPY", "EURCHF", "GBPCHF"],
        "new_york": ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "USDCHF", "AUDUSD"],
        "tokyo": ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY"],
        "sydney": ["AUDUSD", "NZDUSD", "AUDJPY", "AUDNZD"],
        "overlap_london_ny": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD"],
    }

    def __init__(
        self,
        mt5_client: MT5Client,
        config: AppConfig,
        news_filter: Optional[NewsFilter] = None,
        market_intel: Optional[MarketIntelligence] = None,
    ):
        self.mt5 = mt5_client
        self.cfg = config
        self.news_filter = news_filter

        # Initialize market intelligence if not provided
        if market_intel:
            self.intel = market_intel
        else:
            from .market_intel import MarketIntelligence
            self.intel = MarketIntelligence(mt5_client)

        self._last_selection: Optional[Tuple[str, str, float]] = None
        self._last_selection_time: Optional[datetime] = None
        self._selection_cooldown_minutes = 15  # Don't switch pairs too often

    def get_current_session(self) -> str:
        """Determine current trading session."""
        now = datetime.now(timezone.utc)
        hour = now.hour

        # Session times (UTC)
        # Sydney: 22:00 - 07:00
        # Tokyo: 00:00 - 09:00
        # London: 07:00 - 16:00
        # New York: 12:00 - 21:00

        if 12 <= hour < 16:
            return "overlap_london_ny"  # Best time to trade
        elif 7 <= hour < 12:
            return "london"
        elif 12 <= hour < 21:
            return "new_york"
        elif 0 <= hour < 9:
            return "tokyo"
        else:
            return "sydney"

    def _get_session_pairs(self) -> List[str]:
        """Get pairs appropriate for current session."""
        session = self.get_current_session()
        return self.SESSION_PAIRS.get(session, self.TRADEABLE_PAIRS[:7])

    def _calculate_strength_score(
        self,
        symbol: str,
        strengths: Dict[str, CurrencyStrength]
    ) -> Tuple[float, str, str]:
        """
        Calculate score based on currency strength differential.

        Returns:
            (score, direction, reason)
        """
        base = symbol[:3]
        quote = symbol[3:6]

        base_strength = strengths.get(base)
        quote_strength = strengths.get(quote)

        if not base_strength or not quote_strength:
            return 0.0, "", "Missing strength data"

        # Strength differential (-200 to +200 range)
        differential = base_strength.strength - quote_strength.strength

        # Determine direction based on strength
        if differential > 0:
            direction = "BUY"  # Base stronger, buy the pair
            reason = f"{base} stronger ({base_strength.strength:.0f}) vs {quote} ({quote_strength.strength:.0f})"
        else:
            direction = "SELL"  # Quote stronger, sell the pair
            reason = f"{quote} stronger ({quote_strength.strength:.0f}) vs {base} ({base_strength.strength:.0f})"

        # Normalize score to 0-100 scale
        # Minimum 30 point differential for a good trade
        score = min(100, abs(differential) * 0.5)

        if abs(differential) < 30:
            score *= 0.5  # Penalize weak differential

        return score, direction, reason

    def _calculate_sentiment_score(
        self,
        symbol: str,
        strength_direction: str
    ) -> Tuple[float, str]:
        """
        Calculate score based on sentiment alignment with strength direction.

        Returns:
            (score, reason)
        """
        if not self.news_filter:
            return 50.0, "No sentiment data"  # Neutral

        try:
            sentiment = self.news_filter.get_sentiment()
            if not sentiment:
                return 50.0, "No sentiment data"

            overall = sentiment.get("overall_sentiment", 0)
            recommendation = sentiment.get("recommendation", "neutral")

            # Check alignment with strength direction
            if strength_direction == "BUY":
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
            LOG.warning("Error getting sentiment: %s", e)
            return 50.0, "Sentiment error"

    def _calculate_volatility_score(self, symbol: str) -> Tuple[float, str]:
        """
        Calculate score based on volatility suitability.

        Returns:
            (score, reason)
        """
        try:
            state, atr = self.intel.forex_volatility.get_volatility_state(symbol)

            if state == "normal":
                return 100.0, f"Ideal volatility ({atr:.1f} pips)"
            elif state == "low":
                return 60.0, f"Low volatility ({atr:.1f} pips) - tight range"
            elif state == "elevated":
                return 70.0, f"Elevated volatility ({atr:.1f} pips)"
            else:  # high
                return 20.0, f"High volatility ({atr:.1f} pips) - risky"

        except Exception as e:
            LOG.warning("Error getting volatility for %s: %s", symbol, e)
            return 50.0, "Volatility unknown"

    def _calculate_spread_score(self, symbol: str) -> Tuple[float, str]:
        """
        Calculate score based on current spread.

        Returns:
            (score, reason)
        """
        try:
            info = self.mt5.symbol_info(symbol)
            if not info:
                return 50.0, "Symbol info unavailable"

            spread_points = info.spread
            max_spread = self.cfg.risk.max_spread_points

            if spread_points <= max_spread * 0.5:
                return 100.0, f"Tight spread ({spread_points} points)"
            elif spread_points <= max_spread:
                return 70.0, f"Acceptable spread ({spread_points} points)"
            else:
                return 0.0, f"Wide spread ({spread_points} points)"

        except Exception as e:
            LOG.warning("Error getting spread for %s: %s", symbol, e)
            return 50.0, "Spread unknown"

    def score_pair(self, symbol: str) -> Optional[PairScore]:
        """
        Calculate comprehensive score for a currency pair.

        Returns:
            PairScore or None if pair should not be traded
        """
        # Get currency strengths
        context = self.intel.get_market_context()
        strengths = context.currency_strengths

        # Calculate individual scores
        strength_score, direction, strength_reason = self._calculate_strength_score(
            symbol, strengths
        )

        if not direction:
            return None

        sentiment_score, sentiment_reason = self._calculate_sentiment_score(
            symbol, direction
        )

        volatility_score, volatility_reason = self._calculate_volatility_score(symbol)
        spread_score, spread_reason = self._calculate_spread_score(symbol)

        # Weighted total score
        # Strength: 35%, Sentiment: 30%, Volatility: 20%, Spread: 15%
        total_score = (
            strength_score * 0.35 +
            sentiment_score * 0.30 +
            volatility_score * 0.20 +
            spread_score * 0.15
        )

        # Build reason string
        reasons = [strength_reason]
        if sentiment_score >= 70:
            reasons.append(sentiment_reason)
        if volatility_score >= 70:
            reasons.append(volatility_reason)
        if spread_score >= 70:
            reasons.append(spread_reason)

        return PairScore(
            symbol=symbol,
            total_score=total_score,
            strength_score=strength_score,
            sentiment_score=sentiment_score,
            volatility_score=volatility_score,
            spread_score=spread_score,
            direction=direction,
            reason=" | ".join(reasons),
        )

    def get_best_pair(self, min_score: float = 60.0) -> Optional[Tuple[str, str, PairScore]]:
        """
        Get the best pair to trade right now.

        Args:
            min_score: Minimum score required to trade (0-100)

        Returns:
            (symbol, direction, score) or None if no good setup
        """
        # Check cooldown - don't switch pairs too frequently
        now = datetime.now(timezone.utc)
        if self._last_selection_time:
            minutes_since = (now - self._last_selection_time).seconds / 60
            if minutes_since < self._selection_cooldown_minutes and self._last_selection:
                LOG.debug("Selection cooldown active, keeping %s", self._last_selection[0])
                return self._last_selection

        # Get session-appropriate pairs
        pairs_to_check = self._get_session_pairs()

        # Score all pairs
        scores: List[PairScore] = []
        for symbol in pairs_to_check:
            try:
                score = self.score_pair(symbol)
                if score and score.total_score >= min_score:
                    scores.append(score)
            except Exception as e:
                LOG.warning("Error scoring %s: %s", symbol, e)

        if not scores:
            LOG.info("No pairs meet minimum score of %.0f", min_score)
            return None

        # Sort by total score descending
        scores.sort(key=lambda x: x.total_score, reverse=True)

        best = scores[0]

        # Log top 3 candidates
        LOG.info("=== PAIR SELECTION ===")
        for i, s in enumerate(scores[:3]):
            LOG.info(
                "%d. %s %s: %.1f (str:%.0f sent:%.0f vol:%.0f spr:%.0f)",
                i + 1, s.symbol, s.direction, s.total_score,
                s.strength_score, s.sentiment_score,
                s.volatility_score, s.spread_score
            )
        LOG.info("Selected: %s %s - %s", best.symbol, best.direction, best.reason)
        LOG.info("=" * 22)

        # Cache selection
        self._last_selection = (best.symbol, best.direction, best)
        self._last_selection_time = now

        return (best.symbol, best.direction, best)

    def get_all_scores(self) -> List[PairScore]:
        """Get scores for all tradeable pairs (for analysis/debugging)."""
        pairs_to_check = self._get_session_pairs()
        scores = []

        for symbol in pairs_to_check:
            try:
                score = self.score_pair(symbol)
                if score:
                    scores.append(score)
            except Exception as e:
                LOG.warning("Error scoring %s: %s", symbol, e)

        scores.sort(key=lambda x: x.total_score, reverse=True)
        return scores

    def log_market_scan(self):
        """Log a full market scan report."""
        LOG.info("=" * 60)
        LOG.info("MARKET SCAN REPORT - %s", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
        LOG.info("=" * 60)
        LOG.info("Current Session: %s", self.get_current_session().upper())
        LOG.info("-" * 60)

        scores = self.get_all_scores()

        LOG.info("%-8s %-5s %6s %6s %6s %6s %6s",
                 "PAIR", "DIR", "TOTAL", "STR", "SENT", "VOL", "SPR")
        LOG.info("-" * 60)

        for s in scores:
            LOG.info(
                "%-8s %-5s %6.1f %6.0f %6.0f %6.0f %6.0f",
                s.symbol, s.direction, s.total_score,
                s.strength_score, s.sentiment_score,
                s.volatility_score, s.spread_score
            )

        LOG.info("=" * 60)

        if scores:
            best = scores[0]
            LOG.info("RECOMMENDATION: %s %s (score: %.1f)",
                     best.symbol, best.direction, best.total_score)
            LOG.info("Reason: %s", best.reason)
        else:
            LOG.info("RECOMMENDATION: No suitable pairs found")

        LOG.info("=" * 60)
