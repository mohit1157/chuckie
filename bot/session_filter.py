"""
Trading Session Filter.
Ensures we only trade during optimal market hours with high liquidity.
"""
import logging
from datetime import datetime, timezone, time
from enum import Enum
from typing import Optional

LOG = logging.getLogger("bot.session")


class TradingSession(Enum):
    """Major forex trading sessions."""
    SYDNEY = "sydney"
    TOKYO = "tokyo"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP_LONDON_NY = "london_ny_overlap"
    OVERLAP_TOKYO_LONDON = "tokyo_london_overlap"
    OFF_HOURS = "off_hours"


class SessionFilter:
    """
    Filter trades based on market session and liquidity.

    Best times to trade forex (in UTC):
    - London Session: 07:00 - 16:00 UTC (highest EUR/GBP volume)
    - New York Session: 12:00 - 21:00 UTC (highest USD volume)
    - London/NY Overlap: 12:00 - 16:00 UTC (BEST - highest liquidity)
    - Tokyo Session: 00:00 - 09:00 UTC (highest JPY volume)
    - Sydney Session: 22:00 - 07:00 UTC (lower liquidity)

    Sessions to AVOID for scalping:
    - Sunday open (22:00 UTC Sunday) - gaps and low liquidity
    - Friday close (20:00-21:00 UTC Friday) - spreads widen
    - Major news releases (NFP, FOMC, ECB, etc.)
    - Holiday periods
    """

    # Session times in UTC
    SESSIONS = {
        TradingSession.SYDNEY: (22, 7),      # 22:00 - 07:00
        TradingSession.TOKYO: (0, 9),         # 00:00 - 09:00
        TradingSession.LONDON: (7, 16),       # 07:00 - 16:00
        TradingSession.NEW_YORK: (12, 21),    # 12:00 - 21:00
    }

    # Optimal trading windows
    OVERLAP_LONDON_NY = (12, 16)  # 12:00 - 16:00 UTC
    OVERLAP_TOKYO_LONDON = (7, 9)  # 07:00 - 09:00 UTC

    def __init__(self, allowed_sessions: Optional[list] = None):
        """
        Initialize session filter.

        Args:
            allowed_sessions: List of allowed TradingSession values.
                            Default: London, NY, and overlaps.
        """
        if allowed_sessions is None:
            self.allowed_sessions = [
                TradingSession.LONDON,
                TradingSession.NEW_YORK,
                TradingSession.OVERLAP_LONDON_NY,
            ]
        else:
            self.allowed_sessions = allowed_sessions

    def get_current_session(self) -> TradingSession:
        """Determine current trading session."""
        now = datetime.now(timezone.utc)
        hour = now.hour
        weekday = now.weekday()  # 0=Monday, 6=Sunday

        # Check overlaps first (they take priority)
        if self.OVERLAP_LONDON_NY[0] <= hour < self.OVERLAP_LONDON_NY[1]:
            return TradingSession.OVERLAP_LONDON_NY

        if self.OVERLAP_TOKYO_LONDON[0] <= hour < self.OVERLAP_TOKYO_LONDON[1]:
            return TradingSession.OVERLAP_TOKYO_LONDON

        # Check individual sessions
        for session, (start, end) in self.SESSIONS.items():
            if start < end:
                if start <= hour < end:
                    return session
            else:  # Wraps around midnight (Sydney)
                if hour >= start or hour < end:
                    return session

        return TradingSession.OFF_HOURS

    def is_allowed_session(self) -> bool:
        """Check if current session is in allowed list."""
        current = self.get_current_session()
        return current in self.allowed_sessions

    def is_market_closed(self) -> tuple:
        """
        Check if the forex market is closed (weekend).

        Forex market hours:
        - Opens: Sunday 5:00 PM EST (22:00 UTC)
        - Closes: Friday 5:00 PM EST (22:00 UTC)

        Returns:
            (is_closed: bool, reason: str)
        """
        now = datetime.now(timezone.utc)
        hour = now.hour
        weekday = now.weekday()  # 0=Monday, 6=Sunday

        # Saturday: Market fully closed
        if weekday == 5:
            return True, "Market closed (Saturday)"

        # Sunday before 22:00 UTC (5 PM EST): Market closed
        if weekday == 6 and hour < 22:
            return True, "Market closed (Sunday - opens 5 PM EST / 22:00 UTC)"

        # Friday after 22:00 UTC: Market closed
        if weekday == 4 and hour >= 22:
            return True, "Market closed (Friday close)"

        return False, ""

    def should_avoid_now(self) -> tuple:
        """
        Check if trading should be avoided now.

        Returns:
            (should_avoid: bool, reason: str)
        """
        # First check if market is closed (weekend)
        closed, reason = self.is_market_closed()
        if closed:
            return True, reason

        now = datetime.now(timezone.utc)
        hour = now.hour
        minute = now.minute
        weekday = now.weekday()

        # Avoid Sunday market open (first 2 hours after open)
        # Market opens at 22:00 UTC Sunday - gaps and wide spreads common
        # TEMPORARILY DISABLED FOR TESTING - uncomment for live trading:
        # if weekday == 6 and hour >= 22:
        #     return True, "Sunday market open - gaps and wide spreads possible"
        # if weekday == 0 and hour < 0:
        #     return True, "Early Monday - low liquidity"

        # Avoid Friday close (last 2 hours before close)
        if weekday == 4 and hour >= 20:
            return True, "Friday close - spreads widening"

        # Avoid low liquidity hours (3-5 UTC)
        if 3 <= hour < 5:
            return True, "Low liquidity period"

        # Avoid exact hour marks (news release times)
        if minute == 0 or minute == 30:
            return True, "Potential news release time"

        return False, ""

    def get_session_quality(self) -> tuple:
        """
        Rate current session quality for scalping.

        Returns:
            (quality: str, score: int)
            quality: 'excellent', 'good', 'fair', 'poor'
            score: 0-100
        """
        session = self.get_current_session()
        avoid, _ = self.should_avoid_now()

        if avoid:
            return "poor", 10

        quality_map = {
            TradingSession.OVERLAP_LONDON_NY: ("excellent", 100),
            TradingSession.LONDON: ("good", 80),
            TradingSession.NEW_YORK: ("good", 75),
            TradingSession.OVERLAP_TOKYO_LONDON: ("fair", 60),
            TradingSession.TOKYO: ("fair", 50),
            TradingSession.SYDNEY: ("poor", 30),
            TradingSession.OFF_HOURS: ("poor", 10),
        }

        return quality_map.get(session, ("unknown", 0))

    def time_until_good_session(self) -> int:
        """
        Calculate minutes until next good trading session.

        Returns:
            Minutes until London session or NY overlap.
        """
        now = datetime.now(timezone.utc)
        hour = now.hour
        minute = now.minute
        current_minutes = hour * 60 + minute

        # Target: London open (07:00 UTC) or NY overlap (12:00 UTC)
        london_open = 7 * 60  # 420 minutes
        ny_overlap = 12 * 60  # 720 minutes

        if current_minutes < london_open:
            return london_open - current_minutes
        elif current_minutes < ny_overlap:
            return 0  # Already in good session
        elif current_minutes < 16 * 60:
            return 0  # Still in overlap
        else:
            # Wait for next London open (next day)
            return (24 * 60 - current_minutes) + london_open

    def log_session_status(self):
        """Log current session status."""
        session = self.get_current_session()
        quality, score = self.get_session_quality()
        avoid, reason = self.should_avoid_now()

        LOG.info("Session: %s | Quality: %s (%d/100) | Avoid: %s %s",
                 session.value, quality, score, avoid, f"({reason})" if reason else "")


# Quick-access functions
def is_good_time_to_trade() -> bool:
    """Quick check if now is a good time to scalp."""
    sf = SessionFilter()
    avoid, _ = sf.should_avoid_now()
    if avoid:
        return False
    quality, score = sf.get_session_quality()
    return score >= 60


def get_session_info() -> dict:
    """Get comprehensive session information."""
    sf = SessionFilter()
    session = sf.get_current_session()
    quality, score = sf.get_session_quality()
    avoid, reason = sf.should_avoid_now()
    wait = sf.time_until_good_session()

    return {
        "session": session.value,
        "quality": quality,
        "score": score,
        "should_avoid": avoid,
        "avoid_reason": reason,
        "minutes_until_good_session": wait,
    }
