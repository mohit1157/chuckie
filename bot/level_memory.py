"""
Level Memory - Remember How Price Reacted at Key Levels.

Good traders remember:
- "Last time price hit 1.1850, it bounced 30 pips"
- "This resistance at 1.2000 has been tested 4 times"
- "Price broke through support aggressively last week"

This module tracks historical reactions at S/R levels.
"""
import logging
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict
from pathlib import Path

LOG = logging.getLogger("bot.level_memory")


@dataclass
class LevelReaction:
    """Record of how price reacted at a level."""
    timestamp: str
    level_price: float
    level_type: str  # "support" or "resistance"
    reaction_type: str  # "bounce", "break", "reject", "consolidate"
    reaction_strength: float  # Pips moved after touching level
    candles_at_level: int  # How many candles spent at level
    result: str  # "reversal", "continuation", "chop"


@dataclass
class TrackedLevel:
    """A support/resistance level with history."""
    price: float
    level_type: str  # "support" or "resistance"
    first_seen: str
    last_seen: str
    touch_count: int
    reactions: List[LevelReaction]
    avg_bounce_pips: float
    break_count: int  # Times level was broken
    hold_count: int  # Times level held


class LevelMemory:
    """
    Remembers how price has reacted at key levels.

    Like a trader's notebook of important levels:
    - Track each time price hits a level
    - Record the reaction (bounce, break, consolidate)
    - Build statistics on level strength
    - Use history to predict future reactions
    """

    def __init__(self, symbol: str, pip_value: float = 0.0001,
                 memory_file: Optional[str] = None):
        self.symbol = symbol
        self.pip_value = pip_value
        self.memory_file = memory_file or f"level_memory_{symbol}.json"
        self.levels: Dict[str, TrackedLevel] = {}  # price_key -> TrackedLevel
        self.level_tolerance_pips = 3.0  # Levels within 3 pips are same level
        self.max_level_age_days = 30  # Forget levels older than 30 days

        self._load_memory()

    def _price_to_key(self, price: float) -> str:
        """Convert price to a level key (rounded to tolerance)."""
        pips = price / self.pip_value
        rounded_pips = round(pips / self.level_tolerance_pips) * self.level_tolerance_pips
        return f"{rounded_pips:.0f}"

    def _load_memory(self):
        """Load level memory from file."""
        try:
            path = Path(self.memory_file)
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                    for key, level_data in data.get('levels', {}).items():
                        reactions = [LevelReaction(**r) for r in level_data.get('reactions', [])]
                        level_data['reactions'] = reactions
                        self.levels[key] = TrackedLevel(**level_data)
                LOG.info("Loaded %d levels from memory", len(self.levels))
        except Exception as e:
            LOG.warning("Could not load level memory: %s", e)
            self.levels = {}

    def _save_memory(self):
        """Save level memory to file."""
        try:
            data = {
                'symbol': self.symbol,
                'levels': {}
            }
            for key, level in self.levels.items():
                level_dict = asdict(level)
                level_dict['reactions'] = [asdict(r) for r in level.reactions]
                data['levels'][key] = level_dict

            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            LOG.warning("Could not save level memory: %s", e)

    def record_level_touch(self, price: float, level_type: str,
                          reaction_type: str, reaction_pips: float,
                          candles_at_level: int = 1):
        """
        Record when price touched a level and how it reacted.

        Args:
            price: The level price
            level_type: "support" or "resistance"
            reaction_type: "bounce", "break", "reject", "consolidate"
            reaction_pips: How many pips price moved after touching
            candles_at_level: How many candles price spent at level
        """
        key = self._price_to_key(price)
        now = datetime.now(timezone.utc).isoformat()

        # Determine result based on reaction
        if reaction_type in ["bounce", "reject"]:
            result = "reversal"
        elif reaction_type == "break":
            result = "continuation"
        else:
            result = "chop"

        reaction = LevelReaction(
            timestamp=now,
            level_price=price,
            level_type=level_type,
            reaction_type=reaction_type,
            reaction_strength=reaction_pips,
            candles_at_level=candles_at_level,
            result=result
        )

        if key in self.levels:
            level = self.levels[key]
            level.touch_count += 1
            level.last_seen = now
            level.reactions.append(reaction)

            # Update statistics
            if reaction_type in ["bounce", "reject"]:
                level.hold_count += 1
            elif reaction_type == "break":
                level.break_count += 1

            # Update average bounce
            bounces = [r.reaction_strength for r in level.reactions
                      if r.reaction_type in ["bounce", "reject"]]
            if bounces:
                level.avg_bounce_pips = sum(bounces) / len(bounces)
        else:
            # New level
            self.levels[key] = TrackedLevel(
                price=price,
                level_type=level_type,
                first_seen=now,
                last_seen=now,
                touch_count=1,
                reactions=[reaction],
                avg_bounce_pips=reaction_pips if reaction_type in ["bounce", "reject"] else 0,
                break_count=1 if reaction_type == "break" else 0,
                hold_count=1 if reaction_type in ["bounce", "reject"] else 0
            )

        LOG.info("Level recorded: %.5f (%s) - %s (%.1f pips)",
                 price, level_type, reaction_type, reaction_pips)

        self._save_memory()
        self._cleanup_old_levels()

    def get_level_info(self, price: float) -> Optional[TrackedLevel]:
        """Get info about a level near this price."""
        key = self._price_to_key(price)
        return self.levels.get(key)

    def get_level_strength(self, price: float) -> float:
        """
        Get the strength score of a level (0.0 to 1.0).

        Stronger levels:
        - Have been touched multiple times
        - Have held more often than broken
        - Have larger average bounces
        """
        level = self.get_level_info(price)
        if level is None:
            return 0.0

        # Score components
        touch_score = min(1.0, level.touch_count / 5.0)  # Max at 5 touches

        if level.touch_count > 0:
            hold_ratio = level.hold_count / level.touch_count
        else:
            hold_ratio = 0.0

        bounce_score = min(1.0, level.avg_bounce_pips / 20.0)  # Max at 20 pip bounce

        # Combined strength
        strength = (touch_score * 0.3 + hold_ratio * 0.4 + bounce_score * 0.3)

        return strength

    def predict_reaction(self, price: float, level_type: str) -> Dict:
        """
        Predict how price might react at this level based on history.

        Returns dict with:
        - likely_reaction: "bounce", "break", or "unknown"
        - confidence: 0.0 to 1.0
        - avg_bounce_pips: historical average
        - touch_count: how many times tested
        """
        level = self.get_level_info(price)

        if level is None:
            return {
                "likely_reaction": "unknown",
                "confidence": 0.0,
                "avg_bounce_pips": 0.0,
                "touch_count": 0
            }

        # Calculate bounce vs break ratio
        total = level.hold_count + level.break_count
        if total == 0:
            likely_reaction = "unknown"
            confidence = 0.0
        elif level.hold_count > level.break_count:
            likely_reaction = "bounce"
            confidence = level.hold_count / total
        else:
            likely_reaction = "break"
            confidence = level.break_count / total

        # Adjust confidence based on sample size
        sample_factor = min(1.0, total / 5.0)
        confidence = confidence * sample_factor

        return {
            "likely_reaction": likely_reaction,
            "confidence": confidence,
            "avg_bounce_pips": level.avg_bounce_pips,
            "touch_count": level.touch_count
        }

    def get_nearby_levels(self, current_price: float, range_pips: float = 50) -> List[TrackedLevel]:
        """Get all tracked levels within range of current price."""
        nearby = []
        range_price = range_pips * self.pip_value

        for level in self.levels.values():
            if abs(level.price - current_price) <= range_price:
                nearby.append(level)

        # Sort by distance from current price
        nearby.sort(key=lambda l: abs(l.price - current_price))

        return nearby

    def get_strongest_levels(self, n: int = 5) -> List[TrackedLevel]:
        """Get the N strongest levels by strength score."""
        scored = []
        for level in self.levels.values():
            strength = self.get_level_strength(level.price)
            scored.append((level, strength))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored[:n]]

    def _cleanup_old_levels(self):
        """Remove levels that haven't been seen recently."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.max_level_age_days)
        cutoff_str = cutoff.isoformat()

        old_keys = [key for key, level in self.levels.items()
                   if level.last_seen < cutoff_str]

        for key in old_keys:
            del self.levels[key]

        if old_keys:
            LOG.info("Cleaned up %d old levels", len(old_keys))
            self._save_memory()

    def log_level_report(self):
        """Log a report of known levels."""
        if not self.levels:
            LOG.info("No levels in memory")
            return

        LOG.info("=== Level Memory Report ===")
        LOG.info("Total levels tracked: %d", len(self.levels))

        strongest = self.get_strongest_levels(5)
        for level in strongest:
            strength = self.get_level_strength(level.price)
            prediction = self.predict_reaction(level.price, level.level_type)
            LOG.info("  %.5f (%s): %d touches | %.0f%% hold | likely: %s (%.0f%% conf)",
                     level.price, level.level_type, level.touch_count,
                     (level.hold_count / level.touch_count * 100) if level.touch_count > 0 else 0,
                     prediction["likely_reaction"], prediction["confidence"] * 100)
