from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import yaml


@dataclass
class RiskConfig:
    account_risk_per_trade: float
    max_spread_points: int
    max_open_positions: int
    max_daily_loss_pct: float


@dataclass
class TrailingStopConfig:
    enabled: bool
    start_pips: float
    trail_pips: float


@dataclass
class TradeConfig:
    sl_pips: float
    tp_pips: float
    trailing_stop: TrailingStopConfig


@dataclass
class StrategyConfig:
    name: str
    fast_ema: int
    slow_ema: int
    breakout_lookback: int
    # Scalping-specific parameters
    min_confirmations: int = 5
    use_dynamic_sl_tp: bool = True
    atr_sl_multiplier: float = 1.5
    atr_tp_multiplier: float = 1.0
    cooldown_minutes: int = 3


@dataclass
class SessionConfig:
    enabled: bool = True
    london_only: bool = False
    overlap_only: bool = False
    avoid_news_times: bool = True


@dataclass
class NewsFilterConfig:
    enabled: bool
    minutes_before: int
    minutes_after: int
    provider: str


@dataclass
class LoggingConfig:
    level: str


@dataclass
class AppConfig:
    mode: Literal["demo", "live"]
    symbol: str
    timeframe: str
    magic: int
    risk: RiskConfig
    trade: TradeConfig
    strategy: StrategyConfig
    news_filter: NewsFilterConfig
    logging: LoggingConfig
    session: SessionConfig = None

    def __post_init__(self):
        if self.session is None:
            self.session = SessionConfig()

    @staticmethod
    def from_yaml(path: Path) -> "AppConfig":
        raw = yaml.safe_load(path.read_text())

        # Build strategy config with defaults for optional scalping params
        strategy_raw = raw["strategy"]
        strategy_config = StrategyConfig(
            name=strategy_raw["name"],
            fast_ema=strategy_raw.get("fast_ema", 9),
            slow_ema=strategy_raw.get("slow_ema", 21),
            breakout_lookback=strategy_raw.get("breakout_lookback", 20),
            min_confirmations=strategy_raw.get("min_confirmations", 5),
            use_dynamic_sl_tp=strategy_raw.get("use_dynamic_sl_tp", True),
            atr_sl_multiplier=strategy_raw.get("atr_sl_multiplier", 1.5),
            atr_tp_multiplier=strategy_raw.get("atr_tp_multiplier", 1.0),
            cooldown_minutes=strategy_raw.get("cooldown_minutes", 3),
        )

        # Build session config if present
        session_raw = raw.get("session", {})
        session_config = SessionConfig(
            enabled=session_raw.get("enabled", True),
            london_only=session_raw.get("london_only", False),
            overlap_only=session_raw.get("overlap_only", False),
            avoid_news_times=session_raw.get("avoid_news_times", True),
        )

        return AppConfig(
            mode=raw["mode"],
            symbol=raw["symbol"],
            timeframe=raw["timeframe"],
            magic=int(raw["magic"]),
            risk=RiskConfig(**raw["risk"]),
            trade=TradeConfig(
                sl_pips=float(raw["trade"]["sl_pips"]),
                tp_pips=float(raw["trade"]["tp_pips"]),
                trailing_stop=TrailingStopConfig(**raw["trade"]["trailing_stop"]),
            ),
            strategy=strategy_config,
            news_filter=NewsFilterConfig(**raw["news_filter"]),
            logging=LoggingConfig(**raw["logging"]),
            session=session_config,
        )
