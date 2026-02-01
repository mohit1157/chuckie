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

    @staticmethod
    def from_yaml(path: Path) -> "AppConfig":
        raw = yaml.safe_load(path.read_text())
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
            strategy=StrategyConfig(**raw["strategy"]),
            news_filter=NewsFilterConfig(**raw["news_filter"]),
            logging=LoggingConfig(**raw["logging"]),
        )
