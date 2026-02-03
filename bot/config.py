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
    max_lot_size: float = 3.0


@dataclass
class TrailingStopConfig:
    enabled: bool
    start_pips: float
    trail_pips: float


@dataclass
class PartialProfitConfig:
    enabled: bool = True
    profit_pct: float = 0.15       # LEGACY: Take partial at 15% profit (not used anymore)
    profit_pips: float = 5.0       # FIX 9: Take partial at 5 pips profit (scalping friendly)
    close_pct: float = 0.50        # FIX 9: Close 50% of position (was 80%)
    move_sl_to_profit: bool = True # Move SL to lock in profit


@dataclass
class TradeConfig:
    sl_pips: float
    tp_pips: float
    trailing_stop: TrailingStopConfig
    exit_on_ema_cross: bool = True       # Exit if price closes below EMA(9)
    exit_on_structure_break: bool = True  # Exit on support/resistance break
    partial_profit: PartialProfitConfig = None

    def __post_init__(self):
        if self.partial_profit is None:
            self.partial_profit = PartialProfitConfig()


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
    strong_rejection_enabled: bool = False
    
    # High Precision Mode (85% Win Rate Goal)
    high_precision_mode: bool = True
    min_adx: float = 20.0
    max_rsi_buy: float = 70.0
    min_rsi_buy: float = 40.0
    max_rsi_sell: float = 60.0
    min_rsi_sell: float = 30.0
    max_spread_pips: float = 1.5


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
            strong_rejection_enabled=strategy_raw.get("strong_rejection_enabled", False),
            high_precision_mode=strategy_raw.get("high_precision_mode", True),
            min_adx=strategy_raw.get("min_adx", 20.0),
            max_rsi_buy=strategy_raw.get("max_rsi_buy", 70.0),
            min_rsi_buy=strategy_raw.get("min_rsi_buy", 40.0),
            max_rsi_sell=strategy_raw.get("max_rsi_sell", 60.0),
            min_rsi_sell=strategy_raw.get("min_rsi_sell", 30.0),
            max_spread_pips=strategy_raw.get("max_spread_pips", 1.5),
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
            risk=RiskConfig(
                account_risk_per_trade=raw["risk"]["account_risk_per_trade"],
                max_spread_points=raw["risk"]["max_spread_points"],
                max_open_positions=raw["risk"]["max_open_positions"],
                max_daily_loss_pct=raw["risk"]["max_daily_loss_pct"],
                max_lot_size=raw["risk"].get("max_lot_size", 3.0),
            ),
            trade=TradeConfig(
                sl_pips=float(raw["trade"]["sl_pips"]),
                tp_pips=float(raw["trade"]["tp_pips"]),
                trailing_stop=TrailingStopConfig(**raw["trade"]["trailing_stop"]),
                exit_on_ema_cross=raw["trade"].get("exit_on_ema_cross", True),
                exit_on_structure_break=raw["trade"].get("exit_on_structure_break", True),
                partial_profit=PartialProfitConfig(
                    enabled=raw["trade"].get("partial_profit", {}).get("enabled", True),
                    profit_pct=raw["trade"].get("partial_profit", {}).get("profit_pct", 0.15),
                    profit_pips=raw["trade"].get("partial_profit", {}).get("profit_pips", 5.0),
                    close_pct=raw["trade"].get("partial_profit", {}).get("close_pct", 0.80),
                    move_sl_to_profit=raw["trade"].get("partial_profit", {}).get("move_sl_to_profit", True),
                ),
            ),
            strategy=strategy_config,
            news_filter=NewsFilterConfig(**raw["news_filter"]),
            logging=LoggingConfig(**raw["logging"]),
            session=session_config,
        )
