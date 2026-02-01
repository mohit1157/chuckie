"""
Forex MT5 Scalping Bot Package.

High-probability day trading bot targeting 80%+ win rate.
"""

from .config import AppConfig
from .mt5_client import MT5Client
from .indicators import calculate_all_indicators, IndicatorValues
from .scalping_strategy import ScalpingStrategy, ConservativeScalpingStrategy
from .scalping_execution import ScalpingExecutionEngine
from .trade_logger import TradeLogger, PerformanceStats
from .session_filter import SessionFilter, get_session_info, is_good_time_to_trade
from .risk import RiskManager
from .execution import Signal

__all__ = [
    "AppConfig",
    "MT5Client",
    "calculate_all_indicators",
    "IndicatorValues",
    "ScalpingStrategy",
    "ConservativeScalpingStrategy",
    "ScalpingExecutionEngine",
    "TradeLogger",
    "PerformanceStats",
    "SessionFilter",
    "get_session_info",
    "is_good_time_to_trade",
    "RiskManager",
    "Signal",
]

__version__ = "2.0.0"
