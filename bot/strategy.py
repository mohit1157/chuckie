import logging
from typing import Optional
import numpy as np
from .config import AppConfig
from .data import get_recent_bars
from .execution import Signal
from .mt5_client import MT5Client

LOG = logging.getLogger("bot.strategy")

def ema(values: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out

class StrategyEngine:
    """Trend filter + breakout trigger."""
    def __init__(self, cfg: AppConfig, mt5c: MT5Client):
        self.cfg = cfg
        self.mt5 = mt5c
        self._last_signal_time: Optional[int] = None

    def get_signal(self) -> Optional[Signal]:
        rates = get_recent_bars(self.mt5, self.cfg.symbol, self.cfg.timeframe, n=200)
        if rates is None or len(rates) < 60:
            return None

        close = np.array([r["close"] for r in rates], dtype=float)
        high = np.array([r["high"] for r in rates], dtype=float)
        low  = np.array([r["low"]  for r in rates], dtype=float)

        fast = ema(close, self.cfg.strategy.fast_ema)
        slow = ema(close, self.cfg.strategy.slow_ema)

        trend_up = fast[-1] > slow[-1]
        trend_dn = fast[-1] < slow[-1]

        lb = self.cfg.strategy.breakout_lookback
        recent_high = np.max(high[-lb-1:-1])
        recent_low  = np.min(low[-lb-1:-1])
        last_close  = close[-1]

        last_bar_time = int(rates[-1]["time"])
        if self._last_signal_time == last_bar_time:
            return None

        if trend_up and last_close > recent_high:
            self._last_signal_time = last_bar_time
            return Signal(side="BUY", reason="trend_breakout_up")

        if trend_dn and last_close < recent_low:
            self._last_signal_time = last_bar_time
            return Signal(side="SELL", reason="trend_breakout_dn")

        return None
