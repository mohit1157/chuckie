"""
Technical indicators for scalping strategy.
Optimized for speed and accuracy.
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class IndicatorValues:
    """Container for all indicator values at current bar."""
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float
    atr: float
    ema_fast: float
    ema_slow: float
    adx: float
    plus_di: float
    minus_di: float
    momentum: float
    stoch_k: float
    stoch_d: float


def ema(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def sma(values: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""
    out = np.empty_like(values, dtype=float)
    out[:period-1] = np.nan
    for i in range(period - 1, len(values)):
        out[i] = np.mean(values[i - period + 1:i + 1])
    return out


def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index."""
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.empty(len(close), dtype=float)
    avg_loss = np.empty(len(close), dtype=float)
    avg_gain[:] = np.nan
    avg_loss[:] = np.nan

    if len(gains) < period:
        return np.full(len(close), 50.0)

    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period

    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi_values = 100 - (100 / (1 + rs))
    rsi_values[:period] = 50.0
    return rsi_values


def macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD: returns (macd_line, signal_line, histogram)."""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(close: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands: returns (upper, middle, lower)."""
    middle = sma(close, period)
    rolling_std = np.empty_like(close, dtype=float)
    rolling_std[:period-1] = np.nan
    for i in range(period - 1, len(close)):
        rolling_std[i] = np.std(close[i - period + 1:i + 1])

    upper = middle + (std_dev * rolling_std)
    lower = middle - (std_dev * rolling_std)
    return upper, middle, lower


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range - volatility indicator."""
    tr = np.empty(len(close), dtype=float)
    tr[0] = high[0] - low[0]

    for i in range(1, len(close)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)

    atr_values = ema(tr, period)
    return atr_values


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average Directional Index: returns (adx, +DI, -DI)."""
    n = len(close)
    plus_dm = np.zeros(n, dtype=float)
    minus_dm = np.zeros(n, dtype=float)
    tr = np.zeros(n, dtype=float)

    tr[0] = high[0] - low[0]

    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)

    atr_vals = ema(tr, period)
    plus_di = 100 * ema(plus_dm, period) / np.where(atr_vals == 0, 1e-10, atr_vals)
    minus_di = 100 * ema(minus_dm, period) / np.where(atr_vals == 0, 1e-10, atr_vals)

    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1e-10, plus_di + minus_di)
    adx_vals = ema(dx, period)

    return adx_vals, plus_di, minus_di


def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Stochastic Oscillator: returns (%K, %D)."""
    n = len(close)
    stoch_k = np.empty(n, dtype=float)
    stoch_k[:k_period-1] = 50.0

    for i in range(k_period - 1, n):
        highest_high = np.max(high[i - k_period + 1:i + 1])
        lowest_low = np.min(low[i - k_period + 1:i + 1])

        if highest_high == lowest_low:
            stoch_k[i] = 50.0
        else:
            stoch_k[i] = 100 * (close[i] - lowest_low) / (highest_high - lowest_low)

    stoch_d = sma(stoch_k, d_period)
    stoch_d = np.nan_to_num(stoch_d, nan=50.0)

    return stoch_k, stoch_d


def momentum(close: np.ndarray, period: int = 10) -> np.ndarray:
    """Price momentum (rate of change)."""
    mom = np.zeros(len(close), dtype=float)
    for i in range(period, len(close)):
        if close[i - period] != 0:
            mom[i] = ((close[i] - close[i - period]) / close[i - period]) * 100
    return mom


def calculate_all_indicators(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ema_fast_period: int = 9,
    ema_slow_period: int = 21,
    rsi_period: int = 14,
    atr_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
    adx_period: int = 14,
    stoch_k: int = 14,
    stoch_d: int = 3
) -> IndicatorValues:
    """Calculate all indicators and return current values."""

    # EMA
    ema_fast_vals = ema(close, ema_fast_period)
    ema_slow_vals = ema(close, ema_slow_period)

    # RSI
    rsi_vals = rsi(close, rsi_period)

    # MACD
    macd_line, signal_line, histogram = macd(close)

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = bollinger_bands(close, bb_period, bb_std)
    bb_width = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1] if bb_middle[-1] != 0 else 0

    # ATR
    atr_vals = atr(high, low, close, atr_period)

    # ADX
    adx_vals, plus_di, minus_di = adx(high, low, close, adx_period)

    # Stochastic
    stoch_k_vals, stoch_d_vals = stochastic(high, low, close, stoch_k, stoch_d)

    # Momentum
    mom_vals = momentum(close)

    return IndicatorValues(
        rsi=float(rsi_vals[-1]),
        macd=float(macd_line[-1]),
        macd_signal=float(signal_line[-1]),
        macd_histogram=float(histogram[-1]),
        bb_upper=float(bb_upper[-1]),
        bb_middle=float(bb_middle[-1]),
        bb_lower=float(bb_lower[-1]),
        bb_width=float(bb_width),
        atr=float(atr_vals[-1]),
        ema_fast=float(ema_fast_vals[-1]),
        ema_slow=float(ema_slow_vals[-1]),
        adx=float(adx_vals[-1]),
        plus_di=float(plus_di[-1]),
        minus_di=float(minus_di[-1]),
        momentum=float(mom_vals[-1]),
        stoch_k=float(stoch_k_vals[-1]),
        stoch_d=float(stoch_d_vals[-1])
    )
