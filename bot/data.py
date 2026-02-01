import logging
import MetaTrader5 as mt5
from .mt5_client import MT5Client

LOG = logging.getLogger("bot.data")

TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}

def get_timeframe(tf: str):
    if tf not in TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe {tf}. Use one of {list(TIMEFRAMES.keys())}")
    return TIMEFRAMES[tf]

def get_recent_bars(mt5c: MT5Client, symbol: str, timeframe: str, n: int = 200):
    tf = get_timeframe(timeframe)
    mt5c.symbol_select(symbol)
    rates = mt5c.copy_rates_from_pos(symbol, tf, 0, n)
    if rates is None or len(rates) < n:
        code, msg = mt5c.last_error()
        LOG.warning("copy_rates_from_pos insufficient: %s %s", code, msg)
    return rates
