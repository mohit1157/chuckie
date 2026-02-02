"""
Smart Trade Manager - Trades Like a Professional Scalper.

This module manages open positions the way a good trader would:
1. Move to breakeven quickly (after 3 pips profit)
2. Trail stop aggressively to lock in gains
3. Watch for reversal candles and exit early
4. Monitor momentum fading (shrinking candles)
5. Time-based management (don't hold too long in chop)

A good trader doesn't just set SL/TP and walk away.
They actively manage the trade based on what the chart is showing.
"""
import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple
import MetaTrader5 as mt5

from .data import get_recent_bars

LOG = logging.getLogger("bot.trade_manager")


@dataclass
class ManagedTrade:
    """State for a trade being actively managed."""
    ticket: int
    symbol: str
    side: str  # "BUY" or "SELL"
    entry_price: float
    entry_time: datetime
    initial_sl: float
    initial_tp: float
    initial_volume: float

    # Dynamic state
    current_sl: float = 0.0
    highest_profit_pips: float = 0.0  # Track best profit reached
    lowest_profit_pips: float = 0.0   # Track worst drawdown
    breakeven_hit: bool = False
    trail_started: bool = False

    # Candle tracking
    candles_since_entry: int = 0
    consecutive_against_candles: int = 0  # Candles moving against us
    last_candle_time: Optional[datetime] = None

    # Momentum tracking
    recent_candle_sizes: List[float] = field(default_factory=list)
    momentum_fading: bool = False

    def __post_init__(self):
        self.current_sl = self.initial_sl


class SmartTradeManager:
    """
    Manages trades like a professional scalper.

    Key behaviors:
    1. BREAKEVEN: Move SL to entry+0.5 pips after 3 pips profit
    2. AGGRESSIVE TRAIL: Trail SL 2 pips behind price after breakeven
    3. REVERSAL EXIT: Close if strong reversal candle forms while in profit
    4. MOMENTUM EXIT: Close if momentum fades (3+ shrinking candles)
    5. TIME EXIT: Close if trade chops for too long without progress
    """

    def __init__(self, mt5_client, config):
        self.mt5 = mt5_client
        self.cfg = config
        self._trades: Dict[int, ManagedTrade] = {}

        # Configuration - like a good trader would set
        self.breakeven_trigger_pips = 3.0      # Move to BE after 3 pips
        self.breakeven_buffer_pips = 0.5       # Lock in 0.5 pips at BE
        self.trail_start_pips = 4.0            # Start trailing after 4 pips
        self.trail_distance_pips = 2.0         # Trail 2 pips behind
        self.max_chop_candles = 10             # Exit if chopping for 10 candles
        self.min_progress_pips = 2.0           # Need 2 pips progress or exit
        self.reversal_candle_ratio = 0.6       # 60% body = strong reversal

    def register_trade(self, ticket: int, symbol: str, side: str,
                       entry_price: float, sl: float, tp: float, volume: float):
        """Register a new trade for management."""
        trade = ManagedTrade(
            ticket=ticket,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            entry_time=datetime.now(timezone.utc),
            initial_sl=sl,
            initial_tp=tp,
            initial_volume=volume,
            current_sl=sl
        )
        self._trades[ticket] = trade
        LOG.info("Trade registered for management: ticket=%d %s %s @ %.5f",
                 ticket, side, symbol, entry_price)

    def unregister_trade(self, ticket: int):
        """Remove a trade from management (closed externally)."""
        if ticket in self._trades:
            del self._trades[ticket]
            LOG.info("Trade unregistered: ticket=%d", ticket)

    def manage_all_trades(self) -> List[Tuple[int, str]]:
        """
        Manage all registered trades.

        Returns:
            List of (ticket, action) tuples where action is:
            - "close_reversal" - Close due to reversal candle
            - "close_momentum" - Close due to fading momentum
            - "close_chop" - Close due to excessive chop
            - "update_sl" - SL was updated
            - "none" - No action taken
        """
        actions = []

        for ticket, trade in list(self._trades.items()):
            action = self._manage_single_trade(trade)
            actions.append((ticket, action))

        return actions

    def _manage_single_trade(self, trade: ManagedTrade) -> str:
        """
        Manage a single trade like a professional.

        Returns action taken.
        """
        # Get current price
        tick = self.mt5.symbol_info_tick(trade.symbol)
        if tick is None:
            return "none"

        info = self.mt5.symbol_info(trade.symbol)
        if info is None:
            return "none"

        pip_value = info.point * 10

        # Calculate current profit in pips
        if trade.side == "BUY":
            current_price = tick.bid
            profit_pips = (current_price - trade.entry_price) / pip_value
        else:
            current_price = tick.ask
            profit_pips = (trade.entry_price - current_price) / pip_value

        # Track best/worst
        trade.highest_profit_pips = max(trade.highest_profit_pips, profit_pips)
        trade.lowest_profit_pips = min(trade.lowest_profit_pips, profit_pips)

        # Update candle tracking
        self._update_candle_tracking(trade, pip_value)

        # STEP 1: Check for reversal candle (exit immediately if in profit)
        if profit_pips > 1.0:  # At least 1 pip profit
            if self._check_reversal_candle(trade, pip_value):
                LOG.info("REVERSAL EXIT: ticket=%d profit=%.1f pips - strong reversal candle",
                         trade.ticket, profit_pips)
                return "close_reversal"

        # STEP 2: Check for momentum fading
        if profit_pips > 2.0 and trade.momentum_fading:
            LOG.info("MOMENTUM EXIT: ticket=%d profit=%.1f pips - momentum fading",
                     trade.ticket, profit_pips)
            return "close_momentum"

        # STEP 3: Check for excessive chop (no progress)
        if trade.candles_since_entry >= self.max_chop_candles:
            if trade.highest_profit_pips < self.min_progress_pips:
                LOG.info("CHOP EXIT: ticket=%d candles=%d best_profit=%.1f pips - no progress",
                         trade.ticket, trade.candles_since_entry, trade.highest_profit_pips)
                return "close_chop"

        # STEP 4: Breakeven management
        if not trade.breakeven_hit and profit_pips >= self.breakeven_trigger_pips:
            new_sl = self._calculate_breakeven_sl(trade, pip_value)
            if self._update_sl(trade, new_sl, info):
                trade.breakeven_hit = True
                LOG.info("BREAKEVEN: ticket=%d moved SL to %.5f (locked %.1f pips)",
                         trade.ticket, new_sl, self.breakeven_buffer_pips)
                return "update_sl"

        # STEP 5: Aggressive trailing (after breakeven)
        if trade.breakeven_hit and profit_pips >= self.trail_start_pips:
            new_sl = self._calculate_trailing_sl(trade, current_price, pip_value)

            # Only update if it tightens the stop
            should_update = False
            if trade.side == "BUY":
                should_update = new_sl > trade.current_sl
            else:
                should_update = new_sl < trade.current_sl

            if should_update:
                if self._update_sl(trade, new_sl, info):
                    trade.trail_started = True
                    locked_pips = abs(new_sl - trade.entry_price) / pip_value
                    LOG.info("TRAIL: ticket=%d new SL=%.5f (locked %.1f pips)",
                             trade.ticket, new_sl, locked_pips)
                    return "update_sl"

        return "none"

    def _update_candle_tracking(self, trade: ManagedTrade, pip_value: float):
        """Update candle-based tracking for the trade."""
        rates = get_recent_bars(self.mt5, trade.symbol, self.cfg.timeframe, n=10)
        if rates is None or len(rates) < 5:
            return

        # Check if new candle formed
        latest_time = rates[-1]["time"]
        if trade.last_candle_time is None:
            trade.last_candle_time = latest_time
            trade.candles_since_entry = 0
        elif latest_time != trade.last_candle_time:
            trade.last_candle_time = latest_time
            trade.candles_since_entry += 1

        # Track recent candle sizes (body sizes)
        recent_bodies = []
        for i in range(-5, -1):  # Last 4 completed candles
            body = abs(rates[i]["close"] - rates[i]["open"])
            recent_bodies.append(body / pip_value)  # In pips

        trade.recent_candle_sizes = recent_bodies

        # Check if momentum fading (3+ shrinking candles)
        if len(recent_bodies) >= 3:
            shrinking = all(recent_bodies[i] < recent_bodies[i-1]
                          for i in range(1, len(recent_bodies)))
            trade.momentum_fading = shrinking

        # Count candles moving against our position
        consecutive_against = 0
        for i in range(-4, 0):  # Last 4 candles
            candle_bullish = rates[i]["close"] > rates[i]["open"]
            if trade.side == "BUY" and not candle_bullish:
                consecutive_against += 1
            elif trade.side == "SELL" and candle_bullish:
                consecutive_against += 1
            else:
                consecutive_against = 0  # Reset on favorable candle
        trade.consecutive_against_candles = consecutive_against

    def _check_reversal_candle(self, trade: ManagedTrade, pip_value: float) -> bool:
        """
        Check if a strong reversal candle has formed.

        A reversal candle:
        - Large body (>60% of candle range)
        - Closes against our position
        - Forms while we're in profit
        """
        rates = get_recent_bars(self.mt5, trade.symbol, self.cfg.timeframe, n=3)
        if rates is None or len(rates) < 2:
            return False

        # Check the last CLOSED candle (index -2)
        candle = rates[-2]
        body = abs(candle["close"] - candle["open"])
        candle_range = candle["high"] - candle["low"]

        if candle_range == 0:
            return False

        body_ratio = body / candle_range
        candle_bullish = candle["close"] > candle["open"]

        # Strong reversal: large body against our direction
        if body_ratio >= self.reversal_candle_ratio:
            if trade.side == "BUY" and not candle_bullish:
                # Strong bearish candle while we're long
                LOG.debug("Bearish reversal candle detected: body_ratio=%.2f", body_ratio)
                return True
            elif trade.side == "SELL" and candle_bullish:
                # Strong bullish candle while we're short
                LOG.debug("Bullish reversal candle detected: body_ratio=%.2f", body_ratio)
                return True

        return False

    def _calculate_breakeven_sl(self, trade: ManagedTrade, pip_value: float) -> float:
        """Calculate breakeven SL (entry + small buffer)."""
        buffer = self.breakeven_buffer_pips * pip_value

        if trade.side == "BUY":
            return trade.entry_price + buffer
        else:
            return trade.entry_price - buffer

    def _calculate_trailing_sl(self, trade: ManagedTrade,
                                current_price: float, pip_value: float) -> float:
        """Calculate trailing SL (2 pips behind current price)."""
        trail_dist = self.trail_distance_pips * pip_value

        if trade.side == "BUY":
            return current_price - trail_dist
        else:
            return current_price + trail_dist

    def _update_sl(self, trade: ManagedTrade, new_sl: float, info) -> bool:
        """Update the stop loss on MT5."""
        new_sl = round(new_sl, info.digits)

        # Get current position to get current TP
        positions = self.mt5.positions_get(ticket=trade.ticket)
        if not positions:
            return False

        position = positions[0]

        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": trade.symbol,
            "position": trade.ticket,
            "sl": float(new_sl),
            "tp": float(position.tp),
            "magic": self.cfg.magic,
            "comment": "smart_trail"
        }

        res = self.mt5.order_send(req)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            trade.current_sl = new_sl
            return True
        else:
            LOG.warning("SL update failed: ticket=%d res=%s", trade.ticket, res)
            return False

    def close_trade(self, ticket: int, reason: str) -> bool:
        """Close a trade with the given reason."""
        trade = self._trades.get(ticket)
        if trade is None:
            return False

        tick = self.mt5.symbol_info_tick(trade.symbol)
        if tick is None:
            return False

        is_buy = trade.side == "BUY"
        close_price = tick.bid if is_buy else tick.ask
        close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY

        # Get current volume from position
        positions = self.mt5.positions_get(ticket=ticket)
        if not positions:
            return False
        volume = positions[0].volume

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": trade.symbol,
            "volume": float(volume),
            "type": close_type,
            "position": ticket,
            "price": float(close_price),
            "deviation": 10,
            "magic": self.cfg.magic,
            "comment": reason[:20],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        res = self.mt5.order_send(req)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            LOG.info("Trade closed: ticket=%d reason=%s", ticket, reason)
            self.unregister_trade(ticket)
            return True
        else:
            LOG.error("Failed to close trade: ticket=%d reason=%s res=%s", ticket, reason, res)
            return False

    def get_trade_status(self, ticket: int) -> Optional[Dict]:
        """Get current status of a managed trade."""
        trade = self._trades.get(ticket)
        if trade is None:
            return None

        tick = self.mt5.symbol_info_tick(trade.symbol)
        if tick is None:
            return None

        info = self.mt5.symbol_info(trade.symbol)
        pip_value = info.point * 10 if info else 0.0001

        if trade.side == "BUY":
            current_price = tick.bid
            profit_pips = (current_price - trade.entry_price) / pip_value
        else:
            current_price = tick.ask
            profit_pips = (trade.entry_price - current_price) / pip_value

        return {
            "ticket": ticket,
            "symbol": trade.symbol,
            "side": trade.side,
            "entry_price": trade.entry_price,
            "current_price": current_price,
            "profit_pips": profit_pips,
            "highest_profit_pips": trade.highest_profit_pips,
            "breakeven_hit": trade.breakeven_hit,
            "trail_started": trade.trail_started,
            "candles_held": trade.candles_since_entry,
            "momentum_fading": trade.momentum_fading,
            "current_sl": trade.current_sl,
        }
