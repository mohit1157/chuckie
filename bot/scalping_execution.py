"""
Enhanced Execution Engine for Scalping.
Includes trade logging, dynamic SL/TP, and advanced exit strategies.

Exit Strategies:
1. Stop Loss / Take Profit (standard)
2. Trailing Stop (locks in profits)
3. EMA Cross Exit (close below EMA9 for longs)
4. Support/Resistance Break Exit
5. Partial Profit Taking (close 80% at 15% profit)
"""
import logging
import re
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Set, Tuple
from datetime import datetime, timezone
import MetaTrader5 as mt5

from .config import AppConfig
from .mt5_client import MT5Client
from .risk import RiskManager
from .trade_logger import TradeLogger
from .execution import Signal
from .data import get_recent_bars
from .indicators import ema

LOG = logging.getLogger("bot.scalp_exec")


@dataclass
class PositionState:
    """Track state for each position."""
    ticket: int
    trade_id: int
    entry_price: float
    entry_time: datetime
    side: str  # "BUY" or "SELL"
    initial_volume: float
    partial_taken: bool = False  # Has 80% been closed?
    sl_locked_at_profit: bool = False  # Has SL been moved to 15% profit?
    ema_crosses_count: int = 0  # Count consecutive EMA crosses (need 2+ to exit)
    min_hold_candles: int = 3  # Minimum candles to hold before EMA exit allowed
    candles_held: int = 0  # Count of candles since entry


class ScalpingExecutionEngine:
    """
    Enhanced execution engine for scalping with:
    - Trade logging to SQLite
    - Dynamic ATR-based SL/TP
    - EMA cross exit (close below EMA9)
    - Support/Resistance break exit
    - Partial profit taking (80% at 15% profit)
    - Trailing stop on remaining 20%
    """

    def __init__(
        self,
        cfg: AppConfig,
        mt5c: MT5Client,
        risk: RiskManager,
        trade_logger: Optional[TradeLogger] = None
    ):
        self.cfg = cfg
        self.mt5 = mt5c
        self.risk = risk
        self.logger = trade_logger or TradeLogger()
        self._positions: Dict[int, PositionState] = {}  # ticket -> PositionState
        self._support_levels: Dict[str, float] = {}  # symbol -> support price
        self._resistance_levels: Dict[str, float] = {}  # symbol -> resistance price

    def sync_open_positions(self):
        """Synchronize open positions with trade logger.

        NOTE: Gets ALL positions with this magic number, not just current symbol.
        This ensures positions are tracked even after symbol switches.
        """
        # Get ALL positions with our magic number (not just current symbol)
        all_positions = self.mt5.positions_get()
        current_tickets = set()

        if all_positions:
            for p in all_positions:
                if p.magic == self.cfg.magic:
                    current_tickets.add(p.ticket)

        # Check for closed positions
        closed_tickets = set(self._positions.keys()) - current_tickets

        for ticket in closed_tickets:
            state = self._positions.pop(ticket, None)
            if state:
                self._log_position_close(ticket, state.trade_id)

    def _log_position_close(self, ticket: int, trade_id: int):
        """Log a closed position's exit details."""
        from datetime import timedelta
        from_date = datetime.now() - timedelta(days=1)
        to_date = datetime.now() + timedelta(days=1)

        deals = mt5.history_deals_get(from_date, to_date, group=self.cfg.symbol)
        if not deals:
            LOG.warning("Could not find deal history for ticket %d", ticket)
            return

        for deal in deals:
            if deal.position_id == ticket and deal.entry == mt5.DEAL_ENTRY_OUT:
                profit = deal.profit
                info = self.mt5.symbol_info(self.cfg.symbol)
                pip_value = info.point * 10 if info else 0.0001

                if profit > 0:
                    status = "closed_tp"
                elif profit < 0:
                    status = "closed_sl"
                else:
                    status = "closed_be"

                pips = 0
                self.logger.log_exit(trade_id, deal.price, profit, pips, status)
                LOG.info("Position closed: ticket=%d profit=%.2f", ticket, profit)
                break

    def _calculate_support_resistance(self):
        """Calculate support and resistance levels from recent price action."""
        rates = get_recent_bars(self.mt5, self.cfg.symbol, self.cfg.timeframe, n=50)
        if rates is None or len(rates) < 20:
            return

        high = np.array([r["high"] for r in rates], dtype=float)
        low = np.array([r["low"] for r in rates], dtype=float)

        # Simple S/R: recent swing high/low
        lookback = 10
        recent_high = np.max(high[-lookback:])
        recent_low = np.min(low[-lookback:])

        self._resistance_levels[self.cfg.symbol] = recent_high
        self._support_levels[self.cfg.symbol] = recent_low

    def _get_current_ema9(self) -> Optional[float]:
        """Get current EMA(9) value."""
        rates = get_recent_bars(self.mt5, self.cfg.symbol, self.cfg.timeframe, n=50)
        if rates is None or len(rates) < 20:
            return None

        close = np.array([r["close"] for r in rates], dtype=float)
        ema9 = ema(close, 9)
        return float(ema9[-1])

    def _check_ema_exit(self, position) -> bool:
        """
        Check if position should be closed due to EMA cross.

        IMPROVED LOGIC:
        - Only triggers when position is IN PROFIT (protect gains, not cut losses)
        - Requires minimum hold time (3 candles) before EMA exit can trigger
        - Requires 2 consecutive candles on wrong side of EMA (not just 1)
        - This prevents immediate exits after entry

        BUY: Exit if 2+ candles close BELOW EMA(9) AND in profit
        SELL: Exit if 2+ candles close ABOVE EMA(9) AND in profit
        """
        if not self.cfg.trade.exit_on_ema_cross:
            return False

        # CRITICAL FIX: Only use EMA exit when IN PROFIT
        # EMA exit is designed to protect gains, not cut losses early
        if position.profit <= 0:
            LOG.debug("EMA exit skipped: position not in profit (%.2f)", position.profit)
            return False

        state = self._positions.get(position.ticket)

        # If no state (bot restarted), create one and skip this cycle
        if state is None:
            LOG.debug("EMA exit skipped: no position state (bot may have restarted)")
            return False

        # Minimum hold time check - don't exit too early
        # Must hold for at least 3 candles (e.g., 9 minutes on M3)
        if state.candles_held < state.min_hold_candles:
            LOG.debug("EMA exit skipped: only held %d/%d candles",
                     state.candles_held, state.min_hold_candles)
            return False

        ema9 = self._get_current_ema9()
        if ema9 is None:
            return False

        # Get last 3 closed candles to check for consecutive crosses
        rates = get_recent_bars(self.mt5, self.cfg.symbol, self.cfg.timeframe, n=5)
        if rates is None or len(rates) < 4:
            return False

        # Check last 2 completed candles (not current forming candle)
        candle_1_close = rates[-2]["close"]  # Most recent completed
        candle_2_close = rates[-3]["close"]  # Second most recent completed

        is_buy = (position.type == mt5.POSITION_TYPE_BUY)

        if is_buy:
            # Need 2 consecutive candles closing below EMA9
            if candle_1_close < ema9 and candle_2_close < ema9:
                LOG.info("EMA EXIT: BUY position - 2 candles closed below EMA9 (%.5f, %.5f < %.5f)",
                         candle_2_close, candle_1_close, ema9)
                return True
            elif candle_1_close < ema9:
                # Only 1 candle below - increment counter but don't exit yet
                state.ema_crosses_count = 1
                LOG.debug("EMA WARNING: 1 candle below EMA9, waiting for confirmation")
            else:
                state.ema_crosses_count = 0  # Reset counter
        else:  # SELL
            # Need 2 consecutive candles closing above EMA9
            if candle_1_close > ema9 and candle_2_close > ema9:
                LOG.info("EMA EXIT: SELL position - 2 candles closed above EMA9 (%.5f, %.5f > %.5f)",
                         candle_2_close, candle_1_close, ema9)
                return True
            elif candle_1_close > ema9:
                state.ema_crosses_count = 1
                LOG.debug("EMA WARNING: 1 candle above EMA9, waiting for confirmation")
            else:
                state.ema_crosses_count = 0

        return False

    def _check_structure_break(self, position) -> bool:
        """
        Check if position should be closed due to structure break.

        BUY: Exit if price breaks below support
        SELL: Exit if price breaks above resistance
        """
        if not self.cfg.trade.exit_on_structure_break:
            return False

        self._calculate_support_resistance()

        support = self._support_levels.get(self.cfg.symbol)
        resistance = self._resistance_levels.get(self.cfg.symbol)

        if support is None or resistance is None:
            return False

        tick = self.mt5.symbol_info_tick(self.cfg.symbol)
        if tick is None:
            return False

        is_buy = (position.type == mt5.POSITION_TYPE_BUY)
        current_price = tick.bid if is_buy else tick.ask

        # Add buffer (0.5 pips) to avoid premature exit
        info = self.mt5.symbol_info(self.cfg.symbol)
        buffer = info.point * 5 if info else 0.00005

        if is_buy and current_price < (support - buffer):
            LOG.info("STRUCTURE EXIT: BUY position - price broke support (%.5f < %.5f)",
                     current_price, support)
            return True

        if not is_buy and current_price > (resistance + buffer):
            LOG.info("STRUCTURE EXIT: SELL position - price broke resistance (%.5f > %.5f)",
                     current_price, resistance)
            return True

        return False

    def _check_partial_profit(self, position) -> bool:
        """
        Check if position should take partial profit.

        At 15% profit:
        - Close 80% of position
        - Move SL to lock in 15% profit
        - Continue trailing remaining 20%
        """
        if not self.cfg.trade.partial_profit.enabled:
            return False

        state = self._positions.get(position.ticket)
        if state is None or state.partial_taken:
            return False

        # Calculate current profit percentage
        account = self.mt5.account_info()
        if account is None:
            return False

        profit_pct = position.profit / account.balance if account.balance > 0 else 0

        if profit_pct >= self.cfg.trade.partial_profit.profit_pct:
            LOG.info("PARTIAL PROFIT: Position at %.1f%% profit - closing 80%%", profit_pct * 100)
            return True

        return False

    def _close_partial_position(self, position, close_pct: float = 0.80):
        """Close a percentage of the position."""
        info = self.mt5.symbol_info(self.cfg.symbol)
        tick = self.mt5.symbol_info_tick(self.cfg.symbol)
        if info is None or tick is None:
            return False

        # Calculate volume to close
        volume_to_close = position.volume * close_pct
        volume_to_close = max(info.volume_min, volume_to_close)
        volume_to_close = (volume_to_close // info.volume_step) * info.volume_step

        is_buy = (position.type == mt5.POSITION_TYPE_BUY)
        close_price = tick.bid if is_buy else tick.ask
        close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.cfg.symbol,
            "volume": float(volume_to_close),
            "type": close_type,
            "position": position.ticket,
            "price": float(close_price),
            "deviation": 10,
            "magic": self.cfg.magic,
            "comment": "partial_80",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        res = self.mt5.order_send(req)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            LOG.info("Partial close: ticket=%s volume=%.2f (80%% closed)",
                     position.ticket, volume_to_close)

            # Update state
            state = self._positions.get(position.ticket)
            if state:
                state.partial_taken = True

            # Move SL to lock in profit
            if self.cfg.trade.partial_profit.move_sl_to_profit:
                self._move_sl_to_profit(position, info)

            return True
        else:
            LOG.error("Partial close failed: ticket=%s res=%s", position.ticket, res)
            return False

    def _move_sl_to_profit(self, position, info):
        """Move SL to lock in 15% profit level."""
        # Calculate the price level that represents 15% profit
        account = self.mt5.account_info()
        if account is None:
            return

        target_profit = account.balance * self.cfg.trade.partial_profit.profit_pct

        # For remaining 20% position, set SL at entry price level that locks profit
        is_buy = (position.type == mt5.POSITION_TYPE_BUY)

        # Move SL to current price minus small buffer (lock in most profit)
        tick = self.mt5.symbol_info_tick(self.cfg.symbol)
        if tick is None:
            return

        current_price = tick.bid if is_buy else tick.ask
        buffer = info.point * 30  # 3 pips buffer

        if is_buy:
            new_sl = current_price - buffer
            # Ensure SL is above entry (in profit)
            new_sl = max(new_sl, position.price_open + info.point * 10)
        else:
            new_sl = current_price + buffer
            # Ensure SL is below entry (in profit)
            new_sl = min(new_sl, position.price_open - info.point * 10)

        new_sl = round(new_sl, info.digits)

        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.cfg.symbol,
            "position": position.ticket,
            "sl": float(new_sl),
            "tp": float(position.tp),
            "magic": self.cfg.magic,
            "comment": "lock_profit"
        }

        res = self.mt5.order_send(req)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            LOG.info("SL moved to profit lock: ticket=%s new_sl=%.5f",
                     position.ticket, new_sl)
            state = self._positions.get(position.ticket)
            if state:
                state.sl_locked_at_profit = True
        else:
            LOG.warning("Failed to move SL to profit: ticket=%s", position.ticket)

    def _close_position(self, position, reason: str):
        """Close a position completely."""
        tick = self.mt5.symbol_info_tick(self.cfg.symbol)
        if tick is None:
            return False

        is_buy = (position.type == mt5.POSITION_TYPE_BUY)
        close_price = tick.bid if is_buy else tick.ask
        close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY

        # Sanitize comment for MT5
        safe_reason = re.sub(r'[^a-zA-Z0-9_]', '_', reason)[:31]

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.cfg.symbol,
            "volume": float(position.volume),
            "type": close_type,
            "position": position.ticket,
            "price": float(close_price),
            "deviation": 10,
            "magic": self.cfg.magic,
            "comment": safe_reason,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        res = self.mt5.order_send(req)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            LOG.info("Position closed: ticket=%s reason=%s", position.ticket, reason)
            return True
        else:
            LOG.error("Failed to close position: ticket=%s reason=%s", position.ticket, reason)
            return False

    def execute_signal(self, sig: Signal, dynamic_sl: float = None, dynamic_tp: float = None) -> bool:
        """Execute a trading signal with optional dynamic SL/TP."""
        info = self.mt5.symbol_info(self.cfg.symbol)
        tick = self.mt5.symbol_info_tick(self.cfg.symbol)
        if info is None or tick is None:
            LOG.error("Cannot get symbol info for %s", self.cfg.symbol)
            return False

        sl_pips = dynamic_sl if dynamic_sl else self.cfg.trade.sl_pips
        tp_pips = dynamic_tp if dynamic_tp else self.cfg.trade.tp_pips

        lots = self.risk.calc_lot_size(sl_pips)

        # Apply size multiplier from signal (for counter-trend trades)
        size_multiplier = getattr(sig, 'size_multiplier', 1.0)
        if size_multiplier < 1.0:
            original_lots = lots
            lots = lots * size_multiplier
            trade_type = getattr(sig, 'trade_type', 'with_trend')
            LOG.info("Adaptive sizing [%s]: %.2f lots -> %.2f lots (%.0f%%)",
                     trade_type, original_lots, lots, size_multiplier * 100)
        if lots <= 0:
            LOG.error("Invalid lot size calculated: %.4f", lots)
            return False

        pip_in_price = info.point * 10.0
        sl_dist = sl_pips * pip_in_price
        tp_dist = tp_pips * pip_in_price

        if sig.side == "BUY":
            price = tick.ask
            sl = price - sl_dist
            tp = price + tp_dist
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            sl = price + sl_dist
            tp = price - tp_dist
            order_type = mt5.ORDER_TYPE_SELL

        # Sanitize comment - MT5 doesn't accept special characters like | % ,
        # Use simple alphanumeric comment, max 20 chars to be safe
        safe_comment = re.sub(r'[^a-zA-Z0-9]', '', sig.reason)[:20]
        LOG.info("Order comment: original='%s' -> sanitized='%s'", sig.reason, safe_comment)

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.cfg.symbol,
            "volume": float(lots),
            "type": order_type,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 10,
            "magic": self.cfg.magic,
            "comment": safe_comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        # Log order details for debugging
        LOG.info("Order request: %s %s %.2f lots @ %.5f SL=%.5f TP=%.5f",
                 sig.side, self.cfg.symbol, lots, price, sl, tp)

        check = self.mt5.order_check(req)
        if check is None:
            # Get last error for more details
            error = mt5.last_error()
            LOG.error("Order check returned None - symbol may not be enabled. Error: %s", error)
            LOG.error("Attempting to enable symbol %s...", self.cfg.symbol)
            # Try to enable the symbol
            if not mt5.symbol_select(self.cfg.symbol, True):
                LOG.error("Failed to enable symbol %s", self.cfg.symbol)
            return False
        if check.retcode != 0:
            LOG.error("Order check failed: retcode=%s comment=%s", check.retcode, getattr(check, 'comment', ''))
            return False

        res = self.mt5.order_send(req)
        if res is None:
            LOG.error("Order send failed: %s", self.mt5.last_error())
            return False

        if res.retcode not in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED, mt5.TRADE_RETCODE_DONE_PARTIAL):
            LOG.error("Order rejected: retcode=%s comment=%s", res.retcode, getattr(res, "comment", ""))
            return False

        LOG.info("Order executed: %s %.2f lots @ %.5f | SL=%.5f TP=%.5f | reason=%s",
                 sig.side, lots, price, sl, tp, sig.reason)

        trade_id = self.logger.log_entry(
            symbol=self.cfg.symbol,
            side=sig.side,
            lots=lots,
            entry_price=price,
            sl_price=sl,
            tp_price=tp,
            reason=sig.reason,
            magic=self.cfg.magic,
            ticket=res.order
        )

        # Track position state
        self._positions[res.order] = PositionState(
            ticket=res.order,
            trade_id=trade_id,
            entry_price=price,
            entry_time=datetime.now(timezone.utc),
            side=sig.side,
            initial_volume=lots,
            partial_taken=False,
            sl_locked_at_profit=False,
        )

        # Calculate initial support/resistance
        self._calculate_support_resistance()

        return True

    def execute_signal_with_ticket(self, sig: Signal, dynamic_sl: float = None,
                                    dynamic_tp: float = None) -> Tuple[bool, Optional[int]]:
        """
        Execute a trading signal and return (success, ticket).

        Same as execute_signal but returns the ticket for trade management.
        """
        info = self.mt5.symbol_info(self.cfg.symbol)
        tick = self.mt5.symbol_info_tick(self.cfg.symbol)
        if info is None or tick is None:
            LOG.error("Cannot get symbol info for %s", self.cfg.symbol)
            return False, None

        sl_pips = dynamic_sl if dynamic_sl else self.cfg.trade.sl_pips
        tp_pips = dynamic_tp if dynamic_tp else self.cfg.trade.tp_pips

        lots = self.risk.calc_lot_size(sl_pips)

        # Apply size multiplier from signal (for counter-trend trades)
        size_multiplier = getattr(sig, 'size_multiplier', 1.0)
        if size_multiplier < 1.0:
            original_lots = lots
            lots = lots * size_multiplier
            trade_type = getattr(sig, 'trade_type', 'with_trend')
            LOG.info("Adaptive sizing [%s]: %.2f lots -> %.2f lots (%.0f%%)",
                     trade_type, original_lots, lots, size_multiplier * 100)
        if lots <= 0:
            LOG.error("Invalid lot size calculated: %.4f", lots)
            return False, None

        pip_in_price = info.point * 10.0
        sl_dist = sl_pips * pip_in_price
        tp_dist = tp_pips * pip_in_price

        if sig.side == "BUY":
            price = tick.ask
            sl = price - sl_dist
            tp = price + tp_dist
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            sl = price + sl_dist
            tp = price - tp_dist
            order_type = mt5.ORDER_TYPE_SELL

        safe_comment = re.sub(r'[^a-zA-Z0-9]', '', sig.reason)[:20]
        LOG.info("Order comment: original='%s' -> sanitized='%s'", sig.reason, safe_comment)

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.cfg.symbol,
            "volume": float(lots),
            "type": order_type,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 10,
            "magic": self.cfg.magic,
            "comment": safe_comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        LOG.info("Order request: %s %s %.2f lots @ %.5f SL=%.5f TP=%.5f",
                 sig.side, self.cfg.symbol, lots, price, sl, tp)

        check = self.mt5.order_check(req)
        if check is None:
            error = mt5.last_error()
            LOG.error("Order check returned None - symbol may not be enabled. Error: %s", error)
            if not mt5.symbol_select(self.cfg.symbol, True):
                LOG.error("Failed to enable symbol %s", self.cfg.symbol)
            return False, None
        if check.retcode != 0:
            LOG.error("Order check failed: retcode=%s comment=%s", check.retcode, getattr(check, 'comment', ''))
            return False, None

        res = self.mt5.order_send(req)
        if res is None:
            LOG.error("Order send failed: %s", self.mt5.last_error())
            return False, None

        if res.retcode not in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED, mt5.TRADE_RETCODE_DONE_PARTIAL):
            LOG.error("Order rejected: retcode=%s comment=%s", res.retcode, getattr(res, "comment", ""))
            return False, None

        LOG.info("Order executed: %s %.2f lots @ %.5f | SL=%.5f TP=%.5f | reason=%s",
                 sig.side, lots, price, sl, tp, sig.reason)

        trade_id = self.logger.log_entry(
            symbol=self.cfg.symbol,
            side=sig.side,
            lots=lots,
            entry_price=price,
            sl_price=sl,
            tp_price=tp,
            reason=sig.reason,
            magic=self.cfg.magic,
            ticket=res.order
        )

        # Track position state
        self._positions[res.order] = PositionState(
            ticket=res.order,
            trade_id=trade_id,
            entry_price=price,
            entry_time=datetime.now(timezone.utc),
            side=sig.side,
            initial_volume=lots,
            partial_taken=False,
            sl_locked_at_profit=False,
        )

        self._calculate_support_resistance()

        return True, res.order

    def manage_positions(self):
        """
        Manage ALL open positions with this magic number (not just current symbol).
        This ensures positions are managed even after symbol switches.

        Advanced exit strategies:
        1. Update candle counters for minimum hold time
        2. Check EMA cross exit (requires 2 consecutive candles + min hold)
        3. Check structure break exit
        4. Check partial profit taking
        5. Apply trailing stop
        """
        # Get ALL positions with our magic number (not just current symbol)
        all_positions = self.mt5.positions_get()
        if not all_positions:
            return

        for p in all_positions:
            if p.magic != self.cfg.magic:
                continue

            # Get symbol info for THIS position's symbol (may differ from cfg.symbol)
            pos_symbol = p.symbol
            info = self.mt5.symbol_info(pos_symbol)
            tick = self.mt5.symbol_info_tick(pos_symbol)
            if info is None or tick is None:
                continue

            # Update candle counter for this position
            state = self._positions.get(p.ticket)
            if state:
                # Calculate candles held based on time difference
                tf_seconds = self._get_timeframe_seconds()
                time_held = (datetime.now(timezone.utc) - state.entry_time).total_seconds()
                state.candles_held = int(time_held / tf_seconds) if tf_seconds > 0 else 0

            # 1. Check EMA cross exit (with min hold time protection)
            # Only check EMA for current symbol (need bars data)
            if pos_symbol == self.cfg.symbol:
                if self._check_ema_exit(p):
                    self._close_position_by_symbol(p, pos_symbol, "ema_cross_exit")
                    continue

                # 2. Check structure break exit
                if self._check_structure_break(p):
                    self._close_position_by_symbol(p, pos_symbol, "structure_break")
                    continue

            # 3. Check partial profit taking
            if self._check_partial_profit(p):
                self._close_partial_position_by_symbol(p, pos_symbol, self.cfg.trade.partial_profit.close_pct)
                # Don't continue - still manage remaining position

            # 4. Apply trailing stop (standard logic)
            self._apply_trailing_stop_by_symbol(p, pos_symbol, info, tick)

    def _close_position_by_symbol(self, position, symbol: str, reason: str):
        """Close a position completely, using the position's actual symbol."""
        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None:
            return False

        is_buy = (position.type == mt5.POSITION_TYPE_BUY)
        close_price = tick.bid if is_buy else tick.ask
        close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY

        # Sanitize comment for MT5
        safe_reason = re.sub(r'[^a-zA-Z0-9_]', '_', reason)[:31]

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(position.volume),
            "type": close_type,
            "position": position.ticket,
            "price": float(close_price),
            "deviation": 10,
            "magic": self.cfg.magic,
            "comment": safe_reason,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        res = self.mt5.order_send(req)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            LOG.info("Position closed: ticket=%s symbol=%s reason=%s", position.ticket, symbol, reason)
            return True
        else:
            LOG.error("Failed to close position: ticket=%s symbol=%s reason=%s", position.ticket, symbol, reason)
            return False

    def _close_partial_position_by_symbol(self, position, symbol: str, close_pct: float = 0.80):
        """Close a percentage of the position using the position's actual symbol."""
        info = self.mt5.symbol_info(symbol)
        tick = self.mt5.symbol_info_tick(symbol)
        if info is None or tick is None:
            return False

        # Calculate volume to close
        volume_to_close = position.volume * close_pct
        volume_to_close = max(info.volume_min, volume_to_close)
        volume_to_close = (volume_to_close // info.volume_step) * info.volume_step

        is_buy = (position.type == mt5.POSITION_TYPE_BUY)
        close_price = tick.bid if is_buy else tick.ask
        close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume_to_close),
            "type": close_type,
            "position": position.ticket,
            "price": float(close_price),
            "deviation": 10,
            "magic": self.cfg.magic,
            "comment": "partial_close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        res = self.mt5.order_send(req)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            LOG.info("Partial close: ticket=%s symbol=%s volume=%.2f",
                     position.ticket, symbol, volume_to_close)
            state = self._positions.get(position.ticket)
            if state:
                state.partial_taken = True
            return True
        else:
            LOG.error("Partial close failed: ticket=%s symbol=%s", position.ticket, symbol)
            return False

    def _apply_trailing_stop_by_symbol(self, position, symbol: str, info, tick):
        """Apply trailing stop logic using the position's actual symbol."""
        if not self.cfg.trade.trailing_stop.enabled:
            return

        pip_in_price = info.point * 10.0
        start = self.cfg.trade.trailing_stop.start_pips * pip_in_price
        trail = self.cfg.trade.trailing_stop.trail_pips * pip_in_price
        breakeven_trigger = 5 * pip_in_price

        is_buy = (position.type == mt5.POSITION_TYPE_BUY)
        price = tick.bid if is_buy else tick.ask
        moved = (price - position.price_open) if is_buy else (position.price_open - price)

        # Check for breakeven (after 5 pips)
        if moved >= breakeven_trigger and position.sl != 0:
            be_level = position.price_open + (0.5 * pip_in_price) if is_buy else position.price_open - (0.5 * pip_in_price)
            current_sl_profit = (position.sl - position.price_open) if is_buy else (position.price_open - position.sl)

            if current_sl_profit < 0:
                self._update_sl_by_symbol(position, symbol, be_level, info)
                return

        # Standard trailing stop
        if moved < start:
            return

        new_sl = (price - trail) if is_buy else (price + trail)

        if position.sl == 0.0:
            tighten = True
        else:
            tighten = (new_sl > position.sl) if is_buy else (new_sl < position.sl)

        if tighten:
            self._update_sl_by_symbol(position, symbol, new_sl, info)

    def _update_sl_by_symbol(self, position, symbol: str, new_sl: float, info):
        """Update stop loss for a position using the position's actual symbol."""
        new_sl = round(new_sl, info.digits)

        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": position.ticket,
            "sl": float(new_sl),
            "tp": float(position.tp),
            "magic": self.cfg.magic,
            "comment": "trail"
        }

        res = self.mt5.order_send(req)
        if res is None or res.retcode not in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL):
            LOG.warning("SL update failed: ticket=%s symbol=%s res=%s", position.ticket, symbol, res)
        else:
            LOG.info("SL updated: ticket=%s symbol=%s new_sl=%.5f", position.ticket, symbol, new_sl)

    def _get_timeframe_seconds(self) -> int:
        """Get timeframe duration in seconds."""
        tf = self.cfg.timeframe.upper()
        tf_map = {
            "M1": 60, "M2": 120, "M3": 180, "M4": 240, "M5": 300,
            "M6": 360, "M10": 600, "M12": 720, "M15": 900, "M20": 1200,
            "M30": 1800, "H1": 3600, "H2": 7200, "H3": 10800, "H4": 14400,
            "H6": 21600, "H8": 28800, "H12": 43200, "D1": 86400,
        }
        return tf_map.get(tf, 180)  # Default to M3

    def _apply_trailing_stop(self, position, info, tick):
        """Apply trailing stop logic."""
        if not self.cfg.trade.trailing_stop.enabled:
            return

        pip_in_price = info.point * 10.0
        start = self.cfg.trade.trailing_stop.start_pips * pip_in_price
        trail = self.cfg.trade.trailing_stop.trail_pips * pip_in_price
        breakeven_trigger = 5 * pip_in_price

        is_buy = (position.type == mt5.POSITION_TYPE_BUY)
        price = tick.bid if is_buy else tick.ask
        moved = (price - position.price_open) if is_buy else (position.price_open - price)

        # Check for breakeven (after 5 pips)
        if moved >= breakeven_trigger and position.sl != 0:
            be_level = position.price_open + (0.5 * pip_in_price) if is_buy else position.price_open - (0.5 * pip_in_price)
            current_sl_profit = (position.sl - position.price_open) if is_buy else (position.price_open - position.sl)

            if current_sl_profit < 0:
                self._update_sl(position, be_level, info)
                return

        # Standard trailing stop
        if moved < start:
            return

        new_sl = (price - trail) if is_buy else (price + trail)

        if position.sl == 0.0:
            tighten = True
        else:
            tighten = (new_sl > position.sl) if is_buy else (new_sl < position.sl)

        if tighten:
            self._update_sl(position, new_sl, info)

    def _update_sl(self, position, new_sl: float, info):
        """Update stop loss for a position."""
        new_sl = round(new_sl, info.digits)

        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.cfg.symbol,
            "position": position.ticket,
            "sl": float(new_sl),
            "tp": float(position.tp),
            "magic": self.cfg.magic,
            "comment": "trail"
        }

        res = self.mt5.order_send(req)
        if res is None or res.retcode not in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL):
            LOG.warning("SL update failed: ticket=%s res=%s", position.ticket, res)
        else:
            LOG.info("SL updated: ticket=%s new_sl=%.5f", position.ticket, new_sl)

    def close_all_positions(self, reason: str = "manual_close"):
        """Close ALL open positions with this magic number (all symbols)."""
        all_positions = self.mt5.positions_get()
        if not all_positions:
            return

        for p in all_positions:
            if p.magic != self.cfg.magic:
                continue
            self._close_position_by_symbol(p, p.symbol, reason)
