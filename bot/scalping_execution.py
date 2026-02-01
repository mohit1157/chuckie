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
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Set
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
        """Synchronize open positions with trade logger."""
        positions = self.mt5.positions_get(symbol=self.cfg.symbol)
        current_tickets = set()

        if positions:
            for p in positions:
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

        BUY: Exit if candle closes BELOW EMA(9)
        SELL: Exit if candle closes ABOVE EMA(9)
        """
        if not self.cfg.trade.exit_on_ema_cross:
            return False

        ema9 = self._get_current_ema9()
        if ema9 is None:
            return False

        # Get last closed candle
        rates = get_recent_bars(self.mt5, self.cfg.symbol, self.cfg.timeframe, n=5)
        if rates is None or len(rates) < 2:
            return False

        last_close = rates[-2]["close"]  # Previous completed candle

        is_buy = (position.type == mt5.POSITION_TYPE_BUY)

        if is_buy and last_close < ema9:
            LOG.info("EMA EXIT: BUY position - candle closed below EMA9 (%.5f < %.5f)",
                     last_close, ema9)
            return True

        if not is_buy and last_close > ema9:
            LOG.info("EMA EXIT: SELL position - candle closed above EMA9 (%.5f > %.5f)",
                     last_close, ema9)
            return True

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
            "comment": "partial_80%",
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

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.cfg.symbol,
            "volume": float(position.volume),
            "type": close_type,
            "position": position.ticket,
            "price": float(close_price),
            "deviation": 10,
            "magic": self.cfg.magic,
            "comment": reason[:31],
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
            "comment": sig.reason[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        check = self.mt5.order_check(req)
        if check is None or check.retcode != 0:
            LOG.error("Order check failed: %s", check)
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

    def manage_positions(self):
        """
        Manage open positions with advanced exit strategies:
        1. Check EMA cross exit
        2. Check structure break exit
        3. Check partial profit taking
        4. Apply trailing stop
        """
        positions = self.mt5.positions_get(symbol=self.cfg.symbol)
        if not positions:
            return

        info = self.mt5.symbol_info(self.cfg.symbol)
        tick = self.mt5.symbol_info_tick(self.cfg.symbol)
        if info is None or tick is None:
            return

        for p in positions:
            if p.magic != self.cfg.magic:
                continue

            # 1. Check EMA cross exit
            if self._check_ema_exit(p):
                self._close_position(p, "ema_cross_exit")
                continue

            # 2. Check structure break exit
            if self._check_structure_break(p):
                self._close_position(p, "structure_break")
                continue

            # 3. Check partial profit taking
            if self._check_partial_profit(p):
                self._close_partial_position(p, self.cfg.trade.partial_profit.close_pct)
                # Don't continue - still manage remaining position

            # 4. Apply trailing stop (standard logic)
            self._apply_trailing_stop(p, info, tick)

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
        """Close all open positions for this strategy."""
        positions = self.mt5.positions_get(symbol=self.cfg.symbol)
        if not positions:
            return

        for p in positions:
            if p.magic != self.cfg.magic:
                continue
            self._close_position(p, reason)
