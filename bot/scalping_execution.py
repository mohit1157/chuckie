"""
Enhanced Execution Engine for Scalping.
Includes trade logging, dynamic SL/TP, and position monitoring.
"""
import logging
from dataclasses import dataclass
from typing import Optional, Dict
import MetaTrader5 as mt5

from .config import AppConfig
from .mt5_client import MT5Client
from .risk import RiskManager
from .trade_logger import TradeLogger
from .execution import Signal

LOG = logging.getLogger("bot.scalp_exec")


class ScalpingExecutionEngine:
    """
    Enhanced execution engine for scalping with:
    - Trade logging to SQLite
    - Dynamic ATR-based SL/TP
    - Position reconciliation
    - Breakeven management
    - Partial profit taking
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
        self._position_map: Dict[int, int] = {}  # ticket -> trade_id

    def sync_open_positions(self):
        """
        Synchronize open positions with trade logger.
        Detect closed positions and log exits.
        """
        positions = self.mt5.positions_get(symbol=self.cfg.symbol)
        current_tickets = set()

        if positions:
            for p in positions:
                if p.magic == self.cfg.magic:
                    current_tickets.add(p.ticket)

        # Check for closed positions
        closed_tickets = set(self._position_map.keys()) - current_tickets

        for ticket in closed_tickets:
            trade_id = self._position_map.pop(ticket, None)
            if trade_id:
                # Get deal history to find exit details
                self._log_position_close(ticket, trade_id)

    def _log_position_close(self, ticket: int, trade_id: int):
        """Log a closed position's exit details."""
        # Get deals for this position
        from datetime import datetime, timedelta
        from_date = datetime.now() - timedelta(days=1)
        to_date = datetime.now() + timedelta(days=1)

        deals = mt5.history_deals_get(from_date, to_date, group=self.cfg.symbol)
        if not deals:
            LOG.warning("Could not find deal history for ticket %d", ticket)
            return

        for deal in deals:
            if deal.position_id == ticket and deal.entry == mt5.DEAL_ENTRY_OUT:
                profit = deal.profit
                # Calculate pips
                info = self.mt5.symbol_info(self.cfg.symbol)
                pip_value = info.point * 10 if info else 0.0001

                # Determine status
                if profit > 0:
                    status = "closed_tp"
                elif profit < 0:
                    status = "closed_sl"
                else:
                    status = "closed_be"

                pips = 0  # Would need entry price to calculate accurately
                self.logger.log_exit(trade_id, deal.price, profit, pips, status)
                LOG.info("Position closed: ticket=%d profit=%.2f", ticket, profit)
                break

    def execute_signal(self, sig: Signal, dynamic_sl: float = None, dynamic_tp: float = None) -> bool:
        """
        Execute a trading signal with optional dynamic SL/TP.

        Returns:
            True if order was successfully placed.
        """
        info = self.mt5.symbol_info(self.cfg.symbol)
        tick = self.mt5.symbol_info_tick(self.cfg.symbol)
        if info is None or tick is None:
            LOG.error("Cannot get symbol info for %s", self.cfg.symbol)
            return False

        # Use dynamic SL/TP if provided, otherwise use config
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

        # Build order request
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.cfg.symbol,
            "volume": float(lots),
            "type": order_type,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 10,  # Tighter slippage for scalping
            "magic": self.cfg.magic,
            "comment": sig.reason[:31],  # MT5 limits comment to 31 chars
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        # Pre-flight check
        check = self.mt5.order_check(req)
        if check is None or check.retcode != 0:
            LOG.error("Order check failed: %s", check)
            return False

        # Execute order
        res = self.mt5.order_send(req)
        if res is None:
            LOG.error("Order send failed: %s", self.mt5.last_error())
            return False

        if res.retcode not in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED, mt5.TRADE_RETCODE_DONE_PARTIAL):
            LOG.error("Order rejected: retcode=%s comment=%s", res.retcode, getattr(res, "comment", ""))
            return False

        # Log successful trade
        LOG.info("Order executed: %s %.2f lots @ %.5f | SL=%.5f TP=%.5f | reason=%s",
                 sig.side, lots, price, sl, tp, sig.reason)

        # Log to database
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

        # Track position
        self._position_map[res.order] = trade_id

        return True

    def manage_positions(self):
        """
        Manage open positions:
        - Trailing stop
        - Breakeven after X pips profit
        - Partial profit taking
        """
        if not self.cfg.trade.trailing_stop.enabled:
            return

        positions = self.mt5.positions_get(symbol=self.cfg.symbol)
        if not positions:
            return

        info = self.mt5.symbol_info(self.cfg.symbol)
        tick = self.mt5.symbol_info_tick(self.cfg.symbol)
        if info is None or tick is None:
            return

        pip_in_price = info.point * 10.0
        start = self.cfg.trade.trailing_stop.start_pips * pip_in_price
        trail = self.cfg.trade.trailing_stop.trail_pips * pip_in_price
        breakeven_trigger = 5 * pip_in_price  # Move to breakeven after 5 pips

        for p in positions:
            if p.magic != self.cfg.magic:
                continue

            is_buy = (p.type == mt5.POSITION_TYPE_BUY)
            price = tick.bid if is_buy else tick.ask
            moved = (price - p.price_open) if is_buy else (p.price_open - price)

            # Check for breakeven first (after 5 pips profit)
            if moved >= breakeven_trigger and p.sl != 0:
                be_level = p.price_open + (0.5 * pip_in_price) if is_buy else p.price_open - (0.5 * pip_in_price)
                current_sl_profit = (p.sl - p.price_open) if is_buy else (p.price_open - p.sl)

                if current_sl_profit < 0:  # SL is still in loss territory
                    self._update_sl(p, be_level, info)
                    continue

            # Standard trailing stop
            if moved < start:
                continue

            new_sl = (price - trail) if is_buy else (price + trail)

            if p.sl == 0.0:
                tighten = True
            else:
                tighten = (new_sl > p.sl) if is_buy else (new_sl < p.sl)

            if tighten:
                self._update_sl(p, new_sl, info)

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

            tick = self.mt5.symbol_info_tick(self.cfg.symbol)
            if tick is None:
                continue

            close_price = tick.bid if p.type == mt5.POSITION_TYPE_BUY else tick.ask
            close_type = mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY

            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.cfg.symbol,
                "volume": float(p.volume),
                "type": close_type,
                "position": p.ticket,
                "price": float(close_price),
                "deviation": 20,
                "magic": self.cfg.magic,
                "comment": reason,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }

            res = self.mt5.order_send(req)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                LOG.info("Position closed: ticket=%s", p.ticket)
            else:
                LOG.error("Failed to close position: ticket=%s", p.ticket)
