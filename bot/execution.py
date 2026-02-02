import logging
from dataclasses import dataclass
import MetaTrader5 as mt5
from .config import AppConfig
from .mt5_client import MT5Client
from .risk import RiskManager

LOG = logging.getLogger("bot.exec")

@dataclass
class Signal:
    side: str  # BUY or SELL
    reason: str

class ExecutionEngine:
    def __init__(self, cfg: AppConfig, mt5c: MT5Client, risk: RiskManager):
        self.cfg = cfg
        self.mt5 = mt5c
        self.risk = risk

    def sync_open_positions(self) -> None:
        return

    def manage_positions(self) -> None:
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

        for p in positions:
            is_buy = (p.type == mt5.POSITION_TYPE_BUY)
            price = tick.bid if is_buy else tick.ask
            moved = (price - p.price_open) if is_buy else (p.price_open - price)
            if moved < start:
                continue

            new_sl = (price - trail) if is_buy else (price + trail)
            if p.sl == 0.0:
                tighten = True
            else:
                tighten = (new_sl > p.sl) if is_buy else (new_sl < p.sl)
            if not tighten:
                continue

            req = {"action": mt5.TRADE_ACTION_SLTP, "symbol": self.cfg.symbol, "position": p.ticket,
                   "sl": float(new_sl), "tp": float(p.tp), "magic": self.cfg.magic, "comment": "trail"}
            res = self.mt5.order_send(req)
            if res is None or res.retcode not in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL):
                LOG.warning("Trailing SL update failed ticket=%s res=%s", p.ticket, res)
            else:
                LOG.info("Trailing SL updated ticket=%s new_sl=%.5f", p.ticket, new_sl)

    def execute_signal(self, sig: Signal) -> None:
        LOG.info(">>> EXECUTE_SIGNAL CALLED: %s %s reason=%s", self.cfg.symbol, sig.side, sig.reason)
        info = self.mt5.symbol_info(self.cfg.symbol)
        tick = self.mt5.symbol_info_tick(self.cfg.symbol)
        if info is None or tick is None:
            return

        lots = self.risk.calc_lot_size(self.cfg.trade.sl_pips)
        if lots <= 0:
            return

        pip_in_price = info.point * 10.0
        sl_dist = self.cfg.trade.sl_pips * pip_in_price
        tp_dist = self.cfg.trade.tp_pips * pip_in_price

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

        req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": self.cfg.symbol, "volume": float(lots),
               "type": order_type, "price": float(price), "sl": float(sl), "tp": float(tp),
               "deviation": 20, "magic": self.cfg.magic, "comment": sig.reason,
               "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_FOK}

        res = self.mt5.order_send(req)
        if res is None:
            LOG.error("Order send failed: %s", self.mt5.last_error())
            return
        if res.retcode not in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED, mt5.TRADE_RETCODE_DONE_PARTIAL):
            LOG.error("Order rejected retcode=%s comment=%s", res.retcode, getattr(res, "comment", ""))
            return

        LOG.info("Order placed side=%s lots=%.2f price=%.5f sl=%.5f tp=%.5f reason=%s",
                 sig.side, lots, price, sl, tp, sig.reason)
