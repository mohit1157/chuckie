import logging
from datetime import date
from .config import AppConfig
from .mt5_client import MT5Client
from .session_filter import SessionFilter

LOG = logging.getLogger("bot.risk")

class RiskManager:
    def __init__(self, cfg: AppConfig, mt5c: MT5Client):
        self.cfg = cfg
        self.mt5 = mt5c
        self._day = date.today()
        self._start_balance = self.mt5.account_info().balance
        self._trading_halted_today = False
        self._session_filter = SessionFilter()
        self._last_max_pos_log = 0  # Throttle max positions logging

    def refresh_daily_circuit_breaker(self) -> None:
        today = date.today()
        if today != self._day:
            self._day = today
            self._start_balance = self.mt5.account_info().balance
            self._trading_halted_today = False
            LOG.info("New trading day: reset circuit breaker start_balance=%.2f", self._start_balance)

        if self._trading_halted_today:
            return

        bal = self.mt5.account_info().balance
        loss = max(0.0, self._start_balance - bal)
        if self._start_balance > 0 and (loss / self._start_balance) >= self.cfg.risk.max_daily_loss_pct:
            self._trading_halted_today = True
            LOG.error("Daily loss limit hit. start=%.2f current=%.2f halted=true", self._start_balance, bal)

    def can_trade_now(self) -> bool:
        # Check if market is closed (weekend)
        closed, reason = self._session_filter.is_market_closed()
        if closed:
            LOG.info("Risk blocked: market closed - %s", reason)
            return False

        # NOTE: should_avoid_now() disabled for 24/7 testing
        # It blocks during low liquidity (Sydney session)
        # Uncomment for production to avoid low liquidity times
        #
        # avoid, reason = self._session_filter.should_avoid_now()
        # if avoid:
        #     LOG.info("Risk blocked: session avoid - %s", reason)
        #     return False

        if self._trading_halted_today:
            LOG.info("Risk blocked: trading halted today (daily loss limit hit)")
            return False

        tick = self.mt5.symbol_info_tick(self.cfg.symbol)
        info = self.mt5.symbol_info(self.cfg.symbol)
        if tick is None or info is None:
            LOG.info("Risk blocked: cannot get tick/info for %s", self.cfg.symbol)
            return False

        spread_points = int(round((tick.ask - tick.bid) / info.point))
        if spread_points > self.cfg.risk.max_spread_points:
            LOG.info("Risk blocked: spread %d > max %d", spread_points, self.cfg.risk.max_spread_points)
            return False

        positions = self.mt5.positions_get(symbol=self.cfg.symbol)
        if positions and len(positions) >= self.cfg.risk.max_open_positions:
            import time
            now = int(time.time())
            if now - self._last_max_pos_log >= 60:  # Log only once per minute
                LOG.info("Risk blocked: max positions reached (%d/%d)", len(positions), self.cfg.risk.max_open_positions)
                self._last_max_pos_log = now
            return False

        return True

    def calc_lot_size(self, sl_pips: float) -> float:
        # Approx position sizing. For production, compute tick_value precisely per symbol/currency.
        acct = self.mt5.account_info()
        info = self.mt5.symbol_info(self.cfg.symbol)
        if info is None:
            return 0.01

        balance = acct.balance
        risk_amount = balance * self.cfg.risk.account_risk_per_trade

        pip_in_price = info.point * 10.0
        sl_price_distance = sl_pips * pip_in_price

        tick_value = float(getattr(info, "trade_tick_value", 1.0) or 1.0)
        tick_size = float(getattr(info, "trade_tick_size", info.point) or info.point)

        ticks = max(1.0, sl_price_distance / tick_size)
        loss_per_lot = ticks * tick_value

        lots = risk_amount / max(1e-9, loss_per_lot)

        min_lot = float(getattr(info, "volume_min", 0.01) or 0.01)
        max_lot = float(getattr(info, "volume_max", 100.0) or 100.0)
        step = float(getattr(info, "volume_step", 0.01) or 0.01)

        lots = max(min_lot, min(max_lot, lots))
        lots = (lots // step) * step
        lots = float(max(min_lot, lots))

        # Check if we have enough margin for this position using MT5's actual margin calculation
        import MetaTrader5 as mt5

        free_margin = getattr(acct, 'margin_free', None) or getattr(acct, 'free_margin', None) or acct.equity
        tick = self.mt5.symbol_info_tick(self.cfg.symbol)

        if tick and free_margin > 0:
            # Use MT5's order_calc_margin for accurate margin calculation
            margin_required = mt5.order_calc_margin(
                mt5.ORDER_TYPE_BUY,
                self.cfg.symbol,
                lots,
                tick.ask
            )

            if margin_required is not None and margin_required > 0:
                # Keep reducing lots until margin fits within 80% of free margin
                while margin_required > free_margin * 0.8 and lots > min_lot:
                    new_lots = lots * 0.8  # Reduce by 20% each iteration
                    new_lots = max(min_lot, (new_lots // step) * step)
                    if new_lots >= lots:  # Prevent infinite loop
                        new_lots = min_lot
                    LOG.warning("Reducing lots from %.2f to %.2f (margin %.2f > free %.2f)",
                               lots, new_lots, margin_required, free_margin * 0.8)
                    lots = new_lots
                    margin_required = mt5.order_calc_margin(
                        mt5.ORDER_TYPE_BUY,
                        self.cfg.symbol,
                        lots,
                        tick.ask
                    )
                    if margin_required is None:
                        break

        return lots
