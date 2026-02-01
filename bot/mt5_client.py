import os
import logging
from dataclasses import dataclass

import MetaTrader5 as mt5
from tenacity import retry, stop_after_attempt, wait_fixed

LOG = logging.getLogger("bot.mt5")

class MT5Error(RuntimeError):
    pass

@dataclass
class Account:
    login: int
    server: str
    balance: float
    equity: float
    currency: str

class MT5Client:
    def __init__(self) -> None:
        self._connected = False

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
    def connect(self) -> None:
        if not mt5.initialize():
            code, msg = mt5.last_error()
            raise MT5Error(f"MT5 initialize failed: {code} {msg}")

        login = os.getenv("MT5_LOGIN")
        password = os.getenv("MT5_PASSWORD")
        server = os.getenv("MT5_SERVER")

        if not login or not password or not server:
            raise MT5Error("Missing MT5_LOGIN / MT5_PASSWORD / MT5_SERVER in environment (.env).")

        ok = mt5.login(int(login), password=password, server=server)
        if not ok:
            code, msg = mt5.last_error()
            raise MT5Error(f"MT5 login failed: {code} {msg}")

        self._connected = True
        LOG.info("Connected to MT5 login=%s server=%s", login, server)

    def shutdown(self) -> None:
        if self._connected:
            mt5.shutdown()
            self._connected = False

    def account_info(self) -> Account:
        info = mt5.account_info()
        if info is None:
            code, msg = mt5.last_error()
            raise MT5Error(f"account_info failed: {code} {msg}")
        return Account(
            login=info.login,
            server=info.server,
            balance=float(info.balance),
            equity=float(info.equity),
            currency=info.currency,
        )

    def symbol_select(self, symbol: str) -> None:
        if not mt5.symbol_select(symbol, True):
            code, msg = mt5.last_error()
            raise MT5Error(f"symbol_select({symbol}) failed: {code} {msg}")

    def symbol_info(self, symbol: str):
        return mt5.symbol_info(symbol)

    def symbol_info_tick(self, symbol: str):
        return mt5.symbol_info_tick(symbol)

    def copy_rates_from_pos(self, symbol: str, timeframe, start_pos: int, count: int):
        return mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)

    def positions_get(self, **kwargs):
        return mt5.positions_get(**kwargs)

    def order_send(self, request: dict):
        return mt5.order_send(request)

    def order_check(self, request: dict):
        return mt5.order_check(request)

    def last_error(self):
        return mt5.last_error()
