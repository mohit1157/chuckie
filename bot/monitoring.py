import logging
from datetime import datetime, timezone
from .config import AppConfig

LOG = logging.getLogger("bot")

def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

def log_health(mt5, cfg: AppConfig, status: str) -> None:
    acct = mt5.account_info()
    LOG.info(
        "HEALTH status=%s mode=%s symbol=%s balance=%.2f equity=%.2f server=%s time=%s",
        status,
        cfg.mode,
        cfg.symbol,
        getattr(acct, "balance", float("nan")),
        getattr(acct, "equity", float("nan")),
        getattr(acct, "server", "unknown"),
        datetime.now(timezone.utc).isoformat(),
    )
