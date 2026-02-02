import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from .config import AppConfig

LOG = logging.getLogger("bot")

def setup_logging(level: str) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    log_format = "%(asctime)s %(levelname)s %(name)s | %(message)s"

    # FIX 16: Use absolute path to prevent logging to wrong directory
    # Get the directory where this script is located (bot/)
    bot_dir = Path(__file__).parent
    project_dir = bot_dir.parent  # chuckie/chuckie/
    logs_dir = project_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"bot_{timestamp}.log"

    # Configure root logger with both console and file handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)

    # File handler (with immediate flush)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(file_handler)

    # Store log file path for reference
    os.environ['BOT_LOG_FILE'] = str(log_file)
    logging.getLogger("bot").info(f"Logging to file: {log_file}")

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
