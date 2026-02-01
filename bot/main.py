import argparse
import time
from pathlib import Path
from dotenv import load_dotenv

from .config import AppConfig
from .monitoring import setup_logging, log_health
from .mt5_client import MT5Client
from .execution import ExecutionEngine
from .strategy import StrategyEngine
from .news import NewsFilter
from .risk import RiskManager

def run_loop(cfg: AppConfig) -> None:
    load_dotenv()
    setup_logging(cfg.logging.level)

    mt5 = MT5Client()
    mt5.connect()

    risk = RiskManager(cfg, mt5)
    news_filter = NewsFilter(cfg)
    strat = StrategyEngine(cfg, mt5)
    exec_engine = ExecutionEngine(cfg, mt5, risk)

    log_health(mt5, cfg, "STARTED")

    try:
        while True:
            risk.refresh_daily_circuit_breaker()
            exec_engine.sync_open_positions()
            exec_engine.manage_positions()

            if not risk.can_trade_now():
                time.sleep(1.0)
                continue

            if cfg.news_filter.enabled and news_filter.block_new_entries():
                time.sleep(1.0)
                continue

            sig = strat.get_signal()
            if sig is not None:
                exec_engine.execute_signal(sig)

            time.sleep(1.0)

    except KeyboardInterrupt:
        log_health(mt5, cfg, "STOPPED: KeyboardInterrupt")
    except Exception as e:
        log_health(mt5, cfg, f"CRASHED: {e}")
        raise
    finally:
        mt5.shutdown()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config (e.g., configs/demo.yaml)")
    args = ap.parse_args()
    cfg = AppConfig.from_yaml(Path(args.config).resolve())
    run_loop(cfg)

if __name__ == "__main__":
    main()
