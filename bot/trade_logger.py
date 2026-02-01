"""
Trade Logger with SQLite database for performance tracking.
Tracks all trades and calculates win rate, profit factor, etc.
"""
import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

LOG = logging.getLogger("bot.trade_logger")


@dataclass
class TradeRecord:
    """Single trade record."""
    id: int
    symbol: str
    side: str
    lots: float
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float]
    exit_time: Optional[datetime]
    sl_price: float
    tp_price: float
    profit: Optional[float]
    pips: Optional[float]
    reason: str
    status: str  # 'open', 'closed_tp', 'closed_sl', 'closed_manual'


@dataclass
class PerformanceStats:
    """Performance statistics."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_profit: float
    total_pips: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    avg_trade_duration_minutes: float


class TradeLogger:
    """SQLite-based trade logger with performance analytics."""

    def __init__(self, db_path: str = "trades.db"):
        self.db_path = Path(db_path)
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database with trades table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                lots REAL NOT NULL,
                entry_price REAL NOT NULL,
                entry_time TEXT NOT NULL,
                exit_price REAL,
                exit_time TEXT,
                sl_price REAL NOT NULL,
                tp_price REAL NOT NULL,
                profit REAL,
                pips REAL,
                reason TEXT,
                status TEXT DEFAULT 'open',
                magic INTEGER,
                ticket INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE NOT NULL,
                trades_count INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                total_profit REAL DEFAULT 0,
                total_pips REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
        LOG.info("Trade database initialized: %s", self.db_path)

    def log_entry(
        self,
        symbol: str,
        side: str,
        lots: float,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        reason: str,
        magic: int = 0,
        ticket: int = 0
    ) -> int:
        """Log a new trade entry. Returns trade ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        entry_time = datetime.now(timezone.utc).isoformat()

        cursor.execute("""
            INSERT INTO trades (symbol, side, lots, entry_price, entry_time, sl_price, tp_price, reason, magic, ticket, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
        """, (symbol, side, lots, entry_price, entry_time, sl_price, tp_price, reason, magic, ticket))

        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()

        LOG.info("Trade logged: ID=%d %s %s %.2f lots @ %.5f", trade_id, side, symbol, lots, entry_price)
        return trade_id

    def log_exit(
        self,
        trade_id: int,
        exit_price: float,
        profit: float,
        pips: float,
        status: str = "closed_manual"
    ):
        """Log trade exit with profit/loss."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        exit_time = datetime.now(timezone.utc).isoformat()

        cursor.execute("""
            UPDATE trades
            SET exit_price = ?, exit_time = ?, profit = ?, pips = ?, status = ?
            WHERE id = ?
        """, (exit_price, exit_time, profit, pips, status, trade_id))

        conn.commit()
        conn.close()

        LOG.info("Trade closed: ID=%d profit=%.2f pips=%.1f status=%s", trade_id, profit, pips, status)

        # Update daily summary
        self._update_daily_summary()

    def log_exit_by_ticket(
        self,
        ticket: int,
        exit_price: float,
        profit: float,
        pips: float,
        status: str = "closed_manual"
    ):
        """Log trade exit by MT5 ticket number."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        exit_time = datetime.now(timezone.utc).isoformat()

        cursor.execute("""
            UPDATE trades
            SET exit_price = ?, exit_time = ?, profit = ?, pips = ?, status = ?
            WHERE ticket = ? AND status = 'open'
        """, (exit_price, exit_time, profit, pips, status, ticket))

        conn.commit()
        conn.close()

        LOG.info("Trade closed by ticket: %d profit=%.2f pips=%.1f", ticket, profit, pips)
        self._update_daily_summary()

    def _update_daily_summary(self):
        """Update daily summary table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN profit <= 0 THEN 1 ELSE 0 END) as losses,
                COALESCE(SUM(profit), 0) as total_profit,
                COALESCE(SUM(pips), 0) as total_pips
            FROM trades
            WHERE date(entry_time) = ? AND status != 'open'
        """, (today,))

        row = cursor.fetchone()
        if row and row[0] > 0:
            total, wins, losses, total_profit, total_pips = row
            win_rate = (wins / total * 100) if total > 0 else 0

            cursor.execute("""
                INSERT OR REPLACE INTO daily_summary
                (date, trades_count, winning_trades, losing_trades, total_profit, total_pips, win_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (today, total, wins, losses, total_profit, total_pips, win_rate))

        conn.commit()
        conn.close()

    def get_open_trades(self) -> List[Dict[str, Any]]:
        """Get all open trades."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM trades WHERE status = 'open'")
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_performance_stats(self, days: int = 30) -> PerformanceStats:
        """Calculate performance statistics for the last N days."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN profit <= 0 THEN 1 ELSE 0 END) as losses,
                COALESCE(SUM(profit), 0) as total_profit,
                COALESCE(SUM(pips), 0) as total_pips,
                COALESCE(AVG(CASE WHEN profit > 0 THEN profit END), 0) as avg_win,
                COALESCE(AVG(CASE WHEN profit < 0 THEN profit END), 0) as avg_loss,
                COALESCE(MAX(profit), 0) as largest_win,
                COALESCE(MIN(profit), 0) as largest_loss,
                COALESCE(AVG(
                    CASE WHEN exit_time IS NOT NULL THEN
                        (julianday(exit_time) - julianday(entry_time)) * 24 * 60
                    END
                ), 0) as avg_duration
            FROM trades
            WHERE status != 'open'
            AND entry_time >= datetime('now', '-' || ? || ' days')
        """, (days,))

        row = cursor.fetchone()
        conn.close()

        if row is None or row[0] == 0:
            return PerformanceStats(
                total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
                total_profit=0, total_pips=0, avg_win=0, avg_loss=0,
                profit_factor=0, largest_win=0, largest_loss=0,
                consecutive_wins=0, consecutive_losses=0, avg_trade_duration_minutes=0
            )

        total, wins, losses, total_profit, total_pips, avg_win, avg_loss, largest_win, largest_loss, avg_duration = row

        win_rate = (wins / total * 100) if total > 0 else 0

        # Profit factor
        if avg_loss != 0:
            gross_profit = wins * avg_win if avg_win else 0
            gross_loss = abs(losses * avg_loss) if avg_loss else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        else:
            profit_factor = float('inf') if avg_win > 0 else 0

        # Get consecutive wins/losses
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT profit FROM trades
            WHERE status != 'open'
            ORDER BY exit_time DESC
            LIMIT 50
        """)
        recent_trades = [r[0] for r in cursor.fetchall()]
        conn.close()

        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0

        for profit in recent_trades:
            if profit and profit > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            elif profit and profit < 0:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)

        return PerformanceStats(
            total_trades=total,
            winning_trades=wins or 0,
            losing_trades=losses or 0,
            win_rate=win_rate,
            total_profit=total_profit or 0,
            total_pips=total_pips or 0,
            avg_win=avg_win or 0,
            avg_loss=avg_loss or 0,
            profit_factor=profit_factor,
            largest_win=largest_win or 0,
            largest_loss=largest_loss or 0,
            consecutive_wins=max_consecutive_wins,
            consecutive_losses=max_consecutive_losses,
            avg_trade_duration_minutes=avg_duration or 0
        )

    def print_performance_report(self, days: int = 30):
        """Print formatted performance report."""
        stats = self.get_performance_stats(days)

        print("\n" + "=" * 50)
        print(f"  PERFORMANCE REPORT (Last {days} days)")
        print("=" * 50)
        print(f"  Total Trades:      {stats.total_trades}")
        print(f"  Winning Trades:    {stats.winning_trades}")
        print(f"  Losing Trades:     {stats.losing_trades}")
        print(f"  Win Rate:          {stats.win_rate:.1f}%")
        print("-" * 50)
        print(f"  Total Profit:      ${stats.total_profit:.2f}")
        print(f"  Total Pips:        {stats.total_pips:.1f}")
        print(f"  Avg Win:           ${stats.avg_win:.2f}")
        print(f"  Avg Loss:          ${stats.avg_loss:.2f}")
        print(f"  Profit Factor:     {stats.profit_factor:.2f}")
        print("-" * 50)
        print(f"  Largest Win:       ${stats.largest_win:.2f}")
        print(f"  Largest Loss:      ${stats.largest_loss:.2f}")
        print(f"  Max Consec. Wins:  {stats.consecutive_wins}")
        print(f"  Max Consec. Losses:{stats.consecutive_losses}")
        print(f"  Avg Trade Duration:{stats.avg_trade_duration_minutes:.1f} min")
        print("=" * 50 + "\n")

        return stats
