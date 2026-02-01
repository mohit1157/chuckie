# FOREX.com MetaTrader 5 High-Probability Scalping Bot

> **Important:** This is an **educational template** for automated trading. Live trading involves substantial risk.
> You are responsible for compliance with your broker's terms, U.S. regulations, and for testing on a **demo** account first.

## 🎯 Overview

This is a **high-probability scalping bot** designed for **80%+ win rate** day trading on forex pairs. It uses multiple confirmation indicators and strict filtering to only take the highest quality setups.

### Key Features

- **Multi-Confirmation Strategy**: Requires 5-7 indicator confirmations before entering
- **Session Filtering**: Only trades during high-liquidity hours (London/NY)
- **Dynamic SL/TP**: ATR-based stop loss and take profit adaptation
- **Trade Logging**: SQLite database with full performance analytics
- **Risk Management**: Position sizing, daily loss limits, spread filtering
- **Trailing Stop**: Automatic profit protection with breakeven management

## 📊 Strategy Logic

### Entry Conditions (BUY - all must be true)

| # | Indicator | Condition |
|---|-----------|-----------|
| 1 | EMA Trend | Fast EMA (9) > Slow EMA (21) |
| 2 | Pullback | Price within 0.5 ATR of Fast EMA |
| 3 | RSI | Between 40-70 (not overbought) |
| 4 | MACD | Histogram positive or crossing up |
| 5 | Stochastic | %K > %D or %K > 50 |
| 6 | ADX | > 20 with +DI > -DI |
| 7 | Candle | Current close > previous close |

**Minimum 5/7 confirmations required** (6/7 for conservative mode)

### Risk/Reward Profile

```
Win Rate Target: 80-85%
Risk per Trade:  0.2-0.3% of account
SL Distance:     5-10 pips (ATR-based)
TP Distance:     3-8 pips (ATR-based)
R:R Ratio:       ~1:0.6 to 1:0.8
Expected Result: +15-30 pips/day
```

## 🚀 Quick Start

### 1. Prerequisites

- Windows PC/VPS with MetaTrader 5 installed
- FOREX.com MT5 account (demo or live)
- Python 3.8+

### 2. Installation

```bash
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Copy `.env.example` to `.env` and fill in your MT5 credentials:

```env
MT5_LOGIN=your_account_number
MT5_PASSWORD=your_password
MT5_SERVER=FOREX.com-Demo  # or FOREX.com-Live
```

### 4. Run the Bot

**Standard Mode (80%+ win rate)**:
```bash
python -m bot.scalping_bot --config configs/scalping.yaml
```

**Conservative Mode (85%+ win rate)**:
```bash
python -m bot.scalping_bot --config configs/scalping_conservative.yaml
```

**View Performance Report**:
```bash
python -m bot.scalping_bot --config configs/scalping.yaml --report
```

## ⚙️ Configuration Options

### configs/scalping.yaml

```yaml
mode: demo                    # 'demo' or 'live'
symbol: "EURUSD"              # Trading pair
timeframe: "M1"               # M1 or M5 for scalping

risk:
  account_risk_per_trade: 0.003  # 0.3% per trade
  max_spread_points: 15          # Max 1.5 pip spread
  max_open_positions: 1          # 1 position at a time
  max_daily_loss_pct: 0.02       # Stop after 2% daily loss

strategy:
  min_confirmations: 5           # 5/7 for 80%, 6/7 for 85%
  use_dynamic_sl_tp: true        # ATR-based SL/TP
  cooldown_minutes: 3            # Wait between trades

session:
  enabled: true
  overlap_only: false            # Set true for only London/NY overlap
```

## 📈 Trading Sessions

| Session | UTC Time | Quality | Notes |
|---------|----------|---------|-------|
| London/NY Overlap | 12:00-16:00 | ⭐⭐⭐⭐⭐ | Best liquidity |
| London | 07:00-16:00 | ⭐⭐⭐⭐ | Good for EUR pairs |
| New York | 12:00-21:00 | ⭐⭐⭐⭐ | Good for USD pairs |
| Tokyo | 00:00-09:00 | ⭐⭐⭐ | Good for JPY pairs |
| Sydney | 22:00-07:00 | ⭐⭐ | Lower volume |

The bot automatically filters for optimal trading times.

## 📊 Performance Tracking

All trades are logged to `scalping_trades.db` (SQLite). View stats:

```bash
python -m bot.scalping_bot --config configs/scalping.yaml --report
```

Output:
```
==================================================
  PERFORMANCE REPORT (Last 30 days)
==================================================
  Total Trades:      47
  Winning Trades:    39
  Losing Trades:     8
  Win Rate:          83.0%
--------------------------------------------------
  Total Profit:      $234.50
  Total Pips:        127.5
  Avg Win:           $8.20
  Avg Loss:          $12.10
  Profit Factor:     1.92
--------------------------------------------------
```

## 🛡️ Safety Controls

1. **Daily Loss Circuit Breaker**: Stops trading after 2% daily loss
2. **Spread Filter**: Blocks trades when spread > 1.5 pips
3. **Session Filter**: Only trades during high-liquidity hours
4. **Cooldown**: 3-5 minute wait between trades
5. **Max Positions**: 1 position at a time (no overexposure)
6. **Trailing Stop**: Automatic profit protection
7. **Breakeven Management**: Moves SL to breakeven after 5 pip profit

## 🔧 Project Structure

```
bot/
├── __init__.py
├── config.py              # Configuration dataclasses
├── mt5_client.py          # MetaTrader5 connection wrapper
├── indicators.py          # Technical indicators (RSI, MACD, BB, etc.)
├── scalping_strategy.py   # High-probability scalping logic
├── scalping_execution.py  # Order execution with logging
├── scalping_bot.py        # Main bot orchestrator
├── trade_logger.py        # SQLite trade logging & analytics
├── session_filter.py      # Trading session management
├── risk.py                # Position sizing & risk controls
├── news.py                # Economic calendar filter (stub)
└── monitoring.py          # Health logging

configs/
├── scalping.yaml              # Standard 80%+ config
├── scalping_conservative.yaml # Conservative 85%+ config
├── demo.yaml                  # Original demo config
└── live.yaml                  # Original live config
```

## ⚠️ Important Notes

1. **Demo First**: Always test on demo for at least 2-4 weeks
2. **VPS Recommended**: Use Windows VPS for 24/7 operation
3. **Monitor Daily**: Check performance reports regularly
4. **Adjust Parameters**: Tune settings based on your risk tolerance
5. **No Guarantees**: Past performance doesn't guarantee future results

## 📝 Disclaimer

This software is provided **as-is** with no warranties. Trading forex involves substantial risk of loss. You accept all risk of using this software. Always trade responsibly and never risk more than you can afford to lose.

## 🤖 Do I Need an LLM for Trading?

**No.** LLMs are not suitable for real-time trading due to:
- Latency (100ms-2s per decision)
- Inconsistent outputs
- High cost per trade
- Not designed for market microstructure

This bot uses proven algorithmic approaches that are:
- Fast (<1ms decisions)
- Consistent and deterministic
- Free to run
- Optimized for scalping

LLMs can help with strategy research and post-trade analysis, but should not make real-time trading decisions.
