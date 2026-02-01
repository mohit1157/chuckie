# FOREX.com MetaTrader 5 Autonomous Bot (Educational Template)

> **Important:** This is an **educational template** for automated trading. Live trading involves substantial risk.
> You are responsible for compliance with your broker’s terms, U.S. regulations, and for testing on a **demo** account first.

This bot template:
- Connects to **MetaTrader 5 terminal** via the official **MetaTrader5 Python integration**.
- Streams **real-time ticks** from your MT5 terminal.
- Uses a pluggable **strategy** module (default: simple trend + breakout).
- Enforces **risk controls** (position sizing, max daily loss, max open trades, spread filter).
- Places orders, monitors positions, manages **stop-loss / take-profit** and optional **trailing stop**.
- Optional **news filter**: blocks new entries near high-impact events (economic calendar via a pluggable provider).

## Recommended deployment (24/7)
MetaTrader5’s Python bridge requires the **MT5 terminal running on the same machine** (Windows is most common).
Best practice is a **Windows VPS**:
- MT5 stays logged-in and running
- Lower downtime vs a laptop
- Better network stability/latency

## Quick start
1) Install MT5 and log in to your FOREX.com MT5 account  
2) Enable Algo Trading in MT5 (Tools → Options → Expert Advisors)  
3) Create a venv and install deps:
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```
4) Configure `.env`:
- Copy `.env.example` → `.env`
- Fill MT5 credentials and server name shown in MT5

5) Run demo config:
```bash
python -m bot.main --config configs/demo.yaml
```

## Safety controls included
- max spread filter
- max open positions
- max daily loss circuit-breaker (halts trading until next day)
- fixed % risk per trade position sizing (approximation)
- SL/TP set at order send
- optional trailing stop

## “News” note
This template includes a **pluggable** news/economic-calendar filter.
Many institutional feeds are paid. Default provider is a stub that never blocks—replace it with your provider.

## Disclaimer
This software is provided **as-is** with no warranties. You accept all risk of using it.
