"""
BTC 15M Bot V1 - ZEUS-style concept, built from transparent rules.

SAFE DEFAULT: PAPER TRADING ONLY.

Strategy:
- EMA 20/50/200 trend filter
- Bollinger Bands for location/volatility
- RSI for momentum
- ATR for dynamic stops/targets
- Recent swing support/resistance filter
- Signals are evaluated only on CLOSED candles (anti-repainting design)

Install:
    pip install ccxt pandas numpy

Run paper bot:
    python btc_zeus_style_bot_v1.py --mode paper

Backtest a CSV containing timestamp,open,high,low,close,volume:
    python btc_zeus_style_bot_v1.py --mode backtest --csv btc_15m.csv

IMPORTANT:
This is educational software, not financial advice. No strategy guarantees profit.
Do not enable live trading until backtesting and paper trading are satisfactory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd


# ---------------------------- CONFIG ----------------------------
SYMBOL = "BTC/USDT"
TIMEFRAME = "15m"
LOOKBACK = 300

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
ATR_PERIOD = 14
SWING_LOOKBACK = 30

# Risk controls
RISK_PER_TRADE = 0.01          # 1% of paper equity
MAX_DAILY_LOSS = 0.03          # stop new entries after -3% day
ATR_STOP_MULT = 1.6
ATR_TARGET_MULT = 2.6
TRAIL_ATR_MULT = 1.8
FEE_RATE = 0.001               # 0.10% placeholder; update for your account
SLIPPAGE = 0.0003              # 0.03% paper assumption
STARTING_EQUITY = 1000.0

STATE_FILE = Path("bot_state.json")
TRADES_FILE = Path("trades.csv")


@dataclass
class Position:
    side: str
    entry: float
    qty: float
    stop: float
    target: float
    highest: float
    lowest: float
    entry_time: str


# ------------------------- INDICATORS --------------------------
def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - prev).abs(),
         (df["low"] - prev).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema20"] = ema(df["close"], EMA_FAST)
    df["ema50"] = ema(df["close"], EMA_MID)
    df["ema200"] = ema(df["close"], EMA_SLOW)
    df["rsi"] = rsi(df["close"], RSI_PERIOD)
    df["bb_mid"] = df["close"].rolling(BB_PERIOD).mean()
    std = df["close"].rolling(BB_PERIOD).std(ddof=0)
    df["bb_upper"] = df["bb_mid"] + BB_STD * std
    df["bb_lower"] = df["bb_mid"] - BB_STD * std
    df["atr"] = atr(df, ATR_PERIOD)
    # Shifted levels ensure the current candle cannot define its own support/resistance.
    df["resistance"] = df["high"].shift(1).rolling(SWING_LOOKBACK).max()
    df["support"] = df["low"].shift(1).rolling(SWING_LOOKBACK).min()
    return df


# ------------------------- SIGNAL LOGIC ------------------------
def signal(df: pd.DataFrame) -> str:
    """Return LONG, SHORT, or FLAT using only the last CLOSED candle."""
    if len(df) < EMA_SLOW + SWING_LOOKBACK + 5:
        return "FLAT"

    a = df.iloc[-2]  # closed candle; latest row may still be forming
    b = df.iloc[-3]

    trend_up = a.ema20 > a.ema50 > a.ema200
    trend_down = a.ema20 < a.ema50 < a.ema200

    # Momentum confirmation rather than raw overbought/oversold entries.
    rsi_rising = a.rsi > b.rsi and a.rsi >= 50
    rsi_falling = a.rsi < b.rsi and a.rsi <= 50

    # Price location: prefer pullbacks toward the BB middle/lower area in trend.
    long_location = a.close <= a.bb_mid * 1.006 and a.close > a.support
    short_location = a.close >= a.bb_mid * 0.994 and a.close < a.resistance

    # Candle confirmation.
    bullish = a.close > a.open and a.close > b.close
    bearish = a.close < a.open and a.close < b.close

    long_ok = trend_up and rsi_rising and long_location and bullish
    short_ok = trend_down and rsi_falling and short_location and bearish

    if long_ok:
        return "LONG"
    if short_ok:
        return "SHORT"
    return "FLAT"


# ----------------------- RISK MANAGEMENT -----------------------
def position_size(equity: float, entry: float, stop: float) -> float:
    risk_cash = equity * RISK_PER_TRADE
    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0:
        return 0.0
    return risk_cash / risk_per_unit


def make_position(side: str, price: float, atr_value: float, now: str, equity: float) -> Position | None:
    if not np.isfinite(atr_value) or atr_value <= 0:
        return None

    if side == "LONG":
        stop = price - ATR_STOP_MULT * atr_value
        target = price + ATR_TARGET_MULT * atr_value
    else:
        stop = price + ATR_STOP_MULT * atr_value
        target = price - ATR_TARGET_MULT * atr_value

    qty = position_size(equity, price, stop)
    if qty <= 0:
        return None

    return Position(side, price, qty, stop, target, price, price, now)


def update_trailing(pos: Position, row: pd.Series) -> None:
    if pos.side == "LONG":
        pos.highest = max(pos.highest, float(row.high))
        candidate = pos.highest - TRAIL_ATR_MULT * float(row.atr)
        pos.stop = max(pos.stop, candidate)
    else:
        pos.lowest = min(pos.lowest, float(row.low))
        candidate = pos.lowest + TRAIL_ATR_MULT * float(row.atr)
        pos.stop = min(pos.stop, candidate)


# -------------------------- BACKTEST ----------------------------
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().reset_index(drop=True)


def run_backtest(df: pd.DataFrame) -> dict:
    df = add_indicators(df)
    equity = STARTING_EQUITY
    peak = equity
    max_dd = 0.0
    pos: Position | None = None
    trades = []
    day_start_equity = equity
    current_day = None

    for i in range(EMA_SLOW + SWING_LOOKBACK + 5, len(df)):
        row = df.iloc[i - 1]  # closed candle
        next_bar = df.iloc[i]  # execution begins on next candle
        day = pd.Timestamp(next_bar.timestamp).date()
        if day != current_day:
            current_day = day
            day_start_equity = equity

        if pos is not None:
            update_trailing(pos, row)
            exit_price = None
            reason = None

            if pos.side == "LONG":
                if next_bar.low <= pos.stop:
                    exit_price, reason = pos.stop, "STOP/TRAIL"
                elif next_bar.high >= pos.target:
                    exit_price, reason = pos.target, "TARGET"
            else:
                if next_bar.high >= pos.stop:
                    exit_price, reason = pos.stop, "STOP/TRAIL"
                elif next_bar.low <= pos.target:
                    exit_price, reason = pos.target, "TARGET"

            if exit_price is not None:
                # Conservative slippage against the position.
                exit_price *= (1 - SLIPPAGE) if pos.side == "LONG" else (1 + SLIPPAGE)
                gross = (exit_price - pos.entry) * pos.qty if pos.side == "LONG" else (pos.entry - exit_price) * pos.qty
                fees = (abs(pos.entry * pos.qty) + abs(exit_price * pos.qty)) * FEE_RATE
                pnl = gross - fees
                equity += pnl
                trades.append({
                    "entry_time": pos.entry_time,
                    "exit_time": str(next_bar.timestamp),
                    "side": pos.side,
                    "entry": pos.entry,
                    "exit": exit_price,
                    "qty": pos.qty,
                    "pnl": pnl,
                    "reason": reason,
                    "equity": equity,
                })
                pos = None

        # New entries only if flat and daily loss limit has not been breached.
        if pos is None and equity > 0 and equity >= day_start_equity * (1 - MAX_DAILY_LOSS):
            s = signal(df.iloc[: i])
            if s in ("LONG", "SHORT"):
                p = make_position(s, float(next_bar.open), float(row.atr), str(next_bar.timestamp), equity)
                if p:
                    # Entry slippage against us.
                    p.entry *= (1 + SLIPPAGE) if s == "LONG" else (1 - SLIPPAGE)
                    pos = p

        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak else 0
        max_dd = max(max_dd, dd)

    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] <= 0)
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = -sum(t["pnl"] for t in trades if t["pnl"] < 0)
    profit_factor = gross_profit / gross_loss if gross_loss else math.inf
    win_rate = wins / len(trades) if trades else 0

    return {
        "starting_equity": STARTING_EQUITY,
        "ending_equity": round(equity, 2),
        "net_pnl": round(equity - STARTING_EQUITY, 2),
        "return_pct": round((equity / STARTING_EQUITY - 1) * 100, 2),
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(win_rate * 100, 2),
        "profit_factor": round(profit_factor, 3) if math.isfinite(profit_factor) else "inf",
        "max_drawdown_pct": round(max_dd * 100, 2),
    }, trades


# -------------------------- LIVE PAPER --------------------------
def make_exchange():
    # Binance.US is selected as the default for a US user. Public market data works
    # without keys; paper mode intentionally does not place orders.
    return ccxt.binanceus({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })


def fetch_ohlcv(exchange, limit=LOOKBACK):
    data = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit)
    return pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"]).assign(
        timestamp=lambda x: pd.to_datetime(x.timestamp, unit="ms", utc=True)
    )


def run_paper():
    exchange = make_exchange()
    equity = STARTING_EQUITY
    position = None
    last_closed_timestamp = None

    print("BTC 15M BOT V1 — PAPER MODE")
    print("No real orders will be sent.")
    print(f"Symbol: {SYMBOL} | Timeframe: {TIMEFRAME}")

    while True:
        try:
            df = add_indicators(fetch_ohlcv(exchange))
            closed = df.iloc[-2]
            ts = str(closed.timestamp)
            if ts == last_closed_timestamp:
                time.sleep(10)
                continue
            last_closed_timestamp = ts

            s = signal(df)
            price = float(closed.close)
            print(f"[{ts}] close={price:.2f} RSI={closed.rsi:.1f} ATR={closed.atr:.2f} signal={s}")

            if position is None and s in ("LONG", "SHORT"):
                position = make_position(s, price, float(closed.atr), ts, equity)
                if position:
                    print("PAPER ENTRY:", asdict(position))
            elif position is not None:
                update_trailing(position, closed)
                print(f"  position={position.side} stop={position.stop:.2f} target={position.target:.2f}")

            # Persist a tiny state file so a restart is visible/auditable.
            STATE_FILE.write_text(json.dumps({
                "last_closed_candle": ts,
                "equity": equity,
                "position": asdict(position) if position else None,
            }, indent=2))

            time.sleep(10)
        except KeyboardInterrupt:
            print("Stopped.")
            break
        except Exception as exc:
            print("ERROR:", repr(exc))
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["paper", "backtest"], default="paper")
    parser.add_argument("--csv", help="CSV for backtest")
    args = parser.parse_args()

    if args.mode == "backtest":
        if not args.csv:
            raise SystemExit("Use --csv path/to/btc_15m.csv")
        df = load_csv(args.csv)
        stats, trades = run_backtest(df)
        print(json.dumps(stats, indent=2))
        if trades:
            pd.DataFrame(trades).to_csv(TRADES_FILE, index=False)
            print(f"Saved trades to {TRADES_FILE}")
    else:
        run_paper()


if __name__ == "__main__":
    main()
