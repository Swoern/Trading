"""
database.py — SQLite opslag voor paper trades en backtest resultaten.
Alle data wordt lokaal opgeslagen in tradeai.db.
"""
import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime

DB_PATH = Path(os.environ.get("TRADEAI_DB_PATH", "tradeai.db"))


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Maak de tabellen aan als ze nog niet bestaan."""
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at      TEXT    NOT NULL,
            asset           TEXT    NOT NULL,
            direction       TEXT    NOT NULL,
            entry_price     REAL    NOT NULL,
            stop_loss       REAL    NOT NULL,
            take_profit     REAL    NOT NULL,
            capital         REAL    DEFAULT 1000.0,
            reason          TEXT    DEFAULT '',
            status          TEXT    DEFAULT 'open',
            closed_at       TEXT,
            exit_price      REAL,
            result_pct      REAL,
            result_euro     REAL,
            exit_reason     TEXT,
            mistake_analysis TEXT   DEFAULT ''
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS backtest_results (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at          TEXT    NOT NULL,
            asset           TEXT    NOT NULL,
            start_date      TEXT,
            end_date        TEXT,
            total_trades    INTEGER,
            winning_trades  INTEGER,
            losing_trades   INTEGER,
            winrate         REAL,
            total_pnl       REAL,
            avg_win         REAL,
            avg_loss        REAL,
            max_drawdown    REAL,
            strategy_params TEXT
        )
    """)

    conn.commit()
    conn.close()


def add_paper_trade(asset: str, direction: str, entry_price: float,
                    stop_loss: float, take_profit: float,
                    reason: str = "", capital: float = 1000.0) -> int:
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO paper_trades
            (created_at, asset, direction, entry_price, stop_loss, take_profit, reason, capital)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now().isoformat(), asset, direction,
          entry_price, stop_loss, take_profit, reason, capital))
    trade_id = c.lastrowid
    conn.commit()
    conn.close()
    return trade_id


def close_paper_trade(trade_id: int, exit_price: float,
                      exit_reason: str = "manual",
                      mistake_analysis: str = "") -> bool:
    conn = get_connection()
    trade = conn.execute(
        "SELECT * FROM paper_trades WHERE id = ?", (trade_id,)
    ).fetchone()

    if not trade or trade["status"] == "closed":
        conn.close()
        return False

    if trade["direction"] == "long":
        result_pct = (exit_price - trade["entry_price"]) / trade["entry_price"] * 100
    else:
        result_pct = (trade["entry_price"] - exit_price) / trade["entry_price"] * 100

    result_euro = result_pct / 100 * trade["capital"]

    conn.execute("""
        UPDATE paper_trades
        SET status = 'closed', closed_at = ?, exit_price = ?,
            result_pct = ?, result_euro = ?, exit_reason = ?, mistake_analysis = ?
        WHERE id = ?
    """, (datetime.now().isoformat(), exit_price,
          round(result_pct, 4), round(result_euro, 2),
          exit_reason, mistake_analysis, trade_id))
    conn.commit()
    conn.close()
    return True


def get_paper_trades(status: str = None) -> list[dict]:
    conn = get_connection()
    if status:
        rows = conn.execute(
            "SELECT * FROM paper_trades WHERE status = ? ORDER BY created_at DESC",
            (status,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM paper_trades ORDER BY created_at DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_backtest_result(asset: str, start_date, end_date,
                         metrics: dict, strategy_params: dict = None):
    conn = get_connection()
    conn.execute("""
        INSERT INTO backtest_results
            (run_at, asset, start_date, end_date, total_trades, winning_trades,
             losing_trades, winrate, total_pnl, avg_win, avg_loss, max_drawdown, strategy_params)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(), asset, str(start_date), str(end_date),
        metrics["total_trades"], metrics["winning_trades"], metrics["losing_trades"],
        metrics["winrate"], metrics["total_pnl"], metrics["avg_win"],
        metrics["avg_loss"], metrics["max_drawdown"],
        json.dumps(strategy_params or {})
    ))
    conn.commit()
    conn.close()


def get_backtest_results() -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM backtest_results ORDER BY run_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
