"""
database.py — SQLite opslag voor paper trades en backtest resultaten.
Alle data wordt lokaal opgeslagen in tradeai.db.
"""
import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from src.trading_costs import apply_round_trip_costs

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

_AMSTERDAM = ZoneInfo("Europe/Amsterdam")

DB_PATH = Path(os.environ.get("TRADEAI_DB_PATH", "tradeai.db"))


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # timeout=60: wacht langer als de bot en webapp tegelijk willen schrijven.
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 60000")
    return conn


def init_database():
    """Maak de tabellen aan als ze nog niet bestaan."""
    conn = get_connection()
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at       TEXT    NOT NULL,
            asset            TEXT    NOT NULL,
            direction        TEXT    NOT NULL,
            entry_price      REAL    NOT NULL,
            stop_loss        REAL    NOT NULL,
            take_profit      REAL    NOT NULL,
            capital          REAL    DEFAULT 1000.0,
            reason           TEXT    DEFAULT '',
            status           TEXT    DEFAULT 'open',
            closed_at        TEXT,
            exit_price       REAL,
            result_pct       REAL,
            result_euro      REAL,
            exit_reason      TEXT,
            mistake_analysis TEXT    DEFAULT ''
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

    c.execute("""
        CREATE TABLE IF NOT EXISTS bot_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at   TEXT    NOT NULL,
            message     TEXT    NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS bot_learning (
            key     TEXT PRIMARY KEY,
            value   TEXT NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS bot_learning_samples (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at   TEXT NOT NULL,
            source       TEXT NOT NULL,
            asset        TEXT NOT NULL,
            strategy     TEXT,
            timeframe    TEXT,
            context_key  TEXT NOT NULL,
            context_json TEXT DEFAULT '{}',
            outcome      REAL NOT NULL,
            result_pct   REAL DEFAULT 0.0,
            result_euro  REAL DEFAULT 0.0,
            weight       REAL DEFAULT 1.0,
            trade_id     INTEGER,
            closed_at    TEXT,
            UNIQUE(source, trade_id) ON CONFLICT IGNORE
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS bot_learning_scorecards (
            context_key    TEXT PRIMARY KEY,
            asset          TEXT,
            strategy       TEXT,
            context_json   TEXT DEFAULT '{}',
            sample_count   INTEGER DEFAULT 0,
            weighted_score REAL DEFAULT 0.5,
            avg_result_pct REAL DEFAULT 0.0,
            updated_at     TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS bot_learning_decisions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at   TEXT NOT NULL,
            asset        TEXT,
            strategy     TEXT,
            action       TEXT,
            score        REAL,
            sample_count INTEGER DEFAULT 0,
            context_key  TEXT,
            context_json TEXT DEFAULT '{}',
            reason       TEXT DEFAULT '',
            details_json TEXT DEFAULT '{}',
            trade_id     INTEGER
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS bot_bootstrap_runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at          TEXT NOT NULL,
            asset           TEXT NOT NULL,
            timeframe       TEXT,
            samples_created INTEGER DEFAULT 0,
            live_closed_count INTEGER DEFAULT 0,
            reason          TEXT DEFAULT ''
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS bot_api_health (
            asset      TEXT PRIMARY KEY,
            checked_at TEXT NOT NULL,
            status     TEXT NOT NULL,
            message    TEXT DEFAULT ''
        )
    """)

    # Advies van de optionele LLM-adviseslaag (Fase 3). Eén regel per panel-run.
    c.execute("""
        CREATE TABLE IF NOT EXISTS llm_advice (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at  TEXT    NOT NULL,
            asset       TEXT    NOT NULL,
            bias        REAL    DEFAULT 0.0,
            confidence  REAL    DEFAULT 0.0,
            min_conf_delta REAL DEFAULT 0.0,
            rationale   TEXT    DEFAULT '',
            analysts_json TEXT  DEFAULT '{}',
            model       TEXT    DEFAULT ''
        )
    """)

    # Extra kolommen voor bestaande paper_trades tabel (veilig via try/except)
    extra_kolommen = [
        "ALTER TABLE paper_trades ADD COLUMN source TEXT DEFAULT 'manual'",
        "ALTER TABLE paper_trades ADD COLUMN strategy TEXT",
        "ALTER TABLE paper_trades ADD COLUMN market_context TEXT DEFAULT '{}'",
        "ALTER TABLE paper_trades ADD COLUMN learning_score REAL",
        "ALTER TABLE paper_trades ADD COLUMN learning_samples INTEGER DEFAULT 0",
        "ALTER TABLE paper_trades ADD COLUMN learning_reason TEXT DEFAULT ''",
        "ALTER TABLE paper_trades ADD COLUMN original_capital REAL",
        "ALTER TABLE paper_trades ADD COLUMN partial_tp_hit INTEGER DEFAULT 0",
        "ALTER TABLE paper_trades ADD COLUMN partial_pnl_euro REAL DEFAULT 0.0",
        "ALTER TABLE paper_trades ADD COLUMN gross_result_euro REAL",
        "ALTER TABLE paper_trades ADD COLUMN costs_euro REAL DEFAULT 0.0",
    ]
    for stmt in extra_kolommen:
        try:
            c.execute(stmt)
        except sqlite3.OperationalError:
            pass  # Kolom bestaat al

    extra_bootstrap_kolommen = [
        "ALTER TABLE bot_bootstrap_runs ADD COLUMN live_closed_count INTEGER DEFAULT 0",
    ]
    for stmt in extra_bootstrap_kolommen:
        try:
            c.execute(stmt)
        except sqlite3.OperationalError:
            pass  # Kolom bestaat al

    conn.commit()
    conn.close()


def add_paper_trade(
    asset: str,
    direction: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    reason: str = "",
    capital: float = 1000.0,
    source: str = "manual",
    strategy: str = None,
    market_context: dict | None = None,
    learning_score: float | None = None,
    learning_samples: int = 0,
    learning_reason: str = "",
) -> int:
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO paper_trades
            (created_at, asset, direction, entry_price, stop_loss, take_profit,
             reason, capital, source, strategy, market_context,
             learning_score, learning_samples, learning_reason, original_capital)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(_AMSTERDAM).isoformat(), asset, direction, entry_price, stop_loss, take_profit,
        reason, capital, source, strategy, json.dumps(market_context or {}),
        learning_score, learning_samples, learning_reason or "", capital,
    ))
    trade_id = c.lastrowid
    conn.commit()
    conn.close()
    return trade_id


def save_llm_advice(
    asset: str,
    bias: float,
    confidence: float,
    min_conf_delta: float,
    rationale: str = "",
    analysts: dict | None = None,
    model: str = "",
) -> int:
    """Sla één LLM-paneladvies op (Fase 3)."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO llm_advice
            (created_at, asset, bias, confidence, min_conf_delta, rationale, analysts_json, model)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(_AMSTERDAM).isoformat(), asset, float(bias), float(confidence),
        float(min_conf_delta), rationale[:2000], json.dumps(analysts or {}), model,
    ))
    advice_id = c.lastrowid
    conn.commit()
    conn.close()
    return advice_id


def get_latest_llm_advice(asset: str) -> dict | None:
    """Haal het meest recente LLM-advies voor een asset op (of None)."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM llm_advice WHERE asset = ? ORDER BY id DESC LIMIT 1", (asset,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_llm_advice_log(limit: int = 20) -> list[dict]:
    """Haal de laatste LLM-adviezen op voor het dashboard."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM llm_advice ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_trade_sl(trade_id: int, new_sl: float) -> bool:
    """Update de stop-loss van een open trade (voor trailing stop)."""
    conn = get_connection()
    trade = conn.execute(
        "SELECT id FROM paper_trades WHERE id = ? AND status = 'open'", (trade_id,)
    ).fetchone()
    if not trade:
        conn.close()
        return False
    conn.execute(
        "UPDATE paper_trades SET stop_loss = ? WHERE id = ?",
        (round(new_sl, 6), trade_id)
    )
    conn.commit()
    conn.close()
    return True


def update_trade_partial_tp(trade_id: int, partial_pnl_euro: float, new_sl: float) -> bool:
    """
    Registreer een gedeeltelijke TP: halveer het kapitaal, zet SL op break-even
    en sla de gedeeltelijke winst op. Mag maar één keer per trade worden uitgevoerd.
    """
    conn = get_connection()
    trade = conn.execute(
        "SELECT * FROM paper_trades WHERE id = ? AND status = 'open'", (trade_id,)
    ).fetchone()
    if not trade:
        conn.close()
        return False
    if trade["partial_tp_hit"]:
        conn.close()
        return False
    nieuwe_capital = round(float(trade["capital"]) / 2, 2)
    conn.execute(
        """UPDATE paper_trades
           SET capital = ?, stop_loss = ?, partial_tp_hit = 1, partial_pnl_euro = ?
           WHERE id = ?""",
        (nieuwe_capital, round(new_sl, 6), round(partial_pnl_euro, 2), trade_id)
    )
    conn.commit()
    conn.close()
    return True


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
        price_result_pct = (exit_price - trade["entry_price"]) / trade["entry_price"] * 100
    else:
        price_result_pct = (trade["entry_price"] - exit_price) / trade["entry_price"] * 100

    remaining_result_euro = price_result_pct / 100 * trade["capital"]
    partial_pnl = float(trade["partial_pnl_euro"] or 0.0)
    gross_result_euro = remaining_result_euro + partial_pnl
    if trade["original_capital"]:
        original_capital = float(trade["original_capital"])
    elif trade["partial_tp_hit"]:
        original_capital = float(trade["capital"] or 0.0) * 2
    else:
        original_capital = float(trade["capital"] or 0.0)
    result_euro, costs_euro = apply_round_trip_costs(gross_result_euro, original_capital)
    result_pct = (result_euro / original_capital * 100) if original_capital > 0 else price_result_pct

    conn.execute("""
        UPDATE paper_trades
        SET status = 'closed', closed_at = ?, exit_price = ?,
            result_pct = ?, result_euro = ?, exit_reason = ?, mistake_analysis = ?,
            gross_result_euro = ?, costs_euro = ?
        WHERE id = ?
    """, (datetime.now(_AMSTERDAM).isoformat(), exit_price,
          round(result_pct, 4), round(result_euro, 2),
          exit_reason, mistake_analysis, round(gross_result_euro, 2),
          round(costs_euro, 2), trade_id))
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


# ── Bot logging ───────────────────────────────────────────────────────────────

def bot_log(message: str, level: str = "info") -> None:
    """Schrijf een bot-logbericht naar de database."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=0.2)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 200")
        conn.execute(
            "INSERT INTO bot_logs (logged_at, message) VALUES (?, ?)",
            (datetime.now(_AMSTERDAM).isoformat(), f"[{level.upper()}] {message}")
        )
        conn.commit()
        conn.close()
        # Houd de tabel klein: maximaal 2000 regels bewaren
        conn2 = sqlite3.connect(DB_PATH, timeout=0.2)
        conn2.execute("PRAGMA busy_timeout = 200")
        conn2.execute("""
            DELETE FROM bot_logs WHERE id NOT IN (
                SELECT id FROM bot_logs ORDER BY id DESC LIMIT 2000
            )
        """)
        conn2.commit()
        conn2.close()
    except Exception:
        pass


def get_bot_log(limit: int = 100) -> list[dict]:
    """Haal de laatste N bot-logregels op, nieuwste eerst."""
    try:
        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM bot_logs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


# ── Auto trades (bot-gestuurde trades) ───────────────────────────────────────

def get_auto_trades(status: str = None) -> list[dict]:
    """Geef uitsluitend bot-trades (source='auto') terug uit de paper_trades tabel."""
    conn = get_connection()
    if status:
        rows = conn.execute(
            "SELECT * FROM paper_trades WHERE source = 'auto' AND status = ? ORDER BY created_at DESC",
            (status,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM paper_trades WHERE source = 'auto' ORDER BY created_at DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_trade_by_id(trade_id: int) -> dict | None:
    """Geef één trade terug op basis van het ID."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM paper_trades WHERE id = ?", (trade_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# ── Bot leergeheugen ──────────────────────────────────────────────────────────

def _bot_learning_columns(conn) -> set[str]:
    return {row["name"] for row in conn.execute("PRAGMA table_info(bot_learning)").fetchall()}


def _bot_learning_is_key_value(conn) -> bool:
    cols = _bot_learning_columns(conn)
    return "key" in cols and "value" in cols


def get_bot_learning() -> dict:
    """Laad het volledige leergeheugen als dictionary."""
    conn = None
    try:
        conn = get_connection()
        result = {}

        if _bot_learning_is_key_value(conn):
            rows = conn.execute("SELECT key, value FROM bot_learning").fetchall()
            for row in rows:
                try:
                    result[row["key"]] = json.loads(row["value"])
                except Exception:
                    result[row["key"]] = row["value"]
            return result

        # Compatibiliteit met de oude bot_learning tabel op de Raspberry Pi.
        row = conn.execute("SELECT * FROM bot_learning ORDER BY id LIMIT 1").fetchone()
        if not row:
            return {}
        aanpassingen_raw = row["aanpassingen"] or "{}"
        try:
            aanpassingen = json.loads(aanpassingen_raw)
        except Exception:
            aanpassingen = {}
        result.update({
            "min_rr": row["min_rr"],
            "cooling_tot": row["cooling_tot"],
            "winrate_recent": row["winrate_recent"],
            "aanpassingen": aanpassingen,
        })
        return result
    except Exception:
        return {}
    finally:
        if conn:
            conn.close()


def save_bot_learning(data: dict) -> None:
    """Sla het volledige leergeheugen op (overschrijft bestaande waarden)."""
    conn = None
    try:
        conn = get_connection()
        if _bot_learning_is_key_value(conn):
            for key, value in data.items():
                conn.execute(
                    "INSERT OR REPLACE INTO bot_learning (key, value) VALUES (?, ?)",
                    (key, json.dumps(value))
                )
            conn.commit()
            return

        # Compatibiliteit met de oude schema-vorm:
        # id, bijgewerkt, min_rr, cooling_tot, winrate_recent, aanpassingen.
        row = conn.execute("SELECT * FROM bot_learning ORDER BY id LIMIT 1").fetchone()
        bestaande_aanpassingen = {}
        if row and row["aanpassingen"]:
            try:
                bestaande_aanpassingen = json.loads(row["aanpassingen"])
            except Exception:
                bestaande_aanpassingen = {}

        aanpassingen = data.get("aanpassingen", bestaande_aanpassingen) or {}
        if not isinstance(aanpassingen, dict):
            aanpassingen = {"waarde": aanpassingen}

        bekende_velden = {"min_rr", "cooling_tot", "winrate_recent", "aanpassingen"}
        for key, value in data.items():
            if key not in bekende_velden:
                aanpassingen[key] = value

        payload = (
            datetime.now(_AMSTERDAM).isoformat(),
            data.get("min_rr", row["min_rr"] if row else 2.0),
            data.get("cooling_tot", row["cooling_tot"] if row else None),
            data.get("winrate_recent", row["winrate_recent"] if row else 0.5),
            json.dumps(aanpassingen),
        )

        if row:
            conn.execute(
                """
                UPDATE bot_learning
                SET bijgewerkt = ?, min_rr = ?, cooling_tot = ?,
                    winrate_recent = ?, aanpassingen = ?
                WHERE id = ?
                """,
                (*payload, row["id"]),
            )
        else:
            conn.execute(
                """
                INSERT INTO bot_learning
                    (bijgewerkt, min_rr, cooling_tot, winrate_recent, aanpassingen)
                VALUES (?, ?, ?, ?, ?)
                """,
                payload,
            )
        conn.commit()
    except Exception:
        pass
    finally:
        if conn:
            conn.close()


# ── Dagelijkse samenvattingen (compat) ───────────────────────────────────────

def get_daily_summaries() -> list[dict]:
    """Geef gesimuleerde dagelijkse P&L terug op basis van gesloten trades."""
    try:
        conn = get_connection()
        rows = conn.execute("""
            SELECT substr(closed_at, 1, 10) as dag,
                   COUNT(*) as trades,
                   SUM(result_euro) as pnl_euro,
                   SUM(CASE WHEN result_pct > 0 THEN 1 ELSE 0 END) as wins
            FROM paper_trades
            WHERE status = 'closed' AND closed_at IS NOT NULL
            GROUP BY dag
            ORDER BY dag DESC
        """).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def save_daily_summary(datum: str, inhoud: str) -> None:
    """Sla een dagelijkse samenvatting op in het leergeheugen."""
    try:
        sleutel = f"dagrapport_{datum}"
        data = get_bot_learning()
        aanpassingen = data.get("aanpassingen") if isinstance(data.get("aanpassingen"), dict) else {}
        aanpassingen[sleutel] = inhoud
        data["aanpassingen"] = aanpassingen
        save_bot_learning(data)
    except Exception:
        pass
