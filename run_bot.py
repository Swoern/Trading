"""
run_bot.py — TradeAI automatische paper trading daemon.

Start dit script in een apart terminalvenster:
    python run_bot.py

De bot:
  - Controleert elke 15 minuten de markten op setups
  - Sluit automatisch trades als SL of TP bereikt is
  - Genereert elke ochtend om 9:00 Nederlandse tijd een dagrapport
  - Leert van gesloten trades en past zijn strategie aan

⚠️  UITSLUITEND nep-geld. Geen echte orders. Geen echt risico.
"""
import os
import sys
import time
import logging
import atexit
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

from src.database import init_database
from src.auto_trader import run_cycle, genereer_dagelijkse_samenvatting
from src.notifier import melding_dagrapport, melding_bot_gestart, melding_bot_gestopt, start_polling, stuur_melding

AMSTERDAM        = ZoneInfo("Europe/Amsterdam")
CHECK_INTERVAL   = 900     # 15 minuten — past bij 15m candles
SAMENVATTING_UUR = 9       # 09:00 Amsterdam tijd
LOCK_FILE        = Path(__file__).parent / "bot.lock"
_LOCK_FH         = None


def _check_lock() -> None:
    """
    Gebruik een OS-level bestandslock (fcntl) zodat er nooit twee instanties
    tegelijk draaien — ook niet als psutil ontbreekt of het lock-bestand achterbleef.
    De lock wordt automatisch vrijgegeven als het process sterft.
    """
    import fcntl
    global _LOCK_FH
    lock_fh = open(LOCK_FILE, "a+")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        try:
            lock_fh.seek(0)
            pid = lock_fh.read().strip()
        except Exception:
            pid = "onbekend"
        print(f"Bot draait al (PID {pid}). Stop de andere instantie eerst.")
        sys.exit(1)
    lock_fh.seek(0)
    lock_fh.truncate()
    lock_fh.write(str(os.getpid()))
    lock_fh.flush()
    os.fsync(lock_fh.fileno())
    # Houd expliciet een globale reference vast zodat de lock actief blijft.
    _LOCK_FH = lock_fh
    atexit.register(_LOCK_FH.close)

class _NLFormatter(logging.Formatter):
    """Logging formatter die UTC omzet naar Nederlandse tijd."""
    def formatTime(self, record, datefmt=None):
        ct = datetime.fromtimestamp(record.created, tz=AMSTERDAM)
        return ct.strftime(datefmt or "%Y-%m-%d %H:%M:%S")

_handler_console = logging.StreamHandler(sys.stdout)
# Roteer logs zodat bot.log niet ongebonden groeit op een 24/7-server (Pi).
_handler_file    = RotatingFileHandler(
    "bot.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_fmt = _NLFormatter("%(asctime)s [%(levelname)s] %(message)s")
_handler_console.setFormatter(_fmt)
_handler_file.setFormatter(_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_handler_console, _handler_file])
LOG = logging.getLogger("run_bot")


def main():
    _check_lock()
    init_database()
    LOG.info(f"TradeAI Bot gestart (PID {os.getpid()}). Druk op Ctrl+C om te stoppen.")
    LOG.info(f"Check-interval: {CHECK_INTERVAL // 60} minuten | Dagrapport: {SAMENVATTING_UUR:02d}:00 Amsterdam")
    melding_bot_gestart()
    start_polling()

    samenvatting_gegeven_op = None
    fouten_op_rij = 0

    while True:
        nu = datetime.now(AMSTERDAM)

        # Dagelijkse samenvatting om 9:00
        vandaag = nu.date()
        if (nu.hour == SAMENVATTING_UUR
                and nu.minute < (CHECK_INTERVAL // 60)
                and samenvatting_gegeven_op != vandaag):
            try:
                LOG.info("Dagelijkse samenvatting genereren...")
                samenvatting = genereer_dagelijkse_samenvatting()
                LOG.info("\n" + samenvatting)
                melding_dagrapport(samenvatting)
                samenvatting_gegeven_op = vandaag
            except Exception as e:
                LOG.error(f"Samenvatting mislukt: {e}")

        # Handels-cyclus
        try:
            run_cycle()
            fouten_op_rij = 0
        except Exception as e:
            fouten_op_rij += 1
            LOG.error(f"Fout in bot-cyclus ({fouten_op_rij}x op rij): {e}", exc_info=True)
            # Stuur Telegram-melding bij aanhoudende fouten
            if fouten_op_rij == 3:
                stuur_melding(
                    f"⚠️ <b>TradeAI bot fout</b>\n"
                    f"3 opeenvolgende fouten in de cyclus:\n"
                    f"<code>{str(e)[:300]}</code>\n"
                    f"Bot probeert verder te draaien."
                )

        LOG.info(f"Volgende check om {(nu + timedelta(seconds=CHECK_INTERVAL)).strftime('%H:%M')}")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.info("Bot gestopt door gebruiker.")
        melding_bot_gestopt()
