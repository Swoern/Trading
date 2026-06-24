#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

APP_DIR="$(pwd)"
PORT="${TRADEAI_PORT:-8501}"
LOG_FILE="$APP_DIR/streamlit.log"
PYTHON_BIN="${PYTHON_BIN:-python}"
HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python niet gevonden. Installeer python of zet PYTHON_BIN naar het juiste commando."
  exit 1
fi

if ! "$PYTHON_BIN" -m streamlit --version >/dev/null 2>&1; then
  echo "Streamlit niet gevonden voor $PYTHON_BIN. Probeer: $PYTHON_BIN -m pip install -r requirements.txt"
  exit 1
fi

echo "TradeAI dashboard herstarten"
echo "Map:  $APP_DIR"
echo "Port: $PORT"

pkill -f "streamlit run app.py" 2>/dev/null || true
pkill -f "python -m streamlit run app.py" 2>/dev/null || true
sleep 1

nohup "$PYTHON_BIN" -m streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port="$PORT" \
  --server.headless=true \
  > "$LOG_FILE" 2>&1 &

echo "Gestart met PID $!"
echo "Open: http://${HOST_IP:-localhost}:$PORT"
echo "Log:  $LOG_FILE"
