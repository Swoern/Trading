@echo off
title TradeAI Coach — lokale server
cd /d "%~dp0"

echo TradeAI Coach wordt gestart...
echo Sluit dit venster NIET — de app draait hierin.
echo Open je browser op: http://localhost:8501
echo.

python -m streamlit run app.py --server.port 8501 --server.headless true

pause
