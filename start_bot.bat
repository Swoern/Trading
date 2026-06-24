@echo off
title TradeAI Bot — automatisch paper trading
cd /d "%~dp0"

echo ================================================
echo  TradeAI Automatische Paper Trading Bot
echo ================================================
echo.
echo  Sluit dit venster NIET — de bot draait hierin.
echo  Alle activiteit wordt opgeslagen in bot.log
echo  Dagrapport verschijnt elke ochtend om 09:00
echo.
echo  Druk op Ctrl+C om de bot te stoppen.
echo ================================================
echo.

python run_bot.py

pause
