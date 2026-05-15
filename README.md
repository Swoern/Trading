# 📊 TradeAI Coach — Leer traden zonder echt geld

> ⚠️ **WAARSCHUWING**: Dit systeem geeft GEEN financieel advies.
> Resultaten uit het verleden geven geen garantie voor de toekomst.
> Paper trading is uitsluitend bedoeld voor leren en oefenen.
> **Automatisch handelen met echt geld is niet mogelijk in deze versie.**

---

## Wat doet dit systeem?

TradeAI Coach is een leer-tool waarmee je:
- 📈 Marktdata analyseert (trend, volatiliteit, support/resistance, volume)
- 🔄 Strategieën backtestet op historische OHLCV-data
- 📝 Paper trades bijhoudt en beheert (nep-trades zonder echt geld)
- 📚 Leert van fouten via automatische evaluatie en suggesties

**Geen echte orders. Geen API-keys voor echte trading. Alleen leren.**

---

## Installatie

### 1. Vereisten
- Python 3.10 of hoger
- pip (pakketbeheerder, komt standaard met Python)

### 2. Installeer de pakketten

Open een terminal in de projectmap en voer uit:

```bash
pip install -r requirements.txt
```

### 3. Start het dashboard

```bash
streamlit run app.py
```

Het dashboard opent automatisch in je browser op `http://localhost:8501`.

---

## Projectstructuur

```
TRADEAI/
│
├── data/                   ← Historische CSV-bestanden
│   └── sample_btc.csv      ← Automatisch aangemaakt bij eerste start
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py      ← CSV laden en sample data genereren
│   ├── indicators.py       ← Technische indicatoren (SMA, ATR, RSI, ...)
│   ├── strategy.py         ← Strategie-regels en marktanalyse
│   ├── backtest.py         ← Backtesting op historische data
│   ├── paper_trading.py    ← Paper trades openen en sluiten
│   ├── database.py         ← SQLite opslag
│   └── evaluator.py        ← Foutanalyse en verbetersugesties
│
├── app.py                  ← Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Hoe voeg ik eigen CSV-data toe?

### Stap 1: Maak of download een CSV-bestand

Je CSV-bestand moet de volgende kolommen bevatten:

| Kolom     | Beschrijving                        | Voorbeeld          |
|-----------|-------------------------------------|--------------------|
| timestamp | Datum (en optioneel tijdstip)       | 2024-01-15         |
| open      | Openingsprijs van de candle         | 42000.50           |
| high      | Hoogste prijs van de candle         | 43100.00           |
| low       | Laagste prijs van de candle         | 41800.00           |
| close     | Sluitingsprijs van de candle        | 42850.75           |
| volume    | Verhandeld volume in die periode    | 1234.56            |

**Voorbeeld CSV-inhoud:**
```csv
timestamp,open,high,low,close,volume
2024-01-01,42000,43100,41800,42850,1500
2024-01-02,42850,44000,42500,43750,2100
2024-01-03,43750,44200,43000,43200,1800
```

### Stap 2: Zet het bestand in de `data/` map

Kopieer je CSV-bestand naar:
```
TRADEAI/data/jouw_bestand.csv
```

### Stap 3: Selecteer in het dashboard

In het dashboard (linkerzijbalk) verschijnt je bestand automatisch in de keuzelijst.

### Gratis data bronnen (voor later)
- **Crypto**: [CryptoDataDownload](https://www.cryptodatadownload.com/) — gratis historische OHLCV-data
- **Aandelen**: Yahoo Finance via `yfinance` (versie 2 uitbreiding)
- **Forex**: [HistData.com](https://histdata.com/) — gratis historische forex-data

---

## Hoe werkt de strategie?

De standaard strategie gebruikt deze regels:

| Regel | Beschrijving |
|-------|-------------|
| 1 | Alleen LONG kijken als er een duidelijke uptrend is (close > SMA20 > SMA50) |
| 2 | Geen trade midden in een range (positie 30–70% van support–resistance) |
| 3 | Volume moet normaal of hoog zijn |
| 4 | RSI mag niet boven de 75 zijn (niet overbought) |
| 5 | Risk/Reward minimaal 1:2 |
| 6 | Stop-Loss = 1.5 × ATR onder entry |
| 7 | Take-Profit = 2 × risico boven entry |

---

## Hoe gebruik ik paper trading?

1. Ga naar het **📝 Paper Trading** tabblad
2. Klik op **➕ Nieuwe trade**
3. Vul de entry, stop-loss en take-profit in
4. **Schrijf altijd op waarom je de trade opent** — dit is de kern van leren
5. Volg de trade en sluit hem zodra hij de SL of TP raakt
6. Vul in wat je leerde van de trade
7. Bekijk na een paar trades de **📚 Evaluatie**

---

## Hoe interpreteer ik de analyse?

| Term | Uitleg |
|------|--------|
| **SMA20** | Gemiddelde prijs van de laatste 20 candles. Korte termijn trend. |
| **SMA50** | Gemiddelde prijs van de laatste 50 candles. Lange termijn trend. |
| **ATR** | Average True Range. Hoe wild de markt beweegt. Gebruik voor SL. |
| **RSI** | Momentum indicator (0–100). >70 = overbought. <30 = oversold. |
| **Support** | Vloer: prijs waarbij kopers historisch instapten. |
| **Resistance** | Plafond: prijs waarbij verkopers historisch uitstapten. |

---

## Uitbreidingen voor versie 2

- [ ] Short-setups ondersteunen
- [ ] Live crypto-data via `ccxt` (Binance, Kraken) — *geen echte orders*
- [ ] Aandelensdata via `yfinance`
- [ ] Meerdere strategieën vergelijken in de backtest
- [ ] Positiegrootte calculator op basis van risico%
- [ ] Email/notificatie bij setup-signaal
- [ ] Exporteer trades naar Excel
- [ ] Meer indicatoren: MACD, Bollinger Bands, Stochastic
- [ ] Weekrapport automatisch genereren als PDF
- [ ] Multi-asset dashboard (meerdere grafieken naast elkaar)

---

## Veelgestelde vragen

**Kan ik echt geld verliezen?**
Nee. Dit systeem plaatst geen echte orders. Alle trades zijn nep (paper trading).

**Welke data moet ik gebruiken?**
Begin met de meegeleverde sample data. Voeg later je eigen CSV toe.

**Waarom werkt de backtest niet?**
Je hebt minimaal 55 candles nodig voor de indicatoren. Zorg dat je CSV lang genoeg is.

**Hoe reset ik alle trades?**
Verwijder het bestand `tradeai.db`. De database wordt opnieuw aangemaakt bij de volgende start.

---

*Gemaakt voor beginners die rustig willen leren traden, naast school.*
