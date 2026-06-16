# TradeAI Coach deployen

## Aanrader: Streamlit Community Cloud

1. Zet deze map in een GitHub repository.
2. Ga naar https://share.streamlit.io.
3. Klik op `Create app`.
4. Kies je repository en branch.
5. Vul als main file in: `app.py`.
6. Klik op `Deploy`.

Let op: `tradeai.db` wordt bewust niet mee gecommit. Op gratis hosting is lokale SQLite-opslag meestal niet bedoeld als permanente database.

## Render

Gebruik deze instellingen als je handmatig een Render Web Service maakt:

```text
Build command:
pip install -r requirements.txt

Start command:
streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true
```

De meegeleverde `render.yaml` kan ook als Render Blueprint gebruikt worden.

Voor blijvende paper-trades op Render heb je een persistent disk of externe database nodig. Zet dan bijvoorbeeld:

```text
TRADEAI_DB_PATH=/var/data/tradeai.db
```

## Fly.io of andere Docker-hosting

De meegeleverde `Dockerfile` start de app via:

```text
streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true
```

Voor persistente SQLite-opslag kun je een volume mounten en `TRADEAI_DB_PATH` naar dat volume laten wijzen.

## Secrets & beveiliging

Geheimen (Telegram-token, optionele API-keys) horen **niet** in de code. De bot leest ze in deze
volgorde (`src/secure_config.py`): eerst environment-variabelen, dan als fallback een lokaal
`telegram_config.json`. Dat bestand staat in `.gitignore` en wordt nooit gecommit.

**Aanbevolen (env-variabelen):**

```bash
export TRADEAI_TELEGRAM_TOKEN="123456:abc..."   # Telegram bot-token
export TRADEAI_TELEGRAM_CHAT_ID="8248439176"    # jouw chat-id
# optioneel, alleen als je deze diensten gebruikt:
export TRADEAI_COINGLASS_KEY="..."
export TRADEAI_CRYPTOPANIC_TOKEN="..."
```

Op de Pi zet je deze in de systemd-unit (`Environment=`-regels) of in een `EnvironmentFile=`.

**Als je tóch `telegram_config.json` gebruikt** (lokale fallback), beperk de leesrechten:

```bash
chmod 600 telegram_config.json
```

**Token roteren** (bv. als hij ooit gedeeld is): open Telegram → praat met `@BotFather` →
`/revoke` voor het oude token → `/token` voor een nieuwe → werk de env-variabele of
`telegram_config.json` bij en herstart de bot (`systemctl restart tradeai`).

## Markten instellen

De bot verhandelt standaard BTC/ETH/SOL + de majors XRP/DOGE/ADA. Pas dit aan met:

```bash
export TRADEAI_MARKETS="BTC-USD,ETH-USD,SOL-USD"   # komma-gescheiden; BTC blijft het anker
```

Nieuwe (niet-core) munten krijgen automatisch voorzichtige "satelliet"-parameters
(kleinere inzet, alleen handelen als de BTC-markt actief is).

## Optionele LLM-adviseslaag (Fase 3)

Een panel van AI-analisten (technisch/sentiment/risico + coördinator) dat ~1x per uur
per asset een **begrensd** advies geeft. Het kan de strengheid licht bijsturen maar
**nooit** een trade forceren of de discipline-regels overrulen. Standaard **uit**.

Aanzetten (kost een paar euro/maand via Anthropic Haiku):

```bash
pip install anthropic
export TRADEAI_LLM_ENABLED=1
export TRADEAI_ANTHROPIC_KEY="sk-ant-..."      # of ANTHROPIC_API_KEY
export TRADEAI_LLM_MODEL="claude-haiku-4-5-20251001"   # optioneel, dit is de default
```

Zonder de package, de flag of een key blijft de laag uit en draait de bot volledig op
de regel-agents. Het advies en de redenering zijn zichtbaar in het dashboard.
