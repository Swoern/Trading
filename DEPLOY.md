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
