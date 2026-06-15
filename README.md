# rag-pipeline

## Dev Instructions
### Lokales Setup

1. Alle Pakete im Workspace installieren:

```bash
uv sync --all-packages
```

2. Optional: Dependencies nach Aenderungen im Lockfile aktualisieren:

```bash
uv lock
uv sync --all-packages
```

### Services lokal starten

API:

```bash
uv run rag-api
```

Ingestion:

```bash
uv run ingestion
```

### Docker: Images bauen

API:

```bash
docker build -f services/rag_api/Dockerfile -t rag-api:dev .
```

Ingestion:

```bash
docker build -f services/ingestion/Dockerfile -t ingestion:dev .
```

### Docker: Container starten

Modell-Variablen (`EMBEDDING_MODEL_*`, `LLM_MODEL_*`) werden per `--env-file .env` geladen.
Nur service-spezifische Werte muessen zusaetzlich per `-e` uebergeben werden.

Datenbank-Container einmalig starten (vor API oder Ingestion):

```bash
docker compose up -d qdrant mongodb
```

API:

```bash
docker run --rm \
  --env-file .env \
  --network rag-pipeline-net \
  -e MONGO_URI=mongodb://mongodb:27017 \
  -e QDRANT_URL=http://qdrant:6333 \
  -p 8000:8000 \
  rag-api:dev
```

Ingestion (Pfade zu den Daten als Volumes einbinden):

```bash
docker run --rm \
  --env-file .env \
  --network rag-pipeline-net \
  -v "$(pwd)/publications:/data/publications:ro" \
  -v "$(pwd)/full-md-docs:/data/full-md-docs:ro" \
  -v "$(pwd)/ingestion/logs:/data/code:ro" \
  -e MONGO_URI=mongodb://mongodb:27017 \
  -e QDRANT_URL=http://qdrant:6333 \
  -e INGEST_PDF_BASE_PATH=/data/publications \
  -e INGEST_MARKDOWN_BASE_PATH=/data/full-md-docs \
  -e INGEST_CODE_BASE_PATH=/data/code \
  ingestion:dev
```

### Lokal: Ingestion ausfuehren

```bash
set -a && source .env && set +a
INGEST_PDF_BASE_PATH="/pfad/zu/publications" \
INGEST_MARKDOWN_BASE_PATH="/pfad/zu/full-md-docs" \
INGEST_CODE_BASE_PATH="/pfad/zu/code" \
uv run ingestion
```