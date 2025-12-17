## Docker

### Build & run (backend + frontend)

1. Create a GCP service account json file at:
   - `contract-pdf-qna/gcp-service-account.json`

2. Set required env vars in your shell (or create a `.env` for docker-compose):
   - `OPENAI_API_KEY`
   - `MONGO_URI`
   - `MILVUS_HOST`

3. Run:

```bash
docker compose up --build
```

- **Backend**: `http://localhost:8001`
- **Frontend**: `http://localhost:8080`

### Build images individually

```bash
# backend
docker build -t contract-pdf-qna-backend ./contract-pdf-qna

# frontend
docker build -t contract-pdf-qna-frontend ./contract-pdf-qna-frontend
```

