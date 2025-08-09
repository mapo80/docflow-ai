# FastAPI All-in-One — Template-Guided RAG (Markdown-first)

This service extracts **structured fields** from documents with **LLM always on**, optional **RAG per-field** when the Markdown would overflow the model's context, and a **forensic report** per request.

## Quick start
```bash
pip install -r requirements.txt
export API_KEY=dev
export MOCK_LLM=1
uvicorn main:app --host 0.0.0.0 --port 8080
```

## Endpoints
- `POST /extract` — sync extraction (multipart: `file`, `template` JSON).
- `POST /jobs` — async job.
- `GET /jobs/{id}` — job result.
- `GET /jobs/{id}/events` — SSE (backlog replay).
- `GET /reports/{request_id}` — forensic JSON.
- `GET /reports/{request_id}/bundle.zip` — full bundle (JSON + Markdown + artifacts).
- `GET /metrics` — Prometheus.

## Template JSON
```json
{
  "name": "fattura_it_v1",
  "fields": ["numero","data","cf","totale","iban"],
  "llm_text": "Istruzioni operative su dove e come cercare i campi..."
}
```

## Architecture (ASCII)
```
Client --> FastAPI (/extract|/jobs|/reports)
            |
            v
       Orchestrator
       - Markdown (PyMuPDF)
       - PP-Structure policy (auto/always/never/auto_pages)
       - Single-Pass or RAG per-field (BM25+FAISS)
       - LLM (llama.cpp, JSON strict)
       - Token alignment (multi-bbox)
       - Forensic report
```

## Environment variables (detailed)
(See the conversation message above; all variables are documented with defaults.)

## Tests
Run `pytest -q` with `MOCK_LLM=1` to exercise endpoints deterministically.
