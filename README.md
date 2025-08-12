# DocFlow AI

> **Intelligent, policy-driven document processing pipeline with FastAPI and LLM integration**

DocFlow AI is a modular FastAPI backend for automated document ingestion, OCR/structure parsing, markdown conversion, and AI-driven field extraction with confidence scores. It supports multi-page documents, optional overlay rendering for bounding boxes, policy-based parsing strategies, and a comprehensive testing setup with coverage.

This README consolidates what is implemented in the repository and expands it with precise technical/architectural detail, operational guidance, and environment-variable documentation.

> **Repository:** `mapo80/docflow-ai`  
> **Language:** Python  
> **Primary runtime:** FastAPI (ASGI)  
> **Testing:** `pytest` + coverage  
> **Status:** Developer-focused backend service for programmatic document processing

---

## 0) TL;DR

- **POST `/process-document`** accepts a **PDF or image**, applies **DocTR OCR** under a **policy**, converts content to markdown via **MarkItDown**, then asks an **LLM** to extract a **JSON of fields with confidence**.
- **Overlays** can be produced to show bounding boxes for recognized fields.  
- **Mocks** allow **offline** development without external services.  
- **Tests** target **unit and integration** behavior, with coverage reporting.

---

## 1) Use Cases

- **ID / invoice / form processing** with structured JSON output + bbox overlays.  
- **Policy-driven OCR**: enforce OCR always/never/auto depending on source type and quality.  
- **LLM enrichment**: get normalized fields and per-field confidence.  
- **Bulk/async processing**: batch documents and collect metrics.  
- **Deterministic testing**: toggle mocks to stabilize outputs and CI runs.

---

## 2) Architecture Overview

### 2.1 High-Level Diagram

```
Client ──► FastAPI App (ASGI)
            │
            ▼
        Pipeline Controller
            │
    ┌───────┴─────────────┐
    │ Parse/OCR (DocTR) │
    │ Markdown (MarkItDown)│
    │ LLM Enrichment       │
    └───────┬─────────────┘
            ▼
    JSON (fields+confidence) + Optional Overlays
```

### 2.2 Detailed Processing Flow (Sequence)

```
+-------------------+
|  POST /process-   |
|  document (file)  |
+---------+---------+
          |
          v
  [1] Detect media type: PDF (digital/scanned) or Image
          |
          v
  [2] Apply OCR_POLICY: always | never | auto
          |      |         |
          |      |         +--> auto: heuristics decide OCR
          |      +------------> never: skip OCR entirely
          +-------------------> always: force OCR
          |
          v
  [3] Preprocess (split pages, rasterize PDF if needed)
          |
          v
  [4] Parsers: OCR tokens via DocTR (tables unsupported)
          |
          v
  [5] Convert to Markdown using MarkItDown
          |
          v
  [6] LLM extraction → JSON { field: { value, confidence } }
          |
          v
  [7] Overlays (optional): compute bounding boxes for fields
          |
          v
  [8] Response: JSON (+ overlays[] if enabled)
```

### 2.3 Module/Dependency Graph (by major files)

```
main.py (FastAPI app)
 ├─ config.py (env, settings)
 ├─ logger.py (structured logging)
 ├─ parse.py (policy application, type detection, routing)
 ├─ overlay.py (overlay computation/format)
 ├─ llm.py (LLM calls + JSON schema handling/mocking)
 ├─ metrics.py (timings, counters)
 ├─ reports.py (bundle/report formation)
 ├─ indexer.py / retriever.py (optional indexing/RAG hooks)
 ├─ align.py (page/image alignment utilities)
 ├─ jobs.py (batch/async job orchestration hooks)
 └─ clients/
    ├─ doctr_client.py (OCR client, mockable)
     ├─ markitdown_client.py (markdown conversion client)
     └─ llm.py (thin client wrapper or shared LLM helpers)
```

> Notes:
> - The `tests/` directory contains unit/integration tests that validate end-to-end flow, overlays, multi-page handling, policies, and error/reporting behaviors.
> - The LLM and OCR layers are **mockable** to support offline, deterministic test runs.

---

## 3) API

### 3.1 `POST /process-document`

Process a single document (PDF or image) and return structured fields and (optionally) overlays.

**Form-data parameters**

| Name            | Type  | Required | Default | Description                                                                 |
|-----------------|-------|----------|---------|-----------------------------------------------------------------------------|
| `file`          | File  | Yes      | —       | PDF or image (`.pdf`, `.png`, `.jpg` by default).                           |
| `ocr_policy`     | str   | No       | `auto`  | One of: `always`, `never`, `auto`. Controls OCR usage. |
| `llm_model`     | str   | No       | —       | Logical model ID/name resolved by the LLM client.                           |
| `overlays`      | bool  | No       | `false` | If `true`, include `overlays[]` with bounding boxes for recognized fields. |

**Example request**

```bash
curl -X POST "http://localhost:8000/process-document" \
  -F "file=@invoice.pdf" \
  -F "ocr_policy=auto" \
  -F "overlays=true"
```

**Example response**

```json
{
  "fields": {
    "invoice_number": {
      "value": "INV-2025-001",
      "confidence": 0.94
    }
  },
  "overlays": [
    { "field": "invoice_number", "bbox": [100, 50, 200, 80] }
  ]
}
```

> **BBox convention.** Bounding boxes are expressed in **pixel coordinates** with origin at the **top-left** of the source page image. The tuple represents **[x, y, width, height]** unless otherwise indicated by an explicit `mode` field in overlay metadata.

### 3.2 (Typical) Supporting Endpoints

- `GET /` or `GET /healthz` — **Health probe** (depending on `main.py` implementation).
- `GET /metrics` — If exposed, returns basic counters/timers (otherwise available via logs).

> If you want hardened OpenAPI docs at runtime, run with `uvicorn main:app --reload` and open `/docs` (Swagger) or `/redoc`.

---

## 4) Environment Variables (All knobs, types & effects)

**Important:** The list below reflects the variables **explicitly surfaced in the repository README and project structure**. If you add more knobs in `config.py`, keep them documented here.

| Variable              | Type  | Default                  | Allowed values / Format                    | Effect |
|-----------------------|-------|--------------------------|-------------------------------------------|--------|
| `MOCK_LLM`            | int   | `0`                      | `0` or `1`                                | If `1`, the LLM layer returns **mocked** JSON for deterministic tests and offline runs. |
| `MOCK_OCR`            | int   | `0`                      | `0` or `1`                                | If `1`, the OCR layer returns **mocked** tokens/bboxes (no external deps). |
| `OCR_POLICY`          | str   | `auto`                   | `always`, `never`, `auto`                 | Governs whether to call OCR or skip it. `auto` uses heuristics/type detection. |
| `MAX_TOKENS`          | int   | `1024`                   | Positive integer                           | Upper bound for tokens produced/consumed by LLM calls. Used to avoid runaway responses. |
| `ALLOWED_EXTENSIONS`  | str   | `.pdf,.png,.jpg`         | Comma-separated list                       | Restricts uploadable file types at request validation. |
| `LOG_LEVEL`           | str   | `INFO`                   | `DEBUG`, `INFO`, `WARNING`, `ERROR`        | (If supported by `logger.py`): controls logging verbosity. |
| `HOST`                | str   | `0.0.0.0`                | IPv4/IPv6 literal                          | (If used): bind address for the ASGI server. |
| `PORT`                | int   | `8000`                   | `1..65535`                                 | (If used): HTTP port for the ASGI server. |
| `CORS_ALLOW_ORIGINS`  | str   | `*`                      | CSV of origins or `*`                      | (If enabled): CORS control for browser clients. |
| `TMP_DIR`             | str   | system temp              | filesystem path                            | (If used): Working directory for page images/intermediates. |
| `KEEP_INTERMEDIATES`  | int   | `0`                      | `0` or `1`                                | (If used): Keep preprocessed page images to aid debugging. |
| `LLM_MODEL`           | str   | implementation-dependent | logical model name/id                      | Default model to use when `llm_model` not provided per request. |

> **Source-of-truth:** `config.py` is expected to parse/validate these. The repo’s public README enumerates the first five (`MOCK_LLM`, `MOCK_OCR`, `OCR_POLICY`, `MAX_TOKENS`, `ALLOWED_EXTENSIONS`). The remaining knobs are standard operational settings commonly wired via `config.py`/`logger.py`; enable them as needed and keep this table updated.

### 4.1 OCR_POLICY Semantics

```
+---------+---------------------------------------------------------------+
| Policy  | Behavior                                                      |
+---------+---------------------------------------------------------------+
| always  | Force OCR even for digital-native PDFs.          |
| never   | Skip OCR entirely; rely on digital text.         |
| auto    | Heuristics: classify input; OCR only if needed.              |
+---------+---------------------------------------------------------------+
```

### 4.2 Example `.env`

```
MOCK_LLM=1
MOCK_OCR=1
OCR_POLICY=auto
MAX_TOKENS=1024
ALLOWED_EXTENSIONS=.pdf,.png,.jpg
LOG_LEVEL=DEBUG
```

---

## 5) Repository Layout

> Reflects the current top-level tree observed in the repo.

```
docflow-ai/
├── clients/
│   ├── llm.py
│   ├── markitdown_client.py
│   └── doctr_client.py
├── tests/
│   ├── test_overlays_and_bundle.py
│   ├── test_overlays_multi_page.py
│   ├── test_pipeline_ext.py
│   ├── test_pipeline_images_pdf.py
│   └── test_policy_errors_reports.py
├── .gitignore
├── README.md
├── align.py
├── config.py
├── indexer.py
├── jobs.py
├── llm.py
├── logger.py
├── main.py
├── metrics.py
├── overlay.py
├── parse.py
├── reports.py
├── requirements.txt
└── retriever.py
```

> Some earlier diagrams may refer to `core/` and `fastapi_all_in_one_proj/`—this README adapts to the **current** layout. The functional split is the same: **clients** (external integrations), **app** (main/pipeline), **tests**.

---

## 6) Component Details

### 6.1 `main.py` — FastAPI application

- Defines the ASGI app, routes (notably `POST /process-document`), and dependency wiring.  
- Validates incoming uploads (extension/MIME by `ALLOWED_EXTENSIONS`).  
- Binds **DocTR**, **MarkItDown**, and **LLM** services via **clients**.
- Delegates orchestration to the **pipeline** implemented across `parse.py`, `overlay.py`, and helpers.

**Operational hooks**  
- Health endpoint (`/` or `/healthz`).  
- Swagger/OpenAPI at `/docs` and `/redoc`.  
- Uvicorn recommended for local dev: `uvicorn main:app --reload --port 8000`.

### 6.2 `config.py` — Settings

- Reads environment variables, applies defaults, and performs basic validation/casting.  
- Emits resolved configuration to logs at startup (respecting `LOG_LEVEL`).  
- Should keep all **env var defaults** centralized to ensure reproducibility.

### 6.3 `logger.py` — Logging

- Uniform logger setup for modules.  
- Suggested format: timestamp, level, module, request ID (if any), message.  
- Levels governed by `LOG_LEVEL` env var.

### 6.4 `parse.py` — Policy, detection & parsing

- Applies **OCR_POLICY** to decide if/when to use OCR.
- Detects document type (image vs PDF; digital vs scanned where possible).  
- Consolidates page text, layout tokens, and tables.  
- Converts unified content to **Markdown** via MarkItDown client.  
- Feeds the normalized text (and optional structured hints) to the **LLM** to obtain **JSON fields + confidence**.

### 6.5 `overlay.py` — Overlays

- Computes **bbox overlays** for fields recognized by the LLM (via mapping heuristics/anchors from OCR tokens).  
- Supports multi-page input, with page indices in overlay metadata.  
- Outputs **XYWH** pixel coordinates in source space; can be adapted for `xyxy` if needed.  
- Can render visual aids server-side or return coordinates for client-side rendering.

### 6.6 `llm.py` & `clients/llm.py` — LLM integration

- Standard interface: `chat_json_async(prompt, schema, max_tokens, model)` returning a validated JSON.  
- **Mock mode** (`MOCK_LLM=1`) injects fixed JSON to stabilize tests.  
- Model selection: either **per-request** (`llm_model`) or from **`LLM_MODEL`** env default.

### 6.7 `clients/doctr_client.py` — OCR

- Provides `analyze_async(image_or_pdf_page)` → tokens/blocks with coordinates.
- **Mock mode** (`MOCK_OCR=1`) injects synthetic tokens/bboxes for deterministic runs.
- Only invoked when OCR_POLICY is `always` or `auto` (and heuristics decide yes).

### 6.8 `clients/markitdown_client.py` — Markdown conversion

- Wraps **MarkItDown** to transform PDF/image-derived text into **normalized Markdown**.  
- Helps the LLM by providing a clean, structured, low-noise textual representation.

### 6.9 `metrics.py` — Telemetry

- Helpers for timing sections of the pipeline and counting outcomes.  
- May expose counters via logs and/or an endpoint if wired.

### 6.10 `reports.py` — Bundles and summaries

- Builds a **single response bundle** consolidating fields, confidence, overlays, and page-level metadata.  
- Provides error summaries and per-step diagnostics as needed for testing and support.

### 6.11 `align.py` — Alignment utilities

- Page/image alignment helpers to compensate for rotation/skew.  
- Critical for precise overlay placement when dealing with scans/photos.

### 6.12 `indexer.py` / `retriever.py` — Index/RAG hooks (optional)

- Components to index extracted content (e.g., vector store, keyword index).  
- Retrieval helpers to provide **document-context** to future LLM calls.

### 6.13 `jobs.py` — Batch/Async hooks (optional)

- Hooks/utilities to schedule batch processing, background workers, or queues.  
- Useful for large volumes or S3-like ingestion pipelines.

---

## 7) Data Contracts

### 7.1 Response schema

```json
{
  "fields": {
    "<name>": { "value": "<string|number|date|...>", "confidence": 0.0 }
  },
  "overlays": [
    {
      "field": "<name>",
      "bbox": [x, y, width, height],
      "page_index": 0
    }
  ],
  "meta": {
    "pages": 1,
    "ocr_policy": "auto",
    "timings_ms": { "ocr": 0, "llm": 0, "total": 0 }
  }
}
```

### 7.2 Error schema

```json
{
  "error": {
    "code": "UNSUPPORTED_MEDIA_TYPE",
    "message": "Only .pdf,.png,.jpg are allowed"
  }
}
```

**Common error codes**: `BAD_REQUEST`, `UNSUPPORTED_MEDIA_TYPE`, `INTERNAL_ERROR`, `TIMEOUT`, `POLICY_ERROR`.

---

## 8) Running Locally

### 8.1 Prerequisites

- Python 3.10+ (recommended)
- `pip` or `uv`/`pipx`
- OS packages for image/PDF handling (Ghostscript/poppler may be needed depending on MarkItDown setup)

### 8.2 Install

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** the Markdown converter dependency `markitdown` is pinned to version `0.1.2` for compatibility.

### 8.3 Run (development)

```bash
export LOG_LEVEL=DEBUG
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Open:** `http://localhost:8000/docs`

When dependencies are installed (DocTR models and the GGUF embedder), the startup logs include lines like:

```
INFO clients.doctr_client Creating DocTRClient instance
INFO clients.embeddings_local Initializing local embedder from /models/embeddings.gguf
INFO main Warmup finished: DocTR and GGUF embedder loaded
```

### 8.4 Run with mocks (offline)

```bash
export DOCFLOW_DATA_DIR="./data"
export MOCK_LLM=1
export MOCK_OCR=1
uvicorn main:app --reload
```

---

## 9) Testing & Coverage

### 9.1 Unit/Integration Tests

- Validates: policy handling, OCR enablement decisions, overlay correctness, multi-page flows, and error/reporting pathways.

```bash
pytest
```

### 9.2 Coverage

```bash
pytest --cov=. --cov-report=term-missing --cov-report=html
# HTML report: ./htmlcov/index.html
```

**Stabilizing tests**

```bash
MOCK_LLM=1 MOCK_OCR=1 pytest
```

---

## 10) Security & Hardening

- **Input validation**: extensions and MIME types constrained by `ALLOWED_EXTENSIONS`.  
- **Sandboxing**: process files in a temporary directory; **clean up** after run.  
- **LLM output validation**: ensure response is valid JSON before use.  
- **Overlay gating**: render only for recognized, validated fields.  
- **Logging discipline**: avoid logging raw PII; redact sensitive values in debug logs.  
- **CORS**: restrict origins via `CORS_ALLOW_ORIGINS` in production.  
- **Rate limits / auth**: front with an API gateway or FastAPI dependencies when internet-exposed.

**Suggested prod add-ons**

- Reverse proxy (nginx/traefik) TLS termination.  
- WAF rules for uploads (size/type).  
- Virus scan on upload (e.g., ClamAV) before parsing.  
- S3/GCS object storage with signed URLs for large files.  
- Background workers (jobs.py) for long-running tasks.

---

## 11) Observability

- **Logs** (`logger.py`): include request IDs/correlation IDs.  
- **Timings**: capture `ocr_ms`, `llm_ms`, `total_ms` in response `meta.timings_ms` and/or logs.  
- **Metrics**: wire `/metrics` or push to Prometheus via a sidecar/sd-agent if needed.  
- **Tracing**: optional OpenTelemetry integration (FastAPI instrumentation) for latency profiling.

---

## 12) Performance Considerations

- Prefer **policy=auto** to avoid OCR on digital-native PDFs.  
- Parallelize **per-page OCR** for multi-page scans (limit concurrency to CPU cores).  
- Cache **MarkItDown** outputs for identical inputs (hash-based).  
- Choose **compact LLMs** for low-latency JSON extraction; enforce `MAX_TOKENS`.  
- Use **image alignment** (align.py) to improve overlay accuracy for photographed documents.

---

## 13) Extensibility

### 13.1 Add a new field extractor

1. Extend the prompt/schema in `llm.py`.  
2. Add post-processing/validation for the new field.  
3. Update overlay mapping if the field should be localized on page.  
4. Add tests to cover typical and edge cases.

### 13.2 Add a new parser

- Create a `clients/<your_parser>_client.py` with a standard `analyze_*` method.  
- Integrate it into `parse.py` under the selected policy.  
- Ensure outputs use consistent token/table structures.  
- Add mocks and tests.

### 13.3 Change bbox convention

- Update `overlay.py` to emit `xyxy` or normalized coordinates; document in the response `meta` or an `overlay.mode` field.

---

## 14) End-to-End ASCII Example

```
          +------------------------+
Upload -> |  FastAPI /process-doc  | -> Validate ext/MIME
          +-----------+------------+
                      |
                      v
              +-------+--------+
              | Apply OCR_POLICY|  (always/never/auto)
              +-------+--------+
                      |
          +-----------+------------+
          |  DocTR OCR  |  [skip if never]
          +-----------+------------+
                      |
          +-----------+------------+
          |  MarkItDown (Markdown) |
          +-----------+------------+
                      |
          +-----------+------------+
          | LLM (JSON + confidence)|  [MOCK_LLM=1 -> deterministic JSON]
          +-----------+------------+
                      |
          +-----------+------------+
          | Overlays (bbox XYWH)   |  [overlays=true]
          +-----------+------------+
                      |
                      v
          +-----------+------------+
          |   JSON response        |
          +------------------------+
```

---

## 15) Requirements

All dependencies are pinned in `requirements.txt`. Typical stacks include:

- **FastAPI / Uvicorn** (web app & ASGI server)  
- **Pydantic** (validation)  
- **Pillow / PDF tooling** (image/PDF IO)  
- **MarkItDown** (markdown conversion)  
- **pytest / coverage** (tests)  

> Install with `pip install -r requirements.txt`.

---

## 16) FAQ

**Q: Do I need OCR for all PDFs?**
A: No. Use `OCR_POLICY=auto` so digital PDFs skip OCR.

**Q: Can I run completely offline?**
A: Yes. Set `MOCK_LLM=1` and `MOCK_OCR=1`. You’ll get deterministic results for tests and demos.

**Q: How do I add authentication?**  
A: Use FastAPI dependencies or a proxy (e.g., API Key/Token via a header). Keep PII out of logs.

**Q: Where do overlay coordinates come from?**  
A: They are inferred by mapping recognized fields back to OCR token positions and table structures.

---

## 17) License

MIT License (see repository).

---

## 18) Maintainers

- `mapo80` (GitHub)

---

## 19) Change Log (high-level)

- Initial release with FastAPI service, DocTR integration, MarkItDown conversion, LLM JSON extraction, overlays, and tests with coverage.
- Added mock switches (`MOCK_LLM`, `MOCK_OCR`) and policy control (`OCR_POLICY`).
- Hardened tests and added coverage reports (HTML + terminal).

---

## 20) Swagger/OpenAPI Docs

- The FastAPI app serves interactive Swagger docs at `/docs` and the OpenAPI spec at `/openapi.json`.
- Toggle these routes with the `DOCS_ENABLED` environment variable (`1` = enabled, `0` = disabled).

### Embedding senza FastAPI

È possibile utilizzare il motore di embedding GGUF direttamente, senza avviare il server FastAPI:

```bash
export HUGGINGFACE_TOKEN=<token>
python - <<'PY'
from clients.embeddings_local import embed_texts
vec = embed_texts(['hello'])[0]
print(len(vec), vec[:5])
PY
```

Output d'esempio:

```text
nomic-embed-text-v1.5.Q4_K_M.gguf: 100% 84.1M/84.1M [00:03<00:00, 23.4MB/s]
768 [0.049985986202955246, -0.07129103690385818, -4.728538990020752, -0.15377487242221832, 0.4639637768268585]
```

Il primo numero indica la dimensione del vettore (768) seguito dai primi valori dell'embedding.

### PPStructure Light senza FastAPI

Anche l'analizzatore OCR può essere eseguito direttamente:

```bash
python - <<'PY'
import asyncio
from clients.doctr_client import analyze_async
with open("dataset/sample_invoice.png","rb") as f:
    data = f.read()
pages = asyncio.run(analyze_async(data, "sample_invoice.png"))
print(len(pages), pages[0].get("blocks"))
PY
```

Output d'esempio:

```text
1 []
```

La prima cifra indica il numero di pagine elaborate; il secondo valore mostra i blocchi individuati nella prima pagina.

## 📂 Dataset

Il progetto può utilizzare un **dataset di documenti di esempio** per testare e validare l’estrazione di campi strutturati e delle relative bounding box.

### Struttura del dataset di esempio

| File                   | Descrizione |
|------------------------|-------------|
| `sample_invoice.pdf`   | Fattura in formato PDF (A4) con testo editabile. |
| `sample_invoice.png`   | Stessa fattura rasterizzata in PNG (A4 a ~150 DPI) per simulare uno scan. |

### Contenuto dei documenti
I documenti contengono:
- **Intestazione** con:
  - Nome azienda (`ACME S.p.A.`)
  - Tipo documento (`Fattura / Invoice`)
  - Numero fattura (`INV-2025-001`)
  - Data fattura (`2025-08-09`)
- **Tabella prodotti** (solo a scopo di posizionamento e test delle bounding box)
- Messaggio di cortesia a piè di pagina.

### Campi da estrarre
Il sistema è configurato per estrarre i seguenti campi:

| Nome campo        | Tipo | Esempio |
|-------------------|------|---------|
| `company_name`    | string | `"ACME S.p.A."` |
| `document_type`   | string | `"Fattura / Invoice"` |
| `invoice_number`  | string | `"INV-2025-001"` |
| `invoice_date`    | string (YYYY-MM-DD) | `"2025-08-09"` |

### Bounding Box e `locations[]`
Oltre ai valori estratti, per ogni campo il sistema può restituire:
- `locations[]` → array di oggetti con `bbox` (`[x,y,width,height]`) e `page_index` (0-based)
- Alias `bbox` e `page_index` → corrispondenti alla **prima** location trovata.

Queste coordinate sono ottenute tramite **PP-Structure Light** con **Table Cell Detection** attivato, in modo da:
- riconoscere la posizione esatta dei valori
- supportare sia testo in-linea che celle di tabelle
- gestire documenti PDF rasterizzati e immagini.