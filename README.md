# DocFlow AI
> *Intelligent, policy-driven document processing pipeline with FastAPI and LLM integration.*

DocFlow AI is a **modular FastAPI backend** for automated document analysis and enrichment.  
It processes PDFs, scanned images, and multi-page documents using **OCR, structured parsing, and AI-based field extraction**.

This implementation is **heavily tested** (70%+ coverage) and supports **mocked integrations** for development environments without external dependencies.

---

## 1. Features

- **FastAPI-based REST API**
- **OCR + Structured Parsing** via ppstructure
- **Markdown Conversion** with MarkItDown
- **LLM JSON Extraction** with confidence scores
- **Bounding Box Overlay Rendering**
- **Multi-page Document Support**
- **Policy-based Parsing Logic**
- **Full Unit & Integration Testing**
- **Configurable Mocking for Offline Development**
- **Coverage Reports in HTML & Terminal**

---

## 2. Architecture

### High-Level
```
Client ──► FastAPI App
           │
           ▼
       Pipeline Controller
           │
   ┌───────┴────────┐
   │ Parsers         │
   │ Overlays        │
   │ LLM Enrichment  │
   └───────┬────────┘
           ▼
       Response JSON + Optional Overlays
```

### Processing Flow
```
1. Upload Document
2. Detect type (PDF/Digital, PDF/Scanned, Image)
3. Apply policy (ppstructure always/never/auto)
4. Preprocessing (split pages, rasterize if needed)
5. Run parsers (OCR, table extraction)
6. Convert to markdown (MarkItDown)
7. LLM extraction to JSON (fields + confidence)
8. Overlay rendering for bounding boxes
9. Return result
```

---

## 3. API Endpoints

### `POST /process-document`
**Description:** Process a document and extract structured data.

**Parameters (form-data):**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| file | File | Yes | PDF or image |
| pp_policy | string | No | `always`, `never`, `auto` (default: `auto`) |
| llm_model | string | No | Model ID for extraction |
| overlays | bool | No | Return overlays in response |

**Example Request:**
```bash
curl -X POST "http://localhost:8000/process-document"      -F "file=@invoice.pdf"      -F "pp_policy=auto"      -F "overlays=true"
```

**Example Response:**
```json
{
  "fields": {
    "invoice_number": {
      "value": "INV-2025-001",
      "confidence": 0.94
    }
  },
  "overlays": [
    {"field": "invoice_number", "bbox": [100, 50, 200, 80]}
  ]
}
```

---

## 4. Configuration

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `MOCK_LLM` | Use mocked LLM JSON responses | `0` |
| `MOCK_PP` | Use mocked ppstructure output | `0` |
| `PP_POLICY` | Global ppstructure policy (`always`, `never`, `auto`) | `auto` |
| `MAX_TOKENS` | Max tokens for LLM calls | `1024` |
| `ALLOWED_EXTENSIONS` | Comma-separated list of extensions | `.pdf,.png,.jpg` |

---

## 5. Testing & Coverage

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=. --cov-report=term-missing --cov-report=html
```

Coverage HTML report will be in `htmlcov/`.

Mock integrations can be enabled for tests via env vars:
```bash
MOCK_LLM=1 MOCK_PP=1 pytest
```

---

## 6. Developer Notes

- **LLM Mocking**: Tests replace `llm.chat_json_async` with static JSON output to ensure deterministic runs.
- **OCR Mocking**: Tests replace `ppstructure_client.analyze_async` with fixed token/bbox sets.
- **Overlay Tests**: Verify bbox data matches expected mock structure.
- **Hardening**: Fragile assertions replaced with existence checks when data size varies.

---

## 7. Security Considerations

- All file uploads validated by extension & MIME type.
- Temporary files cleaned after processing.
- LLM responses validated as JSON before usage.
- Overlays only generated for recognized fields.

---

## 8. Directory Structure
```
fastapi_all_in_one_proj/
├── clients/
│   ├── llm.py
│   ├── markitdown_client.py
│   └── ppstructure_client.py
├── core/
│   ├── overlays.py
│   ├── parse.py
│   └── pipeline.py
├── tests/
│   ├── test_overlays_and_bundle.py
│   ├── test_overlays_multi_page.py
│   ├── test_pipeline_ext.py
│   ├── test_pipeline_images_pdf.py
│   └── test_policy_errors_reports.py
├── main.py
└── requirements.txt
```

---

## 9. License
MIT License
