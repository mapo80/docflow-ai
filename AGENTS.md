# Agent Instructions

## Setup
- Use Python 3.12+
- Install dependencies: `pip install -r requirements.txt`

## Testing
- Run the full test suite from the repository root:
  - `PYTHONPATH=. pytest`
- Generate coverage reports:
  - `PYTHONPATH=. pytest --cov=. --cov-report=term-missing --cov-report=html`
- External services (LLM, OCR, etc.) are mocked by default via `BACKENDS_MOCK=1` and `MOCK_LLM=1`. Disable these by setting the variables to `0`.

## Development server
- Launch the FastAPI app for manual testing:
  - `uvicorn main:app --reload`

## DocTR & GGUF verification
- Ensure runtime dependencies:
  - System: `apt-get install -y libgl1` (required by OpenCV)
  - Python: `pip install uvicorn fastapi python-doctr pymupdf pdf2image python-multipart prometheus_client llama-cpp-python huggingface_hub setuptools`
- Export required variables:
  - `MOCK_LLM=1`
  - `MOCK_OCR=0`
  - `BACKENDS_MOCK=0`
  - `HUGGINGFACE_TOKEN=<your_hf_token>`
- Start the server:
  - `uvicorn main:app --reload`
- From another shell, send a request:
  - `curl -X POST "http://localhost:8000/process-document" -F "file=@dataset/sample_invoice.png" -F "ocr_policy=auto" -F "overlays=true"`
- Check server logs for:
  - `OCR` analysis being invoked
  - GGUF embedder initialization
- The server performs a warmup at startup that downloads DocTR models and the GGUF embedder; the first launch may take several
  minutes. Subsequent requests should be fast once "Warmup finished" appears in logs.
- If the server fails, test components individually:
  - `python - <<'PY'\nfrom clients.doctr_client import analyze_async\nimport asyncio\nprint(asyncio.run(analyze_async(open('dataset/sample_invoice.png','rb').read(),'sample_invoice.png',None)))\nPY`
  - `python - <<'PY'\nfrom clients.embeddings_local import embed_texts\nprint(len(embed_texts(['hello'])[0]))\nPY`
