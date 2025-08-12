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

## Paddle & GGUF verification
- Ensure runtime dependencies:
  - System: `apt-get install -y libgl1`
  - Python: `pip install uvicorn fastapi paddlepaddle paddleocr pymupdf pdf2image python-multipart prometheus_client llama-cpp-python huggingface_hub setuptools`
- Export required variables:
  - `MOCK_LLM=1`
  - `MOCK_PP=0`
  - `BACKENDS_MOCK=0`
  - `HUGGINGFACE_TOKEN=<your_hf_token>`
- Start the server:
  - `uvicorn main:app --reload`
- From another shell, send a request:
  - `curl -X POST "http://localhost:8000/process-document" -F "file=@dataset/sample_invoice.png" -F "pp_policy=auto" -F "overlays=true"`
- Check server logs for:
  - `PP-Structure` analysis being invoked (Chart2Table and formula modules are disabled by default to prevent segfaults)
  - GGUF embedder initialization
- If the server fails, test components individually:
  - `python - <<'PY'\nfrom clients.ppstructure_light import analyze_async\nimport asyncio\nprint(asyncio.run(analyze_async(open('dataset/sample_invoice.png','rb').read(),'sample_invoice.png',None)))\nPY`
  - `python - <<'PY'\nfrom clients.embeddings_local import embed_texts\nprint(len(embed_texts(['hello'])[0]))\nPY`
