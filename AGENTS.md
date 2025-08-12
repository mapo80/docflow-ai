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
  - Python: `pip install uvicorn fastapi paddlepaddle paddleocr pymupdf pdf2image python-multipart prometheus_client`
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
  - `PP-Structure` analysis being invoked
  - GGUF embedder initialization
