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
