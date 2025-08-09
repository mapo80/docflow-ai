"""
serve.py
--------
Alternative entrypoint that wraps your existing FastAPI app from `main.py` and
registers the overlay middleware without editing your current file.

Run with:
    uvicorn serve:app --host 0.0.0.0 --port 8000
"""

from main import app as _app  # your existing app object
from overlay_middleware import overlay_embedder

# Expose the wrapped app
app = _app
app.middleware("http")(overlay_embedder)
