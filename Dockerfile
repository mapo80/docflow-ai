# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1     PIP_DISABLE_PIP_VERSION_CHECK=1     DEBIAN_FRONTEND=noninteractive

# --- System deps commonly needed by OCR/markup libs and OpenCV ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl ca-certificates \
    libglib2.0-0 libgl1 \
    libjpeg62-turbo libpng16-16 \
    poppler-utils libmagic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python deps first (for better layer caching) ---
# If your repo has requirements.txt it will be used; otherwise this step is skipped.
COPY requirements.txt /app/requirements.txt
RUN if [ -s requirements.txt ]; then \
      python -m pip install --upgrade pip && \
      pip install -r requirements.txt; \
    else echo "No requirements.txt found or empty, skipping."; fi
# ensure server available
RUN pip install "uvicorn[standard]" "gunicorn"

# --- App code ---
COPY . /app

# --- Models ---
# Provide GGUF model URLs at build-time. Leave empty to skip download.
ARG LLM_GGUF_URL=
ARG EMB_GGUF_URL=
# Optional SHA256 to verify downloads (empty to skip verification)
ARG LLM_GGUF_SHA256=
ARG EMB_GGUF_SHA256=

ENV MODELS_DIR=/models \
    LLM_GGUF_PATH=/models/llm.gguf \
    EMBEDDINGS_GGUF_PATH=/models/embeddings.gguf \
    EMBED_OVERLAYS=1

RUN mkdir -p "${MODELS_DIR}"
COPY docker/download_models.sh /usr/local/bin/download_models.sh
RUN bash /usr/local/bin/download_models.sh "${LLM_GGUF_URL}" "${EMB_GGUF_URL}" "${LLM_GGUF_SHA256}" "${EMB_GGUF_SHA256}" "${MODELS_DIR}"

# --- Runtime ---
COPY docker/start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh /usr/local/bin/download_models.sh

EXPOSE 8000
ENV HOST=0.0.0.0 \
    PORT=8000

# If serve.py exists, it registers the overlay middleware; otherwise main:app is used.
CMD ["/usr/local/bin/start.sh"]
