# Dockerfile — downloads GGUF models and runs app
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl ca-certificates \
    libglib2.0-0 libgl1 \
    libjpeg62-turbo libpng16-16 \
    poppler-utils libmagic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && pip install -r requirements.txt
# extra deps for local LLM/emb
RUN pip install "uvicorn[standard]" "gunicorn" "llama-cpp-python>=0.2.85"

COPY . /app

ARG LLM_GGUF_URL=
ARG EMB_GGUF_URL=
ARG LLM_GGUF_SHA256=
ARG EMB_GGUF_SHA256=

ENV MODELS_DIR=/models \
    LLM_GGUF_PATH=/models/llm.gguf \
    EMBEDDINGS_GGUF_PATH=/models/embeddings.gguf \
    EMBED_OVERLAYS=1

RUN mkdir -p "${MODELS_DIR}"
COPY docker/download_models.sh /usr/local/bin/download_models.sh
RUN bash /usr/local/bin/download_models.sh "${LLM_GGUF_URL}" "${EMB_GGUF_URL}" "${LLM_GGUF_SHA256}" "${EMB_GGUF_SHA256}" "${MODELS_DIR}"

COPY docker/start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh /usr/local/bin/download_models.sh

EXPOSE 8000
ENV HOST=0.0.0.0 PORT=8000
CMD ["/usr/local/bin/start.sh"]
