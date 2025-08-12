import os
from typing import Tuple

def get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def get_env_str(name: str, default: str) -> str:
    return os.getenv(name, default)

# RAG / retrieval params
RAG_TOPK               = get_env_int("RAG_TOPK", 5)
RAG_W_BM25             = get_env_float("RAG_W_BM25", 0.5)
RAG_W_VEC              = get_env_float("RAG_W_VEC", 0.4)
RAG_W_ANCHOR           = get_env_float("RAG_W_ANCHOR", 0.1)
RAG_CTX_MARGIN_TOKENS  = get_env_int("RAG_CTX_MARGIN_TOKENS", 256)
RAG_MIN_SEGMENTS       = get_env_int("RAG_MIN_SEGMENTS", 12)

# LLM / determinism
LLM_N_CTX              = get_env_int("LLM_N_CTX", 8192)
LLM_SEED               = get_env_int("LLM_SEED", 42)
LLM_TEMPERATURE        = get_env_float("LLM_TEMPERATURE", 0.0)
LLM_JSON_STRICT        = get_env_int("LLM_JSON_STRICT", 1)

# Validation policy
STRICT_VALIDATION      = get_env_str("STRICT_VALIDATION", "soft")  # soft|hard|off

# OCR policy
OCR_POLICY             = get_env_str("OCR_POLICY", "auto")    # auto|always|never|auto_pages

# Logging
LOG_LEVEL = get_env_str("LOG_LEVEL", "INFO")  # DEBUG|INFO|WARNING|ERROR

# Swagger/OpenAPI docs
DOCS_ENABLED = get_env_int("DOCS_ENABLED", 1)  # 1=enable interactive docs

# Reports
REPORTS_DIR = get_env_str("REPORTS_DIR", "./data/reports")
REPORT_TTL_HOURS = get_env_int("REPORT_TTL_HOURS", 72)

def renormalize_weights(w_bm25: float, w_vec: float, w_anchor: float) -> Tuple[float,float,float]:
    total = max(1e-6, w_bm25 + w_vec + w_anchor)
    return w_bm25/total, w_vec/total, w_anchor/total

# Embeddings (GGUF only)
EMBEDDING_GGUF_PATH    = get_env_str("EMBEDDING_GGUF_PATH", "/models/Qwen3-Embedding-0.6B-Q8_0.gguf")
EMBEDDING_POOLING      = get_env_str("EMBEDDING_POOLING", "last")  # last|mean|cls (if supported by llama.cpp build)
EMBEDDING_N_THREADS    = get_env_int("EMBEDDING_N_THREADS", 4)
EMBEDDING_NORMALIZE    = get_env_int("EMBEDDING_NORMALIZE", 1)


# External service base URLs (can be localhost in single-container)
MARKITDOWN_BASE_URL = get_env_str("MARKITDOWN_BASE_URL", "http://127.0.0.1:8001")
LLM_BASE_URL        = get_env_str("LLM_BASE_URL",        "http://127.0.0.1:8003")

# Preflight + policy
TEXT_LAYER_MIN_CHARS = get_env_int("TEXT_LAYER_MIN_CHARS", 200)
 
# Testing / mocks
BACKENDS_MOCK = get_env_int("BACKENDS_MOCK", 1)  # 1=use mock responses for external services (MarkItDown, OCR, LLM)
HTTP_TIMEOUT_MS = get_env_int("HTTP_TIMEOUT_MS", 60000)

DEBUG_DIR = get_env_str("DEBUG_DIR", "./data/debug")
