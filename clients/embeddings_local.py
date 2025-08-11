# clients/embeddings_local.py
from __future__ import annotations
import os
from typing import List
import numpy as np
from llama_cpp import Llama
from logger import get_logger

EMB_PATH = os.getenv("EMBEDDINGS_GGUF_PATH", "/models/embeddings.gguf")
EMB_THREADS = int(os.getenv("EMB_THREADS", str(os.cpu_count() or 4)))
EMB_N_CTX = int(os.getenv("EMB_N_CTX", "512"))
EMB_GPU_LAYERS = int(os.getenv("EMB_GPU_LAYERS", "0"))

_EMB: Llama | None = None
log = get_logger(__name__)


def get_local_embedder() -> Llama:
    global _EMB
    if _EMB is None:
        if not os.path.exists(EMB_PATH):
            raise RuntimeError(f"Embeddings model not found at {EMB_PATH}")
        log.info("Initializing local embedder from %s", EMB_PATH)
        try:
            _EMB = Llama(
                model_path=EMB_PATH,
                embedding=True,
                n_threads=EMB_THREADS,
                n_ctx=EMB_N_CTX,
                n_gpu_layers=EMB_GPU_LAYERS,
                verbose=False,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embeddings model: {e}") from e
    return _EMB


def embed_texts(texts: List[str]) -> List[List[float]]:
    log.info("Computing embeddings for %d texts", len(texts))
    llm = get_local_embedder()
    out = llm.create_embedding(texts)
    if isinstance(out, dict) and "data" in out:
        log.info("Embeddings computed via dict response")
        vecs = [d["embedding"] for d in out["data"]]
    elif isinstance(out, list):
        log.info("Embeddings computed via list response")
        vecs = out
    else:
        raise RuntimeError("Unexpected embedding output from llama-cpp")
    log.info("Returning %d embedding vectors", len(vecs))
    return vecs
