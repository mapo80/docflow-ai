# clients/embeddings_local.py
from __future__ import annotations
import os
from typing import List
from logger import get_logger

log = get_logger(__name__)

# Optional heavy deps; imported lazily
from llama_cpp import Llama  # type: ignore
from huggingface_hub import hf_hub_download  # type: ignore
import clients  # namespace package for optional monkeypatched funcs

EMB_PATH = os.getenv("EMBEDDINGS_GGUF_PATH", "/models/embeddings.gguf")
EMB_THREADS = int(os.getenv("EMB_THREADS", str(os.cpu_count() or 4)))
EMB_N_CTX = int(os.getenv("EMB_N_CTX", "512"))
EMB_GPU_LAYERS = int(os.getenv("EMB_GPU_LAYERS", "0"))
HF_REPO = os.getenv("EMB_HF_REPO", "Mungert/Qwen3-Embedding-0.6B-GGUF")
HF_FILE = os.getenv("EMB_HF_FILE", "Qwen3-Embedding-0.6B-q4_k_m.gguf")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

_EMB: Llama | None = None


def get_local_embedder() -> Llama:
    global _EMB
    if _EMB is None:
        model_path = EMB_PATH
        if not os.path.exists(model_path):
            log.info(
                "Embeddings model missing at %s, downloading from HuggingFace repo %s",
                model_path,
                HF_REPO,
            )
            try:
                model_path = hf_hub_download(
                    repo_id=HF_REPO,
                    filename=HF_FILE,
                    token=HF_TOKEN,
                )
            except Exception as e:  # pragma: no cover - network issues
                log.error("Failed to download embeddings model: %s", e)
                raise RuntimeError(f"Failed to download embeddings model: {e}") from e
            log.info("Downloaded embeddings model to %s", model_path)
        log.info("Initializing local embedder from %s", model_path)
        try:
            _EMB = Llama(
                model_path=model_path,
                embedding=True,
                n_threads=EMB_THREADS,
                n_ctx=EMB_N_CTX,
                n_gpu_layers=EMB_GPU_LAYERS,
                verbose=False,
            )
        except Exception as e:
            log.error("Failed to initialize embeddings model: %s", e)
            raise RuntimeError(f"Failed to initialize embeddings model: {e}") from e
    return _EMB


def _embed_llm(texts: List[str]) -> List[List[float]]:
    fn = getattr(clients, "llm_embed", None)
    if callable(fn):
        log.info("Using monkeypatched llm_embed for %d texts", len(texts))
        vecs = fn(texts)
        return vecs.tolist() if hasattr(vecs, "tolist") else vecs
    log.info("Using local GGUF embedder for %d texts", len(texts))
    llm = get_local_embedder()
    out = llm.create_embedding(texts)
    if isinstance(out, dict) and "data" in out:
        vecs = [d["embedding"] for d in out["data"]]
    elif isinstance(out, list):
        vecs = out
    else:
        raise RuntimeError("Unexpected embedding output from llama-cpp")
    return vecs


def embed_texts(texts: List[str]) -> List[List[float]]:
    log.info("Computing embeddings for %d texts using GGUF", len(texts))
    return _embed_llm(texts)
