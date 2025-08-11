# clients/llm_local.py
from __future__ import annotations
import os, json, re
from typing import Dict, Any, List
from llama_cpp import Llama
from logger import get_logger

LLM_GGUF_PATH   = os.getenv("LLM_GGUF_PATH", "/models/llm.gguf")
LLAMA_N_CTX     = int(os.getenv("LLM_N_CTX", "4096"))
LLAMA_N_THREADS = int(os.getenv("LLM_N_THREADS", str(os.cpu_count() or 4)))
LLAMA_BATCH     = int(os.getenv("LLM_BATCH", "512"))
LLAMA_GPU_LAYERS= int(os.getenv("LLM_GPU_LAYERS", "0"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_SEED        = int(os.getenv("LLM_SEED", "42"))

_GLOBAL_LLM: Llama | None = None
_JSON_FENCE = re.compile(r"\{.*\}", re.DOTALL)
log = get_logger(__name__)

def get_local_llm() -> Llama:
    global _GLOBAL_LLM
    if _GLOBAL_LLM is None:
        log.info("Initializing local LLM")
        _GLOBAL_LLM = Llama(
            model_path=LLM_GGUF_PATH,
            n_ctx=LLAMA_N_CTX,
            n_threads=LLAMA_N_THREADS,
            n_gpu_layers=LLAMA_GPU_LAYERS,
            seed=LLM_SEED,
            verbose=False,
        )
    return _GLOBAL_LLM

def _build_prompt(fields: List[str], llm_text: str, context: str) -> str:
    field_list = ", ".join(fields)
    return f"""
You are an information extractor. Given the CONTEXT and the EXTRACTION_GUIDE, output a compact JSON with the requested fields.
- Output only JSON, no prose.
- If a value is missing, use null and confidence 0.0.

EXTRACTION_GUIDE:
{llm_text}

REQUESTED_FIELDS: [{field_list}]

CONTEXT:
{context}

JSON SCHEMA EXAMPLE:
{{
  "field_name": {{"value": "<string|null>", "confidence": <0..1>}}
}}
"""

def chat_json(fields: List[str], llm_text: str, context: str) -> Dict[str, Dict[str, Any]]:
    log.info("Calling local LLM for fields %s", fields)
    llm = get_local_llm()
    prompt = _build_prompt(fields, llm_text, context).strip()
    out = llm.create_completion(
        prompt=prompt,
        max_tokens=2048,
        temperature=LLM_TEMPERATURE,
        stop=[],
    )
    text = out["choices"][0]["text"]
    m = _JSON_FENCE.search(text)
    raw = m.group(0) if m else text
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            cleaned = {}
            for k in fields:
                v = data.get(k, {})
                if isinstance(v, dict):
                    cleaned[k] = {"value": v.get("value"), "confidence": float(v.get("confidence") or 0.0)}
                else:
                    cleaned[k] = {"value": v, "confidence": 0.0}
            return cleaned
    except Exception:
        pass
    return {k: {"value": None, "confidence": 0.0} for k in fields}
