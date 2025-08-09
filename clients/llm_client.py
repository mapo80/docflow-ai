from __future__ import annotations
import httpx, os, json
from typing import List, Dict, Any
from config import LLM_BASE_URL, HTTP_TIMEOUT_MS, LLM_SEED, LLM_TEMPERATURE, LLM_JSON_STRICT
from logger import get_logger
log = get_logger(__name__)

def _is_mock():
    return os.getenv("MOCK_LLM","0") in ("1","true","TRUE")

async def chat_json_async(messages: List[Dict[str,str]], max_tokens: int = 1024) -> Dict[str, Any]:
    if _is_mock():
        # Deterministic mock: produce JSON keyed by any "Fields:" line in user content
        # We assume our orchestrator passes field_names in the user content.
        fields = []
        for m in messages:
            if m.get("role")=="user" and "Fields:" in m.get("content",""):
                # naive parse
                import re, json as _json
                mtxt = m["content"]
                # fields list may be printed like ['a', 'b'] in content; try to find it
                arr = re.findall(r"Fields:\s*\[(.*?)\]", mtxt)
                if arr:
                    raw = "["+arr[0]+"]"
                    try:
                        # Python-like list with quotes
                        fields = _json.loads(raw.replace("'","\""))
                    except Exception:
                        pass
        out = { fn: {"value": f"MOCK_{fn.upper()}", "confidence": 0.9} for fn in fields }
        return {"choices":[{"message":{"role":"assistant","content":json.dumps(out, ensure_ascii=False)}}]}
    base = os.getenv("LLM_BASE_URL", LLM_BASE_URL)
    timeout = httpx.Timeout(HTTP_TIMEOUT_MS/1000.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            payload = {
                "model": "llm-gguf",
                "messages": messages,
                "max_tokens": max_tokens,
                "seed": LLM_SEED,
                "temperature": LLM_TEMPERATURE,
            }
            if LLM_JSON_STRICT:
                payload["response_format"] = {"type":"json_object"}
            r = await client.post(f"{base.rstrip('/')}/v1/chat/completions", json=payload)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        log.error("LLM service error: %s", e)
        raise

async def embed_async(texts: List[str]) -> Dict[str, Any]:
    base = os.getenv("LLM_BASE_URL", LLM_BASE_URL)
    timeout = httpx.Timeout(HTTP_TIMEOUT_MS/1000.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            payload = {"model":"embedding-gguf", "input": texts}
            r = await client.post(f"{base.rstrip('/')}/v1/embeddings", json=payload)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        log.error("Embeddings service error: %s", e)
        raise
