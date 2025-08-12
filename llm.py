# llm.py — wrapper using local llama.cpp (async)
from __future__ import annotations
from typing import List, Dict, Any
import os, asyncio
import clients.llm_local as llm_local
from logger import get_logger

log = get_logger(__name__)

def _mock_llm_enabled() -> bool:
    """Check if the LLM should be mocked based on the current environment."""
    return os.getenv("MOCK_LLM", "0") in ("1", "true", "True")

async def extract_fields_async(fields: List[str], llm_text: str, context: str) -> Dict[str, Dict[str, Any]]:
    if _mock_llm_enabled():
        log.info("MOCK_LLM enabled; returning empty fields for %s", fields)
        return {k: {"value": None, "confidence": 0.0} for k in fields}
    log.info("Calling LLM for fields %s", fields)
    loop = asyncio.get_event_loop()
    res = await loop.run_in_executor(None, llm_local.chat_json, fields, llm_text, context)
    log.info("LLM returned data for fields %s", list(res.keys()))
    return res
