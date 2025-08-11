# llm.py — wrapper using local llama.cpp (async)
from __future__ import annotations
from typing import List, Dict, Any
import os, asyncio
from clients.llm_local import chat_json
from logger import get_logger

MOCK_LLM = os.getenv("MOCK_LLM", "0") in ("1","true","True")
log = get_logger(__name__)

async def extract_fields_async(fields: List[str], llm_text: str, context: str) -> Dict[str, Dict[str, Any]]:
    if MOCK_LLM:
        log.info("MOCK_LLM enabled; returning empty fields for %s", fields)
        return {k: {"value": None, "confidence": 0.0} for k in fields}
    log.info("Calling LLM for fields %s", fields)
    loop = asyncio.get_event_loop()
    res = await loop.run_in_executor(None, chat_json, fields, llm_text, context)
    log.info("LLM returned data for fields %s", list(res.keys()))
    return res
