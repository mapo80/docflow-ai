# llm.py — wrapper using local llama.cpp (async)
from __future__ import annotations
from typing import List, Dict, Any
import os, asyncio
from clients.llm_local import chat_json

MOCK_LLM = os.getenv("MOCK_LLM", "0") in ("1","true","True")

async def extract_fields_async(fields: List[str], llm_text: str, context: str) -> Dict[str, Dict[str, Any]]:
    if MOCK_LLM:
        return {k: {"value": None, "confidence": 0.0} for k in fields}
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, chat_json, fields, llm_text, context)
