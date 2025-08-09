from __future__ import annotations
import os, json, threading, asyncio
from typing import List, Dict, Any
from config import LLM_SEED, LLM_TEMPERATURE, LLM_JSON_STRICT
from clients.llm_client import chat_json_async

async def create_chat_completion_async(messages: list, max_tokens: int = 1024):
    return await chat_json_async(messages, max_tokens=max_tokens)


def create_chat_completion(messages: List[Dict[str,str]], max_tokens: int = 1024):
    # Synchronous wrapper using the async client (for reuse in existing pipeline)
    return asyncio.get_event_loop().run_until_complete(chat_json_async(messages, max_tokens=max_tokens))

async def extract_fields_async(field_names: List[str], llm_text: str, context_markdown: str) -> Dict[str, Any]:
    """Async version of extract_fields (mock-friendly)."""
    system = (
        f"You are an information extractor. Be precise, conservative, and honest.\n"
        f"Extraction policy: {llm_text}\n"
        "Return a JSON dictionary where each requested field is present; confidence in [0,1]."
    )
    user = f"Fields: {field_names}\n\nDocument (Markdown):\n```markdown\n{context_markdown}\n```"
    messages=[{"role":"system","content":system},{"role":"user","content":user}]
    out = await create_chat_completion_async(messages, max_tokens=1024)
    content = out["choices"][0]["message"]["content"]
    import json as _json, re as _re
    # try raw
    try:
        return _json.loads(content)
    except Exception:
        pass
    # try to find a JSON object anywhere (handles code fences and prefix/suffix)
    m = _re.search(r"\{[\s\S]*\}", content)
    if m:
        try:
            return _json.loads(m.group(0))
        except Exception:
            pass
    return {fn: {"value": None, "confidence": 0.0} for fn in field_names}
def extract_fields(field_names: List[str], llm_text: str, context_markdown: str) -> Dict[str, Any]:
    """Build a strict prompt and call the LLM to extract the field set.

    Returns a dict mapping field -> {value, confidence}
    """
    system = (
        "You extract structured information from a Markdown document. "
        "Reply ONLY with a valid JSON object where keys are the requested field names. "
        "Each value MUST be an object: {\"value\": <string|null>, \"confidence\": <0..1>}."
    )
    user = f"Fields: {field_names}\n\nDocument (Markdown):\n```markdown\n{context_markdown}\n```"
    messages=[{"role":"system","content":system},{"role":"user","content":user}]
    out = create_chat_completion(messages, max_tokens=1024)
    content = out["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except Exception:
        try:
            return json.loads(content.strip().split("```")[-1])
        except Exception:
            return {fn: {"value": None, "confidence": 0.0} for fn in field_names}
