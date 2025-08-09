from __future__ import annotations
import httpx, os
from typing import Optional
from config import MARKITDOWN_BASE_URL, HTTP_TIMEOUT_MS, BACKENDS_MOCK
from logger import get_logger
log = get_logger(__name__)

async def convert_bytes_to_markdown_async(data: bytes, filename: str, mime: Optional[str]=None) -> str:
    # Short-circuit in mock mode (env-checked at call time)
    import os as _os
    if _os.getenv('BACKENDS_MOCK','0') == '1':
        try:
            return data.decode('utf-8', errors='ignore')
        except Exception:
            return '(binary)'
    base = os.getenv('MARKITDOWN_BASE_URL', MARKITDOWN_BASE_URL)
    timeout = httpx.Timeout(HTTP_TIMEOUT_MS/1000.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            files = {'file': (filename, data, mime or 'application/octet-stream')}
            r = await client.post(f"{base.rstrip('/')}/convert", files=files)
            r.raise_for_status()
            return r.text
    except Exception as e:
        log.warning('MarkItDown service not available (%s). Falling back to naive conversion.', e)
        try:
            return data.decode('utf-8', errors='ignore')
        except Exception:
            return '(binary)'
