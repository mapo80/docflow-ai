from __future__ import annotations
import httpx, os, json
from typing import Optional, List, Dict, Any
from config import PPSTRUCT_BASE_URL, HTTP_TIMEOUT_MS, BACKENDS_MOCK
from logger import get_logger
log = get_logger(__name__)

async def analyze_async(data: bytes, filename: str, pages: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """Call PP-Structure service. Returns list per-page: {page, blocks:[{type,text,bbox,...}]}.
    On failure, returns mock list.
    """
    import os as _os
    if _os.getenv('BACKENDS_MOCK','0') == '1':
        return [{
            'page': 1,
            'page_w': 600,
            'page_h': 800,
            'blocks': [
                {'type':'text','text':'MOCK FIELD','bbox':[60,60,260,110]},
                {'type':'text','text':'ALTRO','bbox':[80,200,300,240]},
            ]
        }]
    base = os.getenv('PPSTRUCT_BASE_URL', PPSTRUCT_BASE_URL)
    timeout = httpx.Timeout(HTTP_TIMEOUT_MS/1000.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            files = {'file': (filename, data, 'application/octet-stream')}
            payload = {'pages': pages or []}
            r = await client.post(f"{base.rstrip('/')}/analyze", files=files, data={'payload': json.dumps(payload)})
            r.raise_for_status()
            return r.json()
    except Exception as e:
        log.warning('PP-Structure service not available (%s). Returning mock result.', e)
        return [{
            'page': 1,
            'page_w': 600,
            'page_h': 800,
            'blocks': [
                {'type':'text','text':'MOCK FIELD','bbox':[60,60,260,110]},
                {'type':'text','text':'ALTRO','bbox':[80,200,300,240]},
            ]
        }]