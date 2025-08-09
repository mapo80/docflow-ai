"""
overlay_middleware.py
---------------------
FastAPI/Starlette middleware that rewrites DocFlow AI responses so that:
- Each logical field becomes an object with optional `locations` (array of positions),
- Each location is **minimal**: { "bbox": [x, y, width, height], "page_index": 0 },
- The first location is mirrored to `bbox` and `page_index` at field level (aliases).

It supports two input response shapes:
A) {"fields": [ { "key": "...", "value": ..., "confidence": ..., "bboxes": [...], "bbox_pages": [...] }, ... ], ...}
B) {"fields": { "<name>": {...} }, "overlays": [ { "field": "<name>", "bbox": [...], "page_index": N }, ... ], ...}

Set EMBED_OVERLAYS=0 to bypass rewriting (pass-through).
"""

import json
import os
from typing import Any, Dict, Iterable, List, Tuple
from starlette.responses import Response

ZEROISH = (0, 0.0, None, False)

def _safe_json_loads(body: bytes) -> Any:
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        return None

def _to_int0(v) -> int:
    try:
        return int(v)
    except Exception:
        return 0

def _to_xywh(bb: List[float]) -> List[float]:
    """
    Heuristic conversion:
    - If looks like corners [x0,y0,x1,y1] with x1>x0, y1>y0 -> convert to [x,y,w,h]
    - Else assume already [x,y,w,h]
    """
    if not isinstance(bb, (list, tuple)) or len(bb) != 4:
        return [0, 0, 0, 0]
    x0, y0, x2, y2 = bb
    try:
        if float(x2) > float(x0) and float(y2) > float(y0):
            return [float(x0), float(y0), max(0.0, float(x2)-float(x0)), max(0.0, float(y2)-float(y0))]
    except Exception:
        pass
    # assume already xywh
    return [float(x0), float(y0), float(x2), float(y2)]

def _merge_from_overlays(payload: Dict[str, Any]) -> Dict[str, Any]:
    fields = payload.get("fields")
    overlays = payload.get("overlays")
    if not isinstance(fields, dict) or not isinstance(overlays, list):
        return payload

    for ov in overlays:
        if not isinstance(ov, dict):
            continue
        name = ov.get("field")
        bbox = ov.get("bbox")
        page_index = ov.get("page_index", 0)
        if not name or not isinstance(bbox, list) or name not in fields:
            continue

        fobj = fields.get(name) or {}
        locs = fobj.get("locations")
        if not isinstance(locs, list):
            locs = []
        locs.append({"bbox": _to_xywh(bbox), "page_index": _to_int0(page_index)})
        fobj["locations"] = locs
        # aliases
        if "bbox" not in fobj:
            fobj["bbox"] = locs[0]["bbox"]
        if "page_index" not in fobj:
            fobj["page_index"] = locs[0]["page_index"]
        fields[name] = fobj

    payload = dict(payload)
    payload.pop("overlays", None)
    payload["fields"] = fields
    return payload

def _merge_from_list(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform fields list (shape A) into dict with locations.
    """
    fields_list = payload.get("fields")
    if not isinstance(fields_list, list):
        return payload

    fmap: Dict[str, Any] = {}
    for it in fields_list:
        if not isinstance(it, dict):
            continue
        name = it.get("key")
        if not name:
            continue

        bxs = it.get("bboxes") or []
        pgs = it.get("bbox_pages") or [1] * len(bxs)
        locs = []
        for bb, pg in zip(bxs, pgs):
            if isinstance(bb, (list, tuple)) and len(bb) == 4:
                locs.append({"bbox": _to_xywh(list(bb)), "page_index": max(0, _to_int0(pg) - 1)})
        fobj = {"value": it.get("value"), "confidence": it.get("confidence")}
        if locs:
            fobj["locations"] = locs
            fobj["bbox"] = locs[0]["bbox"]
            fobj["page_index"] = locs[0]["page_index"]
        fmap[name] = fobj

    payload = dict(payload)
    payload["fields"] = fmap
    return payload

async def overlay_embedder(request, call_next) -> Response:
    """
    Starlette/FastAPI middleware.

    Behavior:
      - If EMBED_OVERLAYS is unset or "1": rewrite response JSON to embed per-field locations.
      - If EMBED_OVERLAYS is "0": pass through.
    """
    if os.getenv("EMBED_OVERLAYS", "1") in ("0", "false", "False", "no", "NO"):
        return await call_next(request)

    resp = await call_next(request)

    # Read response body
    try:
        body = b""
        async for chunk in resp.body_iterator:
            body += chunk
    except Exception:
        return resp

    payload = _safe_json_loads(body)
    if not isinstance(payload, dict):
        # not JSON, pass-through
        return Response(content=body, status_code=resp.status_code, headers=dict(resp.headers), media_type=resp.media_type)

    # Two possible shapes; apply both transforms if applicable
    if isinstance(payload.get("fields"), list):
        payload = _merge_from_list(payload)

    if "fields" in payload and "overlays" in payload:
        payload = _merge_from_overlays(payload)

    new_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return Response(content=new_body, status_code=resp.status_code, headers=dict(resp.headers), media_type="application/json")
