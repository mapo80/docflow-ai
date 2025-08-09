"""
services/bbox_mapper.py
-----------------------
Map PPStructureLight tokens to your fields.<name>.locations[]

Strategy:
- Prefer matching against **cell tokens with text** (exact/loose string match).
- If multiple matches, keep the first (or all if you want); we currently keep ALL.
- Only bbox/page_index are stored in locations; text is NOT stored in locations.

You can customize scoring or include regex normalization as needed.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import re
from difflib import SequenceMatcher

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def map_bboxes_to_fields(fields_map: Dict[str, Dict[str, Any]], tokens: List[Dict[str, Any]], min_ratio: float = 0.82) -> Dict[str, Dict[str, Any]]:
    """
    fields_map: { name: { "value": "...", "confidence": float, ... }, ... }
    tokens: from PPStructureLight.extract_tokens()
    returns: fields_map with added locations/bbox/page_index where matches are found.
    """
    # Build index of cells with text
    cell_tokens = [t for t in tokens if t.get("category") == "cell" and isinstance(t.get("text"), str) and t["text"].strip()]
    for fname, fobj in (fields_map or {}).items():
        val = _norm(str(fobj.get("value", "")))
        if not val:
            continue
        locs = []
        for t in cell_tokens:
            txt = _norm(t.get("text", ""))
            if not txt:
                continue
            if val == txt or (len(val) > 3 and _similar(val, txt) >= min_ratio) or (val in txt) or (txt in val and len(txt) > 3):
                locs.append({"bbox": t["bbox"], "page_index": t["page_index"]})
        if locs:
            fobj["locations"] = locs
            fobj["bbox"] = locs[0]["bbox"]
            fobj["page_index"] = locs[0]["page_index"]
            fields_map[fname] = fobj
    return fields_map
