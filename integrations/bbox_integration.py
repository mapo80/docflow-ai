# integrations/bbox_integration.py
from __future__ import annotations
from typing import Dict, Any
from clients.doctr_client import _get_doctr
from services.bbox_mapper import map_bboxes_to_fields

def attach_locations_to_response(doc_path: str, response: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(response, dict) or "fields" not in response:
        return response
    doctr = _get_doctr()
    pages = doctr.extract_pages(doc_path)
    tokens = []
    for pg in pages:
        for blk in pg.get("blocks", []):
            if blk.get("type") == "text" and blk.get("text"):
                tokens.append(
                    {
                        "category": "text",
                        "bbox": blk.get("bbox", []),
                        "page_index": pg.get("page", 1) - 1,
                        "text": blk.get("text", ""),
                    }
                )
    fields_map = response.get("fields") if isinstance(response.get("fields"), dict) else {}
    fields_map = map_bboxes_to_fields(fields_map, tokens)
    response["fields"] = fields_map
    return response
