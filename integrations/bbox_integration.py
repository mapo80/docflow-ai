# integrations/bbox_integration.py
from __future__ import annotations
from typing import Dict, Any
from clients.ppstructure_light import PPStructureLight
from services.bbox_mapper import map_bboxes_to_fields

def attach_locations_to_response(doc_path: str, response: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(response, dict) or "fields" not in response:
        return response
    pp = PPStructureLight(use_gpu=False, include_cell_text=True)
    tokens = pp.extract_tokens(doc_path)
    fields_map = response.get("fields") if isinstance(response.get("fields"), dict) else {}
    fields_map = map_bboxes_to_fields(fields_map, tokens)
    response["fields"] = fields_map
    return response
