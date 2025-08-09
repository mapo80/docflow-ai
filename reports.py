from __future__ import annotations
import os, json, time, pathlib, zipfile, io
from typing import Any, Dict
from config import REPORTS_DIR
from logger import get_logger
log = get_logger(__name__)

def ensure_dir(p: str) -> str:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True); return p

def save_report_bundle(request_id: str, manifest: Dict[str,Any], field_details: Dict[str,Any], artifacts: Dict[str,str]) -> str:
    base = ensure_dir(os.path.join(REPORTS_DIR, request_id))
    j = {
        "request_id": request_id,
        "created_at": int(time.time()*1000),
        "manifest": manifest,
        "fields": field_details,
        "artifacts": artifacts
    }
    with open(os.path.join(base, "report.json"), "w", encoding="utf-8") as f:
        json.dump(j, f, ensure_ascii=False, indent=2)
    with open(os.path.join(base, "report.md"), "w", encoding="utf-8") as f:
        f.write(_render_markdown_report(j))
    return base

def _render_markdown_report(d: Dict[str,Any]) -> str:
    m = d["manifest"]; fields = d["fields"]; artifacts = d.get("artifacts", {})
    out = []
    out.append(f"# Forensic Report — request `{d['request_id']}`\n")
    out.append("## Manifest\n```json\n")
    out.append(json.dumps(m, ensure_ascii=False, indent=2))
    out.append("\n```\n\n## Artifacts\n")
    for k,v in artifacts.items():
        out.append(f"- **{k}**: `{v}`\n")
    for key, det in fields.items():
        out.append("\n---\n")
        out.append(f"## Field: `{key}`\n\n")
        out.append(f"**Mode**: `{det.get('mode')}`  \n**LLM ms**: {det.get('llm_ms')}  \n**Confidence**: {det.get('confidence')}  \n**Validation**: {det.get('validation_status')}\n\n")
        out.append("### Retrieval (Top-K chunks)\n```json\n")
        out.append(json.dumps(det.get('retrieval', []), ensure_ascii=False, indent=2))
        out.append("\n```\n\n### Prompt (sanitized)\n```text\n")
        out.append(det.get("prompt",""))
        out.append("\n```\n\n### Context sent to LLM\n```markdown\n")
        out.append(det.get("context",""))
        out.append("\n```\n\n### LLM Raw Output\n```json\n")
        out.append(det.get("llm_raw","{}"))
        out.append("\n```\n\n### Token Alignment\n```json\n")
        out.append(json.dumps({
            "token_indices": det.get("token_indices", []),
            "bbox_pages": det.get("bbox_pages", []),
            "bboxes": det.get("bboxes", [])
        }, ensure_ascii=False, indent=2))
        out.append("\n```\n")
        if det.get("table_cell_bbox"):
            out.append("\n### Table cell bbox\n```json\n")
            out.append(json.dumps(det["table_cell_bbox"], ensure_ascii=False, indent=2))
            out.append("\n```\n")
    return ''.join(out)

def zip_report_dir(dir_path: str) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(dir_path):
            for fn in files:
                full = os.path.join(root, fn)
                arc = os.path.relpath(full, dir_path)
                z.write(full, arc)
    mem.seek(0)
    return mem.read()
