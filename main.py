
# =============================================================================
# MAIN ORCHESTRATOR (formatted version with locations integration)
# -----------------------------------------------------------------------------
# - Markdown-first conversion + tokens with bboxes (PyMuPDF or OCR).
# - LLM always (single-pass or field-wise RAG based on context window).
# - Optional PP-Structure policy: auto|always|never|auto_pages.
# - Token-level alignment to produce multi-bbox + per-request forensic report.
# - NEW: attach_locations_to_response integrates PPStructureLight (tables+cells) to enrich fields.*.locations[].
# =============================================================================

from __future__ import annotations

import os, time, json, re, uuid, mimetypes
from typing import List, Dict, Any, Optional, Callable

from fastapi import FastAPI, UploadFile, File, Form, Depends, Response, HTTPException, Header
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from config import *
from logger import setup_logging, get_logger
import indexer, retriever, align, reports
from parse import (
    convert_markdown_async,
    extract_words_with_bboxes_pdf,
    parse_with_ppstructure_async,
    build_markdown_from_pp,
)
from llm import extract_fields_async
import jobs

# NEW: integration hook
try:
    from integrations.bbox_integration import attach_locations_to_response
except Exception:
    attach_locations_to_response = None  # type: ignore

# Some clients import submodules through clients_pkg (keep as in original)
import clients as clients_pkg  # noqa: F401
from clients import *  # noqa: F401,F403

setup_logging()
log = get_logger(__name__)

app = FastAPI(
    title="Template-Guided RAG Extractor",
    docs_url="/docs" if DOCS_ENABLED else None,
    redoc_url="/redoc" if DOCS_ENABLED else None,
    openapi_url="/openapi.json" if DOCS_ENABLED else None,
)

# ---------------- Security ----------------
def get_api_key(x_api_key: Optional[str] = Header(None)):
    required = os.getenv("API_KEY")
    if required and x_api_key != required:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True

# ---------------- Models ----------------
class Template(BaseModel):
    name: str = Field(default="default")
    fields: List[str]
    llm_text: str

class AppError(Exception):
    def __init__(self, code: int, kind: str, msg: str):
        self.code, self.kind, self.msg = code, kind, msg

@app.exception_handler(AppError)
async def app_error_handler(_, exc: AppError):
    return JSONResponse({"error": exc.kind, "message": exc.msg}, status_code=exc.code)

# Simple JSON logger line
def jlog(event: str, **k):
    rec = {"evt": event, "ts": int(time.time() * 1000), **k}
    log.info(json.dumps(rec))

# ---------------- Core processing ----------------
async def _process_request(
    data: bytes,
    filename: str,
    tpl_json: dict,
    req_id: str,
    emit: Callable[[str], None] | None = None,
) -> dict:  # noqa: C901
    try:
        schema = Template(**tpl_json)
    except (ValidationError, TypeError) as e:
        raise AppError(400, "BadTemplate", f"Invalid template: {str(e)}")

    field_details: Dict[str, Any] = {}
    manifest = {
        "request_id": req_id,
        "file": filename,
        "template": schema.name,
        "policy": {
            "rag_topk": RAG_TOPK,
            "rag_weights": {"bm25": RAG_W_BM25, "vec": RAG_W_VEC, "anchor": RAG_W_ANCHOR},
            "ppstruct_policy": PPSTRUCT_POLICY,
            "strict_validation": STRICT_VALIDATION,
            "llm": {
                "n_ctx": LLM_N_CTX,
                "seed": LLM_SEED,
                "temperature": LLM_TEMPERATURE,
                "json_strict": bool(LLM_JSON_STRICT),
            },
        },
    }
    artifacts: Dict[str, str] = {}

    if emit:
        emit("markdown_start")
    t_markdown0 = time.time()
    markdown = await convert_markdown_async(data, filename)
    t_markdown = time.time() - t_markdown0

    # Decide input type
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    is_pdf = (mime == "application/pdf") or (data[:4] == b"%PDF")
    pages_words = extract_words_with_bboxes_pdf(data) if is_pdf else []

    # Decide digital vs raster based on character threshold
    total_chars = 0
    for pinfo in pages_words:
        for w in pinfo.get("words", []):
            t = w.get("text", "") or ""
            try:
                total_chars += len(t)
            except Exception:
                pass
    try:
        thr = int(TEXT_LAYER_MIN_CHARS)  # from config
    except Exception:
        thr = 1
    is_digital_text = bool(pages_words and total_chars >= thr)

    # Convert PDF text-layer words to normalized token bboxes (0..1)
    tokens = []
    if is_digital_text:
        for p in pages_words:
            pw, ph = float(p.get("page_w", 1.0)), float(p.get("page_h", 1.0))
            for w in p.get("words", []):
                x0, y0, x1, y1 = w.get("bbox", [0, 0, 0, 0])
                tokens.append(
                    {
                        "text": w.get("text", ""),
                        "page": p.get("page", 1),
                        "bbox": [x0 / pw, y0 / ph, x1 / pw, y1 / ph],
                        "line_id": None,
                    }
                )

    # PP-Structure policy (deterministic)
    def markdown_has_table(md: str) -> bool:
        lines = md.splitlines()
        for i in range(len(lines) - 1):
            if "|" in lines[i] and re.search(r"\|", lines[i]) and re.search(r"^\s*[:\-\| ]+$", lines[i + 1]):
                return True
        return False

    pp_should_run_global = False
    if PPSTRUCT_POLICY == "always":
        pp_should_run_global = True
    elif PPSTRUCT_POLICY == "never":
        pp_should_run_global = False
    elif PPSTRUCT_POLICY in ("auto", "auto_pages"):
        if markdown_has_table(markdown):
            pp_should_run_global = True
        elif tokens:
            pp_should_run_global = indexer.sniff_tables_from_tokens(
                tokens, PPSTRUCT_SNIFF_MIN_ROWS, PPSTRUCT_SNIFF_MIN_COLS, PPSTRUCT_SNIFF_COL_ALIGN_TOL
            )

    pages_blocks: List[Dict[str, Any]] = []
    t_pp = 0.0

    # For non-PDF inputs (images), or PDF raster pages (no text), we should run PP
    need_pp_for_content = (not is_pdf) or (not is_digital_text)
    if (pp_should_run_global or need_pp_for_content) and PPSTRUCT_POLICY != "auto_pages":
        if emit:
            emit("pp_start")
        t_pp0 = time.time()
        clients_pkg._COUNTERS["pp"] += 1
        pages_blocks = await parse_with_ppstructure_async(data, filename, pages=None)
        t_pp = time.time() - t_pp0
        jlog(
            "ppstructure_done",
            id=req_id,
            pages=len(pages_blocks),
            total_blocks=sum(len(pg.get("blocks", [])) for pg in pages_blocks),
        )

    # Build markdown from PP when no digital text is available
    if not is_digital_text and pages_blocks:
        markdown = build_markdown_from_pp(pages_blocks)

    # Retrieval chunking / mode
    chunks = indexer.split_markdown_into_chunks(markdown)
    md_tokens_est = indexer.approximate_tokens(markdown)
    overhead = RAG_CTX_MARGIN_TOKENS + 256
    c_eff = max(1, LLM_N_CTX - overhead)
    use_single_pass = (md_tokens_est <= c_eff) and (len(chunks) < RAG_MIN_SEGMENTS)

    manifest["llm_context_mode"] = "single_pass" if use_single_pass else "rag_field_wise"
    manifest["md_token_estimate"] = md_tokens_est
    manifest["c_eff"] = c_eff

    # Artifacts dir
    artifacts_dir = os.path.join(REPORTS_DIR, req_id)
    os.makedirs(artifacts_dir, exist_ok=True)
    with open(os.path.join(artifacts_dir, "md.txt"), "w", encoding="utf-8") as f:
        f.write(markdown)

    tok_path = os.path.join(artifacts_dir, "tokens.jsonl")
    with open(tok_path, "w", encoding="utf-8") as f:
        for tkn in tokens:
            f.write(json.dumps(tkn, ensure_ascii=False) + "\n")
    artifacts["tokens.jsonl"] = tok_path

    if pages_blocks:
        tbl_path = os.path.join(artifacts_dir, "tables.json")
        with open(tbl_path, "w", encoding="utf-8") as f:
            json.dump(pages_blocks, f, ensure_ascii=False, indent=2)
        artifacts["tables.json"] = tbl_path

    # RAG / LLM
    idx = None
    if not use_single_pass:
        anchors = [str(f).lower() for f in schema.fields]
        idx = retriever.EphemeralIndex(chunks, anchors=anchors)

    global_chunk_tokens = tokens  # from PDF text layer if available

    results: List[Dict[str, Any]] = []

    if use_single_pass:
        t_llm0 = time.time()
        fields_out = await extract_fields_async([f for f in schema.fields], schema.llm_text, markdown)
        t_llm = int((time.time() - t_llm0) * 1000)
        jlog("llm_single_pass_done", id=req_id, ms=t_llm)

        for key in schema.fields:
            item = fields_out.get(key, {}) or {}
            val = (item.get("value") or "")
            llm_conf = float(item.get("confidence") or 0.0)
            tok_idx, coverage = align.align_value_to_tokens(val, global_chunk_tokens)
            bboxes = [global_chunk_tokens[i]["bbox"] for i in tok_idx]

            # Fallback bbox (first PP text/cell block) when raster with no aligned tokens
            if not bboxes and (not is_digital_text) and pages_blocks:
                for _pg in pages_blocks:
                    blk = next((b for b in _pg.get("blocks", []) if b.get("type") in ("text", "cell")), None)
                    if blk:
                        pw, ph = float(_pg.get("page_w", 1.0)), float(_pg.get("page_h", 1.0))
                        x0, y0, x1, y1 = blk.get("bbox", [0, 0, 0, 0])
                        bboxes = [[x0 / pw, y0 / ph, x1 / pw, y1 / ph]]
                        break

            pages = [global_chunk_tokens[i]["page"] for i in tok_idx]
            confidence = max(0.0, min(1.0, 0.7 * coverage + 0.3 * llm_conf))
            results.append(
                {
                    "key": key,
                    "value": val or None,
                    "confidence": round(confidence, 4),
                    "bboxes": bboxes,
                    "bbox_pages": pages,
                }
            )

            field_details[key] = {
                "mode": "single_pass",
                "llm_ms": t_llm,
                "confidence": round(confidence, 4),
                "validation_status": "skipped",
                "retrieval": [],
                "prompt": schema.llm_text,
                "context": (markdown[:4000] + ("... [truncated]" if len(markdown) > 4000 else "")),
                "llm_raw": json.dumps(item, ensure_ascii=False, indent=2),
                "token_indices": tok_idx,
                "bboxes": bboxes,
                "bbox_pages": pages,
            }
    else:
        # field-wise RAG
        for key in schema.fields:
            hits = idx.search(key, topk=RAG_TOPK)  # type: ignore
            ctx_parts, retrieval_dump = [], []
            for idx_id, score in hits:
                ch = chunks[idx_id]
                ctx_parts.append(ch["text"])
                retrieval_dump.append(
                    {"chunk_id": idx_id, "score": score, "kind": ch["kind"], "start": ch["start"], "len": len(ch["text"])}
                )
            context = "\n\n".join(ctx_parts)
            t_llm0 = time.time()
            fields_out = await extract_fields_async([key], schema.llm_text, context)
            t_llm = int((time.time() - t_llm0) * 1000)
            item = fields_out.get(key, {}) or {}
            val = (item.get("value") or "")
            llm_conf = float(item.get("confidence") or 0.0)
            tok_idx, coverage = align.align_value_to_tokens(val, global_chunk_tokens)
            bboxes = [global_chunk_tokens[i]["bbox"] for i in tok_idx]
            pages = [global_chunk_tokens[i]["page"] for i in tok_idx]
            confidence = max(0.0, min(1.0, 0.7 * coverage + 0.3 * llm_conf))
            results.append(
                {
                    "key": key,
                    "value": val or None,
                    "confidence": round(confidence, 4),
                    "bboxes": bboxes,
                    "bbox_pages": pages,
                }
            )
            field_details[key] = {
                "mode": "rag_field_wise",
                "llm_ms": t_llm,
                "confidence": round(confidence, 4),
                "validation_status": "skipped",
                "retrieval": retrieval_dump,
                "prompt": schema.llm_text,
                "context": context,
                "llm_raw": json.dumps(item, ensure_ascii=False, indent=2),
                "token_indices": tok_idx,
                "bboxes": bboxes,
                "bbox_pages": pages,
            }

    # Overlays (debug images)
    debug_files = []
    if os.getenv("DEBUG_OVERLAY", "0") in ("1", "true", "yes"):
        matches_per_page: Dict[int, List[Dict[str, Any]]] = {}
        for item in results:
            key = item["key"]
            bxs = item.get("bboxes", []) or []
            pgs = item.get("bbox_pages", []) or [1] * len(bxs)
            for bbox_norm, pg in zip(bxs, pgs):
                matches_per_page.setdefault(int(pg or 1), []).append({"bbox_norm": bbox_norm, "label": key})
        out_dir = os.path.join(DEBUG_DIR, req_id)
        try:
            import overlay as _overlay  # local helper to draw overlays
            debug_files = _overlay.save_overlays(data, matches_per_page, out_dir, filename or "input.bin")
        except Exception as _e:  # best-effort
            jlog("overlay_error", id=req_id, error=str(_e))

    # ---------------- NEW: map to per-field dict and attach locations ----------------
    # 1) map list -> dict (value+confidence only)
    fields_map: Dict[str, Dict[str, Any]] = {}
    for r in results:
        fields_map[r["key"]] = {"value": r.get("value"), "confidence": r.get("confidence", 0.0)}

    # 2) persist original input to artifacts dir for downstream processors
    input_path = os.path.join(artifacts_dir, filename or "input.bin")
    try:
        with open(input_path, "wb") as f:
            f.write(data)
    except Exception as _e:
        jlog("input_save_error", id=req_id, error=str(_e))

    # 3) base response
    response = {
        "request_id": req_id,
        "template": schema.name,
        "text": markdown,
        "fields": fields_map,
        "debug_overlays": debug_files,
        "status": "done",
    }

    # 4) enrich with locations via PPStructureLight integration (if available)
    if attach_locations_to_response is not None:
        try:
            response = attach_locations_to_response(input_path, response)  # adds locations[] + aliases
        except Exception as _e:
            jlog("locations_attach_error", id=req_id, error=str(_e))

    manifest.update(
        {
            "timings_ms": {"markdown": int(t_markdown * 1000), "pp": int(t_pp * 1000)},
            "ppstructure": {"policy": PPSTRUCT_POLICY},
        }
    )

    # Save response bundle
    artifacts["response.json"] = os.path.join(REPORTS_DIR, req_id, "response.json")
    os.makedirs(os.path.dirname(artifacts["response.json"]), exist_ok=True)
    with open(artifacts["response.json"], "w", encoding="utf-8") as f:
        json.dump(response, f, ensure_ascii=False, indent=2)

    reports.save_report_bundle(req_id, manifest, field_details, artifacts)
    return response

# ---------------- Routes ----------------
@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    template: str = Form(...),
    _auth_ok: bool = Depends(get_api_key),
):
    data = await file.read()
    try:
        tpl = json.loads(template)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid template JSON")
    req_id = str(uuid.uuid4())
    result = await _process_request(data, file.filename, tpl, req_id)
    return JSONResponse(result)

@app.post("/jobs")
async def submit_job(
    file: UploadFile = File(...),
    template: str = Form(...),
    priority: int = Form(5),
    _auth_ok: bool = Depends(get_api_key),
):
    data = await file.read()
    try:
        tpl = json.loads(template)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid template JSON")
    req_id = str(uuid.uuid4())

    def handler(payload: dict):
        return _process_request(data, file.filename, tpl, req_id)

    try:
        jid = jobs.global_q.submit({"req_id": req_id}, priority=priority)
    except RuntimeError:
        return JSONResponse({"error": "QueueFull"}, status_code=429)
    if not jobs.global_q.workers:
        jobs.global_q.start(n_workers=int(os.getenv("MAX_CONCURRENT_JOBS", "2")), handler=handler)
    return JSONResponse({"job_id": jid, "request_id": req_id})

@app.get("/jobs/{job_id}")
async def job_status(job_id: str, _auth_ok: bool = Depends(get_api_key)):
    res = jobs.global_q.get_result(job_id)
    return JSONResponse(res or {"status": "pending"})

@app.get("/jobs/{job_id}/events")
async def job_events(job_id: str, _auth_ok: bool = Depends(get_api_key)):
    async def event_gen():
        for ev in jobs.global_q.get_events(job_id):
            yield f"event: {ev['event']}\ndata: {json.dumps(ev['data'])}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")

@app.get("/metrics")
async def metrics_route():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/reports/{request_id}")
async def get_report_json(request_id: str, _auth_ok: bool = Depends(get_api_key)):
    base = os.path.join(REPORTS_DIR, request_id)
    path = os.path.join(base, "report.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(data)

@app.get("/reports/{request_id}/bundle.zip")
async def get_report_zip(request_id: str, _auth_ok: bool = Depends(get_api_key)):
    base = os.path.join(REPORTS_DIR, request_id)
    if not os.path.isdir(base):
        raise HTTPException(status_code=404, detail="Report not found")
    zbytes = reports.zip_report_dir(base)
    return Response(
        content=zbytes,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={request_id}.zip"},
    )
