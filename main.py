# main.py — drop-in replacement
# FastAPI app with full pipeline and locations integration
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

# integration hook (added)
try:
    from integrations.bbox_integration import attach_locations_to_response
except Exception:
    attach_locations_to_response = None  # type: ignore

# keep import side-effects for clients pkg (if any)
import clients as clients_pkg  # noqa: F401
from clients import *  # noqa: F401,F403

setup_logging()
log = get_logger(__name__)

docs_url = "/docs" if DOCS_ENABLED else None
redoc_url = "/redoc" if DOCS_ENABLED else None
openapi_url = "/openapi.json" if DOCS_ENABLED else None

app = FastAPI(
    title="DocFlow AI",
    docs_url=docs_url,
    redoc_url=redoc_url,
    openapi_url=openapi_url,
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
    log.info("Starting _process_request for %s", filename)
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
            "ppstruct_policy": os.getenv("PPSTRUCT_POLICY", "auto"),
        },
    }
    artifacts: Dict[str, str] = {}

    if emit:
        emit("markdown_start")
    t_markdown0 = time.time()
    log.info("Converting document to markdown")
    markdown = await convert_markdown_async(data, filename)
    t_markdown = time.time() - t_markdown0
    log.info("Markdown conversion completed in %.3fs", t_markdown)

    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    is_pdf = (mime == "application/pdf") or (data[:4] == b"%PDF")
    log.info("Extracting words with bboxes from PDF: %s", is_pdf)
    pages_words = extract_words_with_bboxes_pdf(data) if is_pdf else []

    total_chars = 0
    for p in pages_words:
        for w in p.get("words", []):
            t = w.get("text", "") or ""
            total_chars += len(t)
    thr = int(os.getenv("TEXT_LAYER_MIN_CHARS", "1"))
    is_digital_text = bool(pages_words and total_chars >= thr)

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

    # simple heuristic for PP-Structure
    def markdown_has_table(md: str) -> bool:
        lines = md.splitlines()
        for i in range(len(lines) - 1):
            if "|" in lines[i] and re.search(r"\|", lines[i]) and re.search(r"^\s*[:\-\| ]+$", lines[i + 1]):
                return True
        return False

    PPSTRUCT_POLICY = os.getenv("PPSTRUCT_POLICY", "auto")
    pp_should_run_global = PPSTRUCT_POLICY == "always" or (PPSTRUCT_POLICY in ("auto", "auto_pages") and markdown_has_table(markdown))
    need_pp_for_content = (not is_pdf) or (not is_digital_text)

    pages_blocks: List[Dict[str, Any]] = []
    t_pp = 0.0
    if (pp_should_run_global or need_pp_for_content) and PPSTRUCT_POLICY != "auto_pages":
        if emit:
            emit("pp_start")
        log.info("Calling PP-Structure analysis")
        t_pp0 = time.time()
        pages_blocks = await parse_with_ppstructure_async(data, filename, pages=None)
        t_pp = time.time() - t_pp0
        log.info("PP-Structure returned %d pages in %.3fs", len(pages_blocks), t_pp)
        jlog(
            "ppstructure_done",
            id=req_id,
            pages=len(pages_blocks),
            total_blocks=sum(len(pg.get("blocks", [])) for pg in pages_blocks),
        )

    if not is_digital_text and pages_blocks:
        markdown = build_markdown_from_pp(pages_blocks)

    log.info("Splitting markdown into chunks")
    chunks = indexer.split_markdown_into_chunks(markdown)
    md_tokens_est = indexer.approximate_tokens(markdown)
    overhead = int(os.getenv("RAG_CTX_MARGIN_TOKENS", "256")) + 256
    LLM_N_CTX = int(os.getenv("LLM_N_CTX", "4096"))
    c_eff = max(1, LLM_N_CTX - overhead)
    use_single_pass = (md_tokens_est <= c_eff) and (len(chunks) < int(os.getenv("RAG_MIN_SEGMENTS", "12")))

    manifest["llm_context_mode"] = "single_pass" if use_single_pass else "rag_field_wise"
    manifest["md_token_estimate"] = md_tokens_est
    manifest["c_eff"] = c_eff

    log.info("LLM context mode: %s", manifest["llm_context_mode"])

    artifacts_dir = os.path.join(os.getenv("REPORTS_DIR", "reports"), req_id)
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

    results: List[Dict[str, Any]] = []
    if use_single_pass:
        t_llm0 = time.time()
        log.info("Calling LLM for all fields in single pass")
        fields_out = await extract_fields_async([f for f in schema.fields], schema.llm_text, markdown)
        t_llm = int((time.time() - t_llm0) * 1000)
        log.info("LLM single pass completed in %d ms", t_llm)
        jlog("llm_single_pass_done", id=req_id, ms=t_llm)

        global_chunk_tokens = tokens
        for key in schema.fields:
            item = fields_out.get(key, {}) or {}
            val = (item.get("value") or "")
            llm_conf = float(item.get("confidence") or 0.0)
            # simple alignment (no-op if tokens empty)
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
    else:
        # field-wise RAG
        anchors = [str(f).lower() for f in schema.fields]
        log.info("Creating RAG index")
        idx = retriever.EphemeralIndex(chunks, anchors=anchors)
        global_chunk_tokens = tokens
        for key in schema.fields:
            log.info("Searching index for field %s", key)
            hits = idx.search(key, topk=int(os.getenv("RAG_TOPK", "6")))
            ctx = "\n\n".join(chunks[h[0]]["text"] for h in hits)
            t_llm0 = time.time()
            log.info("Calling LLM for field %s", key)
            fields_out = await extract_fields_async([key], schema.llm_text, ctx)
            t_llm = int((time.time() - t_llm0) * 1000)
            log.info("LLM returned for field %s in %d ms", key, t_llm)
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

    # Optional debug overlays
    debug_files = []
    if os.getenv("DEBUG_OVERLAY", "0") in ("1", "true", "yes"):
        matches_per_page: Dict[int, List[Dict[str, Any]]] = {}
        for item in results:
            key = item["key"]
            bxs = item.get("bboxes", []) or []
            pgs = item.get("bbox_pages", []) or [1] * len(bxs)
            for bbox_norm, pg in zip(bxs, pgs):
                matches_per_page.setdefault(int(pg or 1), []).append({"bbox_norm": bbox_norm, "label": key})
        out_dir = os.path.join(os.getenv("DEBUG_DIR", "debug"), req_id)
        try:
            import overlay as _overlay
            debug_files = _overlay.save_overlays(data, matches_per_page, out_dir, filename or "input.bin")
        except Exception as _e:
            jlog("overlay_error", id=req_id, error=str(_e))

    # Map list -> dict
    fields_map: Dict[str, Dict[str, Any]] = {}
    for r in results:
        fields_map[r["key"]] = {"value": r.get("value"), "confidence": r.get("confidence", 0.0)}

    # Persist input for PP integration
    artifacts_dir = os.path.join(os.getenv("REPORTS_DIR", "reports"), req_id)
    os.makedirs(artifacts_dir, exist_ok=True)
    input_path = os.path.join(artifacts_dir, filename or "input.bin")
    try:
        with open(input_path, "wb") as f:
            f.write(data)
    except Exception as _e:
        jlog("input_save_error", id=req_id, error=str(_e))

    response = {
        "request_id": req_id,
        "template": schema.name,
        "text": markdown,
        "fields": fields_map,
        "debug_overlays": debug_files,
        "status": "done",
    }

    # Enrich fields with locations[] (bbox+page_index) via PPStructureLight
    if attach_locations_to_response is not None:
        try:
            response = attach_locations_to_response(input_path, response)
        except Exception as _e:
            jlog("locations_attach_error", id=req_id, error=str(_e))

    manifest.update({"timings_ms": {"markdown": int(t_markdown * 1000)}})

    # Save response bundle
    rdir = artifacts_dir
    with open(os.path.join(rdir, "response.json"), "w", encoding="utf-8") as f:
        json.dump(response, f, ensure_ascii=False, indent=2)

    reports.save_report_bundle(
        req_id, manifest, {}, {"response.json": os.path.join(rdir, "response.json")}
    )
    log.info("Completed _process_request for %s", filename)
    return response

# ---------------- Routes ----------------

@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    template: str = Form(...),
    pp_policy: str = Form("auto"),
    overlays: bool = Form(False),
    _auth_ok: bool = Depends(get_api_key),
):
    data = await file.read()
    try:
        tpl = json.loads(template)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid template")
    req_id = str(uuid.uuid4())
    res = await _process_request(data, file.filename, tpl, req_id)
    return JSONResponse(res)
@app.post("/process-document")
async def process_document(
    file: UploadFile = File(...),
    pp_policy: str = Form("auto"),
    llm_model: Optional[str] = Form(None),
    overlays: bool = Form(False),
    _auth_ok: bool = Depends(get_api_key),
):
    data = await file.read()
    try:
        tpl = {"name": "default", "fields": os.getenv("FIELDS", "invoice_number,invoice_date,total").split(","), "llm_text": os.getenv("LLM_TEXT", "Extract fields.")}
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid template")
    req_id = str(uuid.uuid4())
    res = await _process_request(data, file.filename, tpl, req_id)
    return JSONResponse(res)


@app.get("/reports/{rid}")
async def get_report(rid: str, ok: bool = Depends(get_api_key)):
    path = os.path.join(REPORTS_DIR, rid, "report.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(data)


@app.get("/reports/{rid}/bundle.zip")
async def get_report_bundle(rid: str, ok: bool = Depends(get_api_key)):
    dir_path = os.path.join(REPORTS_DIR, rid)
    if not os.path.isdir(dir_path):
        raise HTTPException(status_code=404, detail="Report not found")
    data = reports.zip_report_dir(dir_path)
    return Response(data, media_type="application/zip")

@app.get("/metrics")
async def metrics_route():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
async def health():
    return JSONResponse({"ok": True})
