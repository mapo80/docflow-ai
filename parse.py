from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import fitz, io, mimetypes, os, asyncio, re

from config import MARKITDOWN_BASE_URL, PPSTRUCT_POLICY, TEXT_LAYER_MIN_CHARS, ALLOW_PP_ON_DIGITAL
from clients.markitdown_client import convert_bytes_to_markdown_async
import clients.ppstructure_light as ppc

def _guess_mime(filename: str, header: bytes) -> str:
    m = mimetypes.guess_type(filename)[0]
    if m: return m
    if header[:4] == b'%PDF': return 'application/pdf'
    if header[:8] == b'\x89PNG\r\n\x1a\n': return 'image/png'
    if header[:3] == b'\xff\xd8\xff': return 'image/jpeg'
    return 'application/octet-stream'

async def convert_markdown_async(data: bytes, filename: str = "input.bin") -> str:
    """Async version: PDF -> PyMuPDF text, else call MarkItDown async; fallback to naive decode."""
    import fitz
    mime = _guess_mime(filename, data[:8])
    if mime == 'application/pdf':
        try:
            doc = fitz.open(stream=data, filetype="pdf")
            parts = []
            for p in doc:
                parts.append(p.get_text("text").strip()+"\n")
            return "\n".join(parts).strip() or "(empty)"
        except Exception:
            pass
    try:
        return await convert_bytes_to_markdown_async(data, filename, mime)
    except Exception:
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return "(binary)"

def convert_markdown(data: bytes, filename: str = "input.bin") -> str:

    """Markdown-first: for PDFs use PyMuPDF/plain text; for images or other types, use MarkItDown (OCR disabled).

    This is a *fallback* path—on raster/images we prefer PP-Structure to also get tokens & tables.
    """
    mime = _guess_mime(filename, data[:8])
    if mime == 'application/pdf':
        try:
            doc = fitz.open(stream=data, filetype="pdf")
            parts = []
            for p in doc:
                text = p.get_text("text")
                parts.append(text.strip()+"\n")
            md = "\n".join(parts)
            return md if md.strip() else "(empty)"
        except Exception:
            pass
    # Non-PDF: call MarkItDown service (OCR disabled on that service)
    try:
        raise RuntimeError('convert_markdown() used in sync path; use convert_markdown_async in async contexts')
    except Exception:
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return "(binary)"

def extract_words_with_bboxes_pdf(data: bytes) -> list:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:
        return []
    pages = []
    for pno, page in enumerate(doc, start=1):
        words = page.get_text("words")  # tuples
        wlist = []
        for w in words:
            x0,y0,x1,y1, word, block_no, line_no, word_no = w
            wlist.append({"text": word, "bbox":[x0,y0,x1,y1]})
        pages.append({"page": pno, "page_w": page.rect.width, "page_h": page.rect.height, "words": wlist})
    return pages

async def parse_with_ppstructure_async(data: bytes, filename: str, pages: Optional[list]=None):
    """Async call to PP-Structure service."""
    return await ppc.analyze_async(data, filename, pages=pages)

def parse_with_ppstructure(data: bytes, filename: str, pages: Optional[list]=None):

    """Call PP-Structure service (async) and return page blocks; empty on failure."""
    return asyncio.get_event_loop().run_until_complete(ppc.analyze_async(data, filename, pages=pages))

def build_markdown_from_pp(pages_blocks: list) -> str:
    """Construct simple Markdown from PP-Structure blocks (lines/tables)."""
    out = []
    for pg in pages_blocks:
        for b in pg.get("blocks", []):
            typ = b.get("type","text")
            if typ == "table":
                out.append(b.get("markdown",""))
            else:
                out.append(b.get("text",""))
        out.append("\n")
    return "\n".join(out).strip() or "(empty)"
