from __future__ import annotations
import os, asyncio, tempfile
from typing import List, Dict, Any, Optional

import clients
from logger import get_logger

_DOCTR_IMPORT_ERROR: Optional[Exception] = None
try:  # pragma: no cover - optional heavy dependency
    from doctr.io import DocumentFile  # type: ignore
    from doctr.models import ocr_predictor  # type: ignore
except Exception as e:  # pragma: no cover - doctr not installed
    DocumentFile = None  # type: ignore
    ocr_predictor = None  # type: ignore
    _DOCTR_IMPORT_ERROR = e

log = get_logger(__name__)

__all__ = ["DocTRClient", "analyze_async"]


class DocTRClient:
    def __init__(self) -> None:
        if DocumentFile is None or ocr_predictor is None:
            raise RuntimeError("python-doctr is not installed")
        self.model = ocr_predictor(pretrained=True)

    def _load(self, path: str):
        if path.lower().endswith(".pdf"):
            return DocumentFile.from_pdf(path)
        return DocumentFile.from_images(path)

    def extract_pages(self, path: str) -> List[Dict[str, Any]]:
        doc = self._load(path)
        result = self.model(doc)
        doc_pages = getattr(doc, "pages", doc)
        result_pages = getattr(result, "pages", result)
        pages: List[Dict[str, Any]] = []
        for pidx, (img, page) in enumerate(zip(doc_pages, result_pages)):
            h, w = img.shape[0], img.shape[1]
            blocks = []
            for block in page.blocks:
                for line in block.lines:
                    txt = " ".join(word.value for word in line.words)
                    (x0, y0), (x1, y1) = line.geometry
                    blocks.append(
                        {
                            "type": "text",
                            "text": txt,
                            "bbox": [x0 * w, y0 * h, (x1 - x0) * w, (y1 - y0) * h],
                        }
                    )
            pages.append(
                {
                    "page": pidx + 1,
                    "page_w": float(w),
                    "page_h": float(h),
                    "blocks": blocks,
                }
            )
        return pages


_DOCTR_INSTANCE: DocTRClient | None = None


def _get_doctr() -> DocTRClient:
    global _DOCTR_INSTANCE
    if _DOCTR_INSTANCE is None:
        if os.getenv("MOCK_OCR", "0") == "1":
            class _Mock:
                def extract_pages(self, path: str) -> List[Dict[str, Any]]:
                    return [{"page": 1, "page_w": 1.0, "page_h": 1.0, "blocks": []}]
            log.info("MOCK_OCR=1 - using DocTR mock")
            _DOCTR_INSTANCE = _Mock()  # type: ignore[assignment]
        elif DocumentFile is None or ocr_predictor is None:
            class _Stub:
                def extract_pages(self, path: str) -> List[Dict[str, Any]]:
                    return [{"page": 1, "page_w": 1.0, "page_h": 1.0, "blocks": []}]
            if _DOCTR_IMPORT_ERROR is not None:
                log.warning("DocTR import failed: %s", _DOCTR_IMPORT_ERROR)
            log.warning("python-doctr not installed; using stub")
            _DOCTR_INSTANCE = _Stub()  # type: ignore[assignment]
        else:
            log.info("Creating DocTRClient instance")
            _DOCTR_INSTANCE = DocTRClient()
    return _DOCTR_INSTANCE


def _analyze_sync(data: bytes, filename: str, pages: Optional[list] = None) -> List[Dict[str, Any]]:
    doctr = _get_doctr()
    suffix = os.path.splitext(filename)[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        return doctr.extract_pages(path)
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


async def analyze_async(data: bytes, filename: str, pages: Optional[list] = None) -> List[Dict[str, Any]]:
    log.info("Calling DocTR analyze_async")
    try:
        clients._mock_counters["ocr"] += 1
    except Exception:
        pass
    loop = asyncio.get_event_loop()
    res = await loop.run_in_executor(None, _analyze_sync, data, filename, pages)
    log.info("DocTR analyze_async completed")
    return res
