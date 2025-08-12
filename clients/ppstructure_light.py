# clients/ppstructure_light.py
from __future__ import annotations
import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from pdf2image import convert_from_path
import asyncio, tempfile
from logger import get_logger
import clients

try:  # pragma: no cover - optional heavy dependency
    from paddleocr import PaddleOCR  # type: ignore

    try:
        from paddleocr import PPStructureV3  # type: ignore
    except Exception:  # pragma: no cover - some installs lack PPStructureV3
        PPStructureV3 = None  # type: ignore
except Exception:  # pragma: no cover - paddleocr not installed
    PaddleOCR = None  # type: ignore
    PPStructureV3 = None  # type: ignore

log = get_logger(__name__)

__all__ = ["PPStructureLight", "analyze_async"]


def _poly_to_xywh(poly) -> List[float]:
    """Normalize polygon or bbox into [x, y, w, h]."""
    arr = np.asarray(poly, dtype=float)
    if arr.ndim == 1:
        if arr.size >= 8:
            arr = arr.reshape(-1, 2)
        elif arr.size == 4:
            x0, y0, x1, y1 = arr
            if x1 > x0 and y1 > y0:
                return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]
            return [float(x0), float(y0), float(x1), float(y1)]
        else:
            return [0.0, 0.0, 0.0, 0.0]
    xs, ys = arr[:, 0], arr[:, 1]
    x, y = float(xs.min()), float(ys.min())
    w, h = float(xs.max() - x), float(ys.max() - y)
    return [x, y, max(0.0, w), max(0.0, h)]


class PPStructureLight:
    def __init__(
        self,
        use_gpu: bool = False,
        pdf_dpi: int | None = None,
        include_cell_text: bool = True,
    ):
        self.pdf_dpi = int(os.getenv("PDF_DPI", "200") if pdf_dpi is None else pdf_dpi)
        self.include_cell_text = include_cell_text
        # PaddleOCR changed the constructor signature in 3.x removing the
        # ``show_log`` argument. Construct the kwargs dynamically to stay
        # compatible with both old and new versions.
        if PaddleOCR is None:
            raise RuntimeError("PaddleOCR is not installed")
        kwargs = {"lang": "it"}
        try:  # pragma: no cover - defensive
            import inspect

            sig = inspect.signature(PaddleOCR.__init__)
            if "use_angle_cls" in sig.parameters:
                kwargs["use_angle_cls"] = False
            if "use_gpu" in sig.parameters:
                kwargs["use_gpu"] = use_gpu
            elif "device" in sig.parameters:
                kwargs["device"] = "gpu" if use_gpu else "cpu"
            if "show_log" in sig.parameters:
                kwargs["show_log"] = False
        except Exception:  # pragma: no cover - if inspect fails we just skip
            kwargs.update(use_angle_cls=False, use_gpu=use_gpu)
        try:
            self.det = PaddleOCR(**kwargs)
        except ModuleNotFoundError as e:  # pragma: no cover - missing paddle
            if "paddle" in str(e).lower():
                raise RuntimeError(
                    "paddlepaddle is required for PaddleOCR; install paddlepaddle or set BACKENDS_MOCK=1"
                ) from e
            raise
        if PPStructureV3 is None:
            raise RuntimeError(
                "PPStructureV3 unavailable: please install paddleocr>=2.7.0"
            )
        self.pp = PPStructureV3(
            layout_detection_model_name="PicoDet_layout_1x_table",
            layout_detection_model_dir=None,
        )

    def extract_tokens(self, path: str) -> List[Dict[str, Any]]:
        pages = self._load_pages(path)
        tokens: List[Dict[str, Any]] = []
        for pidx, img in pages:
            np_img = np.array(img)
            log.info("Running text detector on page %s", pidx)
            det_res = self.det.ocr(np_img)
            log.info("OCR raw result type=%s", type(det_res).__name__)
            lines: List[Any] = []
            if isinstance(det_res, list):
                cand = det_res
                if len(det_res) == 1 and isinstance(det_res[0], list):
                    cand = det_res[0]
                for obj in cand:
                    if isinstance(obj, (list, tuple)) and obj:
                        lines.append(obj)
                    elif isinstance(obj, dict):
                        poly = (
                            obj.get("poly")
                            or obj.get("bbox")
                            or obj.get("points")
                            or obj.get("text_box_position")
                        )
                        text = (
                            obj.get("text")
                            or obj.get("label")
                            or obj.get("transcription")
                            or ""
                        )
                        score = obj.get("score", 0.0)
                        if poly is not None:
                            lines.append([poly, [text, score]])
            elif isinstance(det_res, dict):
                cand = None
                for k in ("res", "result", "data", "boxes"):
                    v = det_res.get(k)
                    if isinstance(v, list):
                        cand = v
                        break
                if cand:
                    for obj in cand:
                        if isinstance(obj, (list, tuple)):
                            lines.append(obj)
                        elif isinstance(obj, dict):
                            poly = (
                                obj.get("poly")
                                or obj.get("bbox")
                                or obj.get("points")
                                or obj.get("text_box_position")
                            )
                            text = (
                                obj.get("text")
                                or obj.get("label")
                                or obj.get("transcription")
                                or ""
                            )
                            score = obj.get("score", 0.0)
                            if poly is not None:
                                lines.append([poly, [text, score]])
            log.info("Detected %d text lines on page %s", len(lines), pidx)
            for item in lines:
                poly = (
                    item[0]
                    if isinstance(item, (list, tuple)) and len(item) > 0
                    else None
                )
                if poly is None:
                    continue
                txt = ""
                if len(item) > 1 and isinstance(item[1], (list, tuple)):
                    txt = str(item[1][0])
                tokens.append(
                    {
                        "category": "text",
                        "bbox": _poly_to_xywh(poly),
                        "page_index": pidx,
                        "text": txt,
                    }
                )
            tables, cell_tokens = self._table_regions_and_cells(np_img, pidx)
            for bb in tables:
                tokens.append({"category": "table", "bbox": bb, "page_index": pidx})
            tokens.extend(cell_tokens)
        return tokens

    def _load_pages(self, path: str) -> List[Tuple[int, "PIL.Image.Image"]]:
        lower = path.lower()
        if lower.endswith(".pdf"):
            pil_pages = convert_from_path(path, dpi=self.pdf_dpi)
            return list(enumerate(pil_pages))
        else:
            from PIL import Image

            return [(0, Image.open(path).convert("RGB"))]

    def _table_regions_and_cells(self, np_img, pidx: int):
        res = self.pp.predict(
            input=np_img,
            use_region_detection=True,
            use_table_recognition=True,
            use_seal_recognition=False,
            use_formula_recognition=False,
            visualize=False,
        )
        tables_xywh: List[List[float]] = []
        cell_tokens: List[Dict[str, Any]] = []

        r = res[0] if isinstance(res, list) and res else res

        layout_lists = []
        for attr in ("layout_res_list", "layout_res", "res", "layoutParsingResults"):
            v = getattr(r, attr, None) if not isinstance(r, dict) else r.get(attr)
            if isinstance(v, list) and v:
                layout_lists.append(v)
        if not layout_lists and isinstance(r, list) and r and isinstance(r[0], dict):
            layout_lists.append(r)

        def _as_xywh(bb):
            if isinstance(bb, (list, tuple)) and len(bb) == 4:
                x0, y0, w_or_x1, h_or_y1 = bb
                try:
                    if float(w_or_x1) > float(x0) and float(h_or_y1) > float(y0):
                        return [
                            float(x0),
                            float(y0),
                            float(w_or_x1) - float(x0),
                            float(h_or_y1) - float(y0),
                        ]
                except Exception:
                    pass
                return [float(x0), float(y0), float(w_or_x1), float(h_or_y1)]
            return None

        for cand in layout_lists:
            for obj in cand:
                if not isinstance(obj, dict):
                    continue
                lab = (
                    (obj.get("type") or obj.get("cls") or obj.get("label") or "")
                    .strip()
                    .lower()
                )
                if lab.startswith("table"):
                    if "bbox" in obj:
                        xywh = _as_xywh(obj["bbox"])
                        if xywh:
                            tables_xywh.append(xywh)
                    elif "poly" in obj:
                        tables_xywh.append(_poly_to_xywh(obj["poly"]))
                    elif "dt_polys" in obj and isinstance(obj["dt_polys"], list):
                        for poly in obj["dt_polys"]:
                            tables_xywh.append(_poly_to_xywh(poly))

        table_res_list = (
            getattr(r, "table_res_list", None)
            if not isinstance(r, dict)
            else r.get("table_res_list")
        )
        if isinstance(table_res_list, list):
            for t in table_res_list:
                cell_boxes = t.get("cell_box_list") or t.get("cell_boxes") or []
                cell_texts = t.get("cell_text_list") or t.get("cell_texts") or []
                cell_rc = t.get("cell_rc_list") or []
                for idx, cb in enumerate(cell_boxes):
                    xywh = _poly_to_xywh(cb) if isinstance(cb, (list, tuple)) else None
                    if not xywh:
                        continue
                    tok = {"category": "cell", "bbox": xywh, "page_index": pidx}
                    if self.include_cell_text:
                        txt = ""
                        if isinstance(cell_texts, list) and idx < len(cell_texts):
                            txt = str(cell_texts[idx] or "")
                        tok["text"] = txt
                    if (
                        isinstance(cell_rc, list)
                        and idx < len(cell_rc)
                        and isinstance(cell_rc[idx], (list, tuple))
                        and len(cell_rc[idx]) >= 2
                    ):
                        tok["row"], tok["col"] = int(cell_rc[idx][0]), int(
                            cell_rc[idx][1]
                        )
                    cell_tokens.append(tok)

        return tables_xywh, cell_tokens


_PP_INSTANCE: PPStructureLight | None = None


def _get_pp() -> PPStructureLight:
    """Return a singleton PPStructureLight or a lightweight stub.

    When PaddleOCR is not installed the real implementation cannot be
    constructed.  For tests we fall back to a minimal stub that yields one
    empty page so callers still receive a valid structure instead of a
    runtime error.
    """
    global _PP_INSTANCE
    if _PP_INSTANCE is None:
        if PaddleOCR is None or PPStructureV3 is None:
            class _Stub:
                def _load_pages(self, path):
                    from PIL import Image
                    return [(0, Image.new("RGB", (1, 1)))]

                def extract_tokens(self, path):
                    return []

            log.warning("PaddleOCR not installed; using PPStructureLight stub")
            _PP_INSTANCE = _Stub()  # type: ignore[assignment]
        else:
            log.info("Creating PPStructureLight instance")
            _PP_INSTANCE = PPStructureLight()
    return _PP_INSTANCE


def _analyze_sync(
    data: bytes, filename: str, pages: Optional[list] = None
) -> List[Dict[str, Any]]:
    pp = _get_pp()
    suffix = os.path.splitext(filename)[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        log.info("Loading pages for PPStructureLight")
        pages_imgs = pp._load_pages(tmp_path)
        tokens = pp.extract_tokens(tmp_path)
        pages_blocks: List[Dict[str, Any]] = []
        for pidx, img in pages_imgs:
            blocks = []
            for t in tokens:
                if t.get("page_index") != pidx:
                    continue
                cat = t.get("category", "text")
                blk: Dict[str, Any] = {
                    "type": cat if cat != "cell" else "text",
                    "bbox": t.get("bbox", []),
                }
                if t.get("text"):
                    blk["text"] = t["text"]
                if cat == "table":
                    blk["markdown"] = t.get("markdown", "")
                blocks.append(blk)
            pages_blocks.append(
                {
                    "page": pidx + 1,
                    "page_w": float(img.width),
                    "page_h": float(img.height),
                    "blocks": blocks,
                }
            )
        return pages_blocks
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


async def analyze_async(
    data: bytes, filename: str, pages: Optional[list] = None
) -> List[Dict[str, Any]]:
    log.info("Calling PPStructureLight analyze_async")
    try:
        clients._mock_counters["pp"] += 1
    except Exception:
        pass
    loop = asyncio.get_event_loop()
    res = await loop.run_in_executor(None, _analyze_sync, data, filename, pages)
    log.info("PPStructureLight analyze_async completed")
    return res
