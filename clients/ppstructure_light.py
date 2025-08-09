# clients/ppstructure_light.py
from __future__ import annotations
import os
from typing import List, Dict, Any, Tuple
import numpy as np
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
try:
    from paddleocr import PPStructureV3
except Exception:
    PPStructureV3 = None  # type: ignore

def _poly_to_xywh(poly) -> List[float]:
    arr = np.asarray(poly, dtype=float)
    xs, ys = arr[:,0], arr[:,1]
    x, y = float(xs.min()), float(ys.min())
    w, h = float(xs.max() - x), float(ys.max() - y)
    return [x, y, max(0.0, w), max(0.0, h)]

class PPStructureLight:
    def __init__(self, use_gpu: bool = False, pdf_dpi: int | None = None, include_cell_text: bool = True):
        self.pdf_dpi = int(os.getenv("PDF_DPI", "200") if pdf_dpi is None else pdf_dpi)
        self.include_cell_text = include_cell_text
        self.det = PaddleOCR(use_angle_cls=False, use_gpu=use_gpu, lang="it", show_log=False)
        if PPStructureV3 is None:
            raise RuntimeError("PPStructureV3 unavailable: please install paddleocr>=2.7.0")
        self.pp = PPStructureV3(layout_detection_model_name="PicoDet_layout_1x_table", layout_detection_model_dir=None)

    def extract_tokens(self, path: str) -> List[Dict[str, Any]]:
        pages = self._load_pages(path)
        tokens: List[Dict[str, Any]] = []
        for pidx, img in pages:
            np_img = np.array(img)
            # TEXT (det only)
            det_res = self.det.ocr(np_img, det=True, rec=False, cls=False)
            lines = det_res[0] if (isinstance(det_res, list) and len(det_res) > 0) else []
            for item in lines:
                poly = item[0] if isinstance(item, (list, tuple)) and len(item) > 0 else None
                if poly is None:
                    continue
                tokens.append({"category":"text", "bbox": _poly_to_xywh(poly), "page_index": pidx})
            # TABLE + CELLS
            tables, cell_tokens = self._table_regions_and_cells(np_img, pidx)
            for bb in tables:
                tokens.append({"category":"table", "bbox": bb, "page_index": pidx})
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
                x0,y0,w_or_x1,h_or_y1 = bb
                try:
                    if float(w_or_x1) > float(x0) and float(h_or_y1) > float(y0):
                        return [float(x0), float(y0), float(w_or_x1)-float(x0), float(h_or_y1)-float(y0)]
                except Exception:
                    pass
                return [float(x0), float(y0), float(w_or_x1), float(h_or_y1)]
            return None

        for cand in layout_lists:
            for obj in cand:
                if not isinstance(obj, dict):
                    continue
                lab = (obj.get("type") or obj.get("cls") or obj.get("label") or "").strip().lower()
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

        table_res_list = getattr(r, "table_res_list", None) if not isinstance(r, dict) else r.get("table_res_list")
        if isinstance(table_res_list, list):
            for t in table_res_list:
                cell_boxes = t.get("cell_box_list") or t.get("cell_boxes") or []
                cell_texts = t.get("cell_text_list") or t.get("cell_texts") or []
                cell_rc = t.get("cell_rc_list") or []
                for idx, cb in enumerate(cell_boxes):
                    xywh = _poly_to_xywh(cb) if isinstance(cb, (list, tuple)) else None
                    if not xywh:
                        continue
                    tok = {"category":"cell", "bbox": xywh, "page_index": pidx}
                    if self.include_cell_text:
                        txt = ""
                        if isinstance(cell_texts, list) and idx < len(cell_texts):
                            txt = str(cell_texts[idx] or "")
                        tok["text"] = txt
                    if isinstance(cell_rc, list) and idx < len(cell_rc) and isinstance(cell_rc[idx], (list, tuple)) and len(cell_rc[idx]) >= 2:
                        tok["row"], tok["col"] = int(cell_rc[idx][0]), int(cell_rc[idx][1])
                    cell_tokens.append(tok)

        return tables_xywh, cell_tokens
