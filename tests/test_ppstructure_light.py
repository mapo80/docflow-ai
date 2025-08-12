import os
from typing import Any
import tempfile

import numpy as np
from PIL import Image

import clients.ppstructure_light as ppl
from tests._pdfutils import make_pdf_text


class DummyOCR:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def ocr(self, img, det=True, rec=False, cls=False):
        # Return one line with a simple square polygon
        return [[[[[1, 1], [11, 1], [11, 11], [1, 11]], None]]]


class DummyPP:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def predict(self, input, **kwargs):
        return [
            {
                "layout_res_list": [
                    {"type": "table", "bbox": [2, 2, 8, 8]},
                ],
                "table_res_list": [
                    {
                        "cell_box_list": [
                            [[2, 2], [4, 2], [4, 4], [2, 4]],
                        ],
                        "cell_text_list": ["foo"],
                        "cell_rc_list": [[0, 0]],
                    }
                ],
            }
        ]


def _make_pdf_path() -> str:
    pdf_bytes = make_pdf_text(pages=1, text="hello")
    fd, path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        f.write(pdf_bytes)
    return path


def test_extract_tokens(monkeypatch):
    path = _make_pdf_path()
    monkeypatch.setattr(ppl, "PaddleOCR", DummyOCR)
    monkeypatch.setattr(ppl, "PPStructureV3", DummyPP)
    monkeypatch.setattr(ppl, "convert_from_path", lambda p, dpi=200: [Image.new("RGB", (10,10), "white")])
    pp = ppl.PPStructureLight()
    tokens = pp.extract_tokens(path)
    cats = {t["category"] for t in tokens}
    assert {"text", "table", "cell"} <= cats
    text_token = next(t for t in tokens if t["category"] == "text")
    assert text_token["bbox"] == [1.0, 1.0, 10.0, 10.0]
    cell_token = next(t for t in tokens if t["category"] == "cell")
    assert cell_token["text"] == "foo"
    assert cell_token["row"] == 0 and cell_token["col"] == 0


def test_extract_tokens_without_cell_text(monkeypatch):
    path = _make_pdf_path()
    monkeypatch.setattr(ppl, "PaddleOCR", DummyOCR)
    monkeypatch.setattr(ppl, "PPStructureV3", DummyPP)
    monkeypatch.setattr(ppl, "convert_from_path", lambda p, dpi=200: [Image.new("RGB", (10,10), "white")])
    pp = ppl.PPStructureLight(include_cell_text=False)
    tokens = pp.extract_tokens(path)
    cell_token = next(t for t in tokens if t["category"] == "cell")
    assert "text" not in cell_token
