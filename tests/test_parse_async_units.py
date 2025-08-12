
import anyio
import parse
from clients import doctr_client as ocr

def test_convert_markdown_async_png():
    data = b"\x89PNG\r\n\x1a\n" + b"x"*64
    md = anyio.run(parse.convert_markdown_async, data, "x.png")
    assert isinstance(md, str)

def test_ocr_async_mock_result(monkeypatch):
    monkeypatch.setenv("MOCK_OCR", "1")
    ocr._DOCTR_INSTANCE = None  # reset to pick up mock
    data = b"\x89PNG\r\n\x1a\n" + b"x"*64
    blocks = anyio.run(ocr.analyze_async, data, "x.png")
    assert isinstance(blocks, list) and len(blocks) >= 1
    pg = blocks[0]
    assert "page" in pg and "blocks" in pg and isinstance(pg["blocks"], list)
