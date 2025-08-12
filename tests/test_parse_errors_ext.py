
import anyio, parse

def test_convert_markdown_async_binary_fallback():
    data = b"\x00\x01\x02\xff"*32
    md = anyio.run(parse.convert_markdown_async, data, "blob.bin")
    assert isinstance(md, str)

def test_parse_with_ocr_async_empty_ok(monkeypatch):
    import clients.doctr_client as ocr
    async def fake_ocr(data, filename, pages=None):
        return []
    monkeypatch.setattr(ocr, "analyze_async", fake_ocr)
    blocks = anyio.run(parse.parse_with_ocr_async, b"x", "x.png")
    assert blocks == []
