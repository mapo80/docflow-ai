
import anyio, parse

def test_convert_markdown_async_corrupted_pdf_header():
    # Looks like PDF but corrupted -> should not crash, returns some string
    data = b"%PDF-\x00\x01corrupted"
    md = anyio.run(parse.convert_markdown_async, data, "bad.pdf")
    assert isinstance(md, str)

def test_convert_markdown_async_unknown_binary_to_text():
    data = b"\x00\xff\x10ABC\x11DEF" * 4
    md = anyio.run(parse.convert_markdown_async, data, "blob.bin")
    assert isinstance(md, str)

def test_parse_with_ocr_async_returns_empty(monkeypatch):
    import clients.doctr_client as ocr
    async def fake(data, filename, pages=None): return []
    monkeypatch.setattr(ocr, "analyze_async", fake)
    blocks = anyio.run(parse.parse_with_ocr_async, b"x", "x.png")
    assert blocks == []
