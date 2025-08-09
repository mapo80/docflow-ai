
import anyio, parse

def test_convert_markdown_async_binary_fallback():
    data = b"\x00\x01\x02\xff"*32
    md = anyio.run(parse.convert_markdown_async, data, "blob.bin")
    assert isinstance(md, str)

def test_parse_with_ppstructure_async_empty_ok(monkeypatch):
    import clients.ppstructure_client as ppc
    async def fake_pp(data, filename, pages=None):
        return []
    monkeypatch.setattr(ppc, "analyze_async", fake_pp)
    blocks = anyio.run(parse.parse_with_ppstructure_async, b"x", "x.png")
    assert blocks == []
