
import anyio
import parse
from clients import ppstructure_client as ppc

def test_convert_markdown_async_png():
    data = b"\x89PNG\r\n\x1a\n" + b"x"*64
    md = anyio.run(parse.convert_markdown_async, data, "x.png")
    assert isinstance(md, str)

def test_ppstructure_async_mock_result():
    data = b"\x89PNG\r\n\x1a\n" + b"x"*64
    blocks = anyio.run(ppc.analyze_async, data, "x.png")
    assert isinstance(blocks, list) and len(blocks) >= 1
    pg = blocks[0]
    assert "page" in pg and "blocks" in pg and isinstance(pg["blocks"], list)
