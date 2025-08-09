
import anyio, parse

def test_build_markdown_from_pp_pages_and_cells():
    pages = [
        {
            "page": 1, "page_w": 600, "page_h": 800,
            "blocks": [
                {"type": "text", "text": "Riga 1", "bbox": [10,10,200,30]},
                {"type": "cell", "text": "A1", "bbox": [20,40,120,70]},
                {"type": "cell", "text": "B1", "bbox": [130,40,230,70]},
            ],
        },
        {
            "page": 2, "page_w": 600, "page_h": 800,
            "blocks": [
                {"type": "text", "text": "Pagina 2", "bbox": [10,10,200,30]}
            ],
        },
    ]
    md = parse.build_markdown_from_pp(pages)
    assert "Riga 1" in md and "Pagina 2" in md and "A1" in md and "B1" in md

def test_convert_markdown_async_pdf_exception_path(monkeypatch):
    # Force fitz to raise to hit the fallback branch
    import parse as _p
    class DummyDoc:
        def __iter__(self): return iter([])
    def fake_open(stream, filetype): raise RuntimeError("boom")
    monkeypatch.setattr(_p.fitz, "open", fake_open, raising=True)
    out = anyio.run(_p.convert_markdown_async, b"%PDF-XXX", "x.pdf")
    assert isinstance(out, str)
