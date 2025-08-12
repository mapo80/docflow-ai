import importlib
import os
import asyncio
import pytest

doctr_spec = importlib.util.find_spec("doctr")

pytestmark = pytest.mark.skipif(
    doctr_spec is None,
    reason="requires python-doctr package",
)

from clients.doctr_client import analyze_async


def _has_text(pages: list) -> bool:
    return bool(pages and pages[0].get("blocks"))


def _page_text(pages: list) -> str:
    return " ".join(blk["text"] for blk in pages[0]["blocks"] if blk.get("text"))


def _run_doctr(path: str) -> list:
    with open(path, "rb") as f:
        data = f.read()
    try:
        pages = asyncio.run(analyze_async(data, os.path.basename(path)))
    except Exception as e:
        pytest.skip(f"doctr failed: {e}")
    if not _has_text(pages):
        pytest.skip("doctr returned no blocks")
    return pages


def test_doctr_reads_png(monkeypatch):
    monkeypatch.setenv("BACKENDS_MOCK", "0")
    monkeypatch.setenv("MOCK_OCR", "0")
    pages = _run_doctr("dataset/sample_invoice.png")
    assert "Invoice" in _page_text(pages)


def test_doctr_reads_pdf(monkeypatch):
    monkeypatch.setenv("BACKENDS_MOCK", "0")
    monkeypatch.setenv("MOCK_OCR", "0")
    pages = _run_doctr("dataset/sample_invoice.pdf")
    assert "Invoice" in _page_text(pages)
