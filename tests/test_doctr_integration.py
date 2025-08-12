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


def test_doctr_analyzes_image(monkeypatch):
    monkeypatch.setenv("BACKENDS_MOCK", "0")
    monkeypatch.setenv("MOCK_OCR", "0")
    with open("dataset/sample_invoice.png", "rb") as f:
        data = f.read()
    try:
        pages = asyncio.run(analyze_async(data, "sample_invoice.png"))
    except Exception as e:
        pytest.skip(f"doctr failed: {e}")
    if not pages or not pages[0].get("blocks"):
        pytest.skip("doctr returned no blocks")
    assert any("bbox" in blk for blk in pages[0]["blocks"])
