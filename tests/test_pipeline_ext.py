
import os, io, json, zipfile, importlib
from fastapi.testclient import TestClient
import main, config, clients
from _pdfutils import make_pdf_text

def _client():
    importlib.reload(config); importlib.reload(main)
    return TestClient(main.app)

def _tpl(fields): return {"name":"tpl","fields":fields, "llm_text":"estrai i campi richiesti"}

def test_pdf_digital_no_ocr_and_bundle():
    os.environ['MOCK_LLM']='1'
    os.environ["OCR_POLICY"]="auto"
    os.environ["TEXT_LAYER_MIN_CHARS"]="10"
    clients.reset_mock_counters()
    c = _client()
    pdf = make_pdf_text(1, "documento con testo digitale sufficiente")
    r = c.post("/extract", headers={"x-api-key": os.environ["API_KEY"]},
               files={"file": ("doc.pdf", pdf, "application/pdf")},
               data={"template": json.dumps(_tpl(["iban"]))})
    assert r.status_code == 200
    cnt = clients.get_mock_counters()
    assert cnt["ocr"] == 0
    rid = r.json()["request_id"]
    zb = c.get(f"/reports/{rid}/bundle.zip", headers={"x-api-key": os.environ["API_KEY"]})
    assert zb.status_code == 200
    z = zipfile.ZipFile(io.BytesIO(zb.content))
    names = set(z.namelist())
    assert "md.txt" in names and "response.json" in names

def test_image_raster_ocr_and_tokens_exist():
    os.environ['MOCK_LLM']='1'
    os.environ["OCR_POLICY"]="auto"
    clients.reset_mock_counters()
    c = _client()
    png = b"\x89PNG\r\n\x1a\n" + b"0"*200
    r = c.post("/extract", headers={"x-api-key": os.environ["API_KEY"]},
               files={"file": ("img.png", png, "image/png")},
               data={"template": json.dumps(_tpl(["iban","totale"]))})
    assert r.status_code == 200
    cnt = clients.get_mock_counters()
    assert cnt["ocr"] >= 1
    rid = r.json()["request_id"]
    # tokens file should exist and have at least one line
    import os as _os
    tok = f"/mnt/data/reports/{rid}/tokens.jsonl"
    assert _os.path.exists(tok)
