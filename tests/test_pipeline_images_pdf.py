import os, json, base64, io
from fastapi.testclient import TestClient
import main
from _pdfutils import make_pdf_text
import clients
from clients import *  # unified import

client = TestClient(main.app)
API = {"x-api-key": os.environ["API_KEY"]}

def _tpl(fields):
    return {"name":"tpl","fields":fields, "llm_text":"trova i campi richiesti"}

def test_pdf_digital_no_pp_on_auto_when_no_tables():
    os.environ["TEXT_LAYER_MIN_CHARS"] = "2"
    os.environ["PPSTRUCT_POLICY"] = "auto"
    os.environ["ALLOW_PP_ON_DIGITAL"] = "0"
    clients.reset_mock_counters()

    pdf = make_pdf_text(1, "Questo è un PDF digitale con un IBAN: IT60X0542811101000000123456")
    r = client.post("/extract", headers=API, files={"file": ("doc.pdf", pdf, "application/pdf")}, data={"template": json.dumps(_tpl(["iban"]))})
    assert r.status_code == 200
    cnt = clients.get_mock_counters()
    # No PP calls because digital and no table sniff in MD
    assert cnt["pp"] == 0

def test_pdf_raster_calls_pp_once():
    os.environ["TEXT_LAYER_MIN_CHARS"] = "10000"  # force raster classification
    os.environ["PPSTRUCT_POLICY"] = "auto"
    clients.reset_mock_counters()
    pdf = make_pdf_text(1, "")  # empty text -> treated as raster by threshold
    r = client.post("/extract", headers=API, files={"file": ("scan.pdf", pdf, "application/pdf")}, data={"template": json.dumps(_tpl(["iban","totale"]))})
    assert r.status_code == 200
    cnt = clients.get_mock_counters()
    assert cnt["pp"] == 1  # one page -> one PP call

def test_image_goes_to_pp_and_tokens_exist():
    clients.reset_mock_counters()
    # minimal PNG header + bytes (content doesn't matter in mock)
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"0"*100
    r = client.post("/extract", headers=API, files={"file": ("img.png", png_bytes, "image/png")}, data={"template": json.dumps(_tpl(["iban","totale"]))})
    assert r.status_code == 200
    cnt = clients.get_mock_counters()
    assert cnt["pp"] >= 1

def test_rag_singlepass_switch():
    # Small doc => single-pass
    os.environ["LLM_N_CTX"]="8192"
    os.environ["RAG_MIN_SEGMENTS"]="100"
    clients.reset_mock_counters()
    pdf = make_pdf_text(1, "Piccolo doc")
    r = client.post("/extract", headers=API, files={"file": ("small.pdf", pdf, "application/pdf")}, data={"template": json.dumps(_tpl(["iban"]))})
    assert r.status_code == 200
    js = r.json()
    assert js["template"] == "tpl"
    # Large doc => force RAG by shrinking ctx and segments
    os.environ["LLM_N_CTX"]="512"
    os.environ["RAG_MIN_SEGMENTS"]="1"
    r2 = client.post("/extract", headers=API, files={"file": ("large.pdf", pdf*1000, "application/pdf")}, data={"template": json.dumps(_tpl(["iban"]))})
    assert r2.status_code == 200

def test_report_bundle_available():
    pdf = make_pdf_text(1, "Doc per report")
    r = client.post("/extract", headers=API, files={"file": ("rep.pdf", pdf, "application/pdf")}, data={"template": json.dumps(_tpl(["iban"]))})
    assert r.status_code == 200
    rid = r.json()["request_id"]
    rj = client.get(f"/reports/{rid}", headers=API)
    assert rj.status_code == 200
    rz = client.get(f"/reports/{rid}/bundle.zip", headers=API)
    assert rz.status_code == 200
