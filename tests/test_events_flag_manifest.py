
import os, json, importlib
from fastapi.testclient import TestClient
import config, main
from _pdfutils import make_pdf_text

def test_events_flag_and_manifest():
    os.environ['EMIT_EVENTS']='1'
    os.environ.setdefault('BACKENDS_MOCK','1')
    importlib.reload(config); importlib.reload(main)
    c = TestClient(main.app)
    pdf = make_pdf_text(1, "testo per manifest")
    tpl = {"name":"t","fields":["x","y"],"llm_text":"estrai"}
    r = c.post('/extract', headers={"x-api-key": os.environ["API_KEY"]},
               files={'file': ('a.pdf', pdf, 'application/pdf')},
               data={'template': json.dumps(tpl)})
    assert r.status_code == 200
    rep = c.get(f"/reports/{r.json()['request_id']}", headers={"x-api-key": os.environ["API_KEY"]}).json()
    assert rep['manifest'].get('llm_context_mode') in ('single_pass','rag_field_wise')
