
import os, json, importlib
from fastapi.testclient import TestClient
import config, main
from _pdfutils import make_pdf_text

def test_reports_unauthorized_and_404():
    importlib.reload(config); importlib.reload(main)
    c = TestClient(main.app)
    # Unauthorized
    r = c.get('/reports/some-id/bundle.zip')
    assert r.status_code == 401
    # Authorized but 404
    r2 = c.get('/reports/missing-id', headers={"x-api-key": os.environ["API_KEY"]})
    assert r2.status_code in (400,404)
