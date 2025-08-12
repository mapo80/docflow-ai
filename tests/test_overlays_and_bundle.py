
import os, json, importlib
from fastapi.testclient import TestClient
import config, main

def _client():
    os.environ.setdefault('BACKENDS_MOCK','1')
    importlib.reload(config); importlib.reload(main)
    return TestClient(main.app)

def test_overlays_bundle(monkeypatch):
    os.environ['DEBUG_OVERLAY']='1'
    from clients import ppstructure_client as ppc
    from clients import markitdown_client as mk
    import clients.llm_local as _llm

    async def fake_pp(data, filename, pages=None):
        return [{"page":1,"page_w":600,"page_h":800,"blocks":[{"type":"text","text":"Numero: 12345","bbox":[60,60,260,110]}]}]
    monkeypatch.setattr(ppc, 'analyze_async', fake_pp)
    async def fake_md(data, filename, mime=None): return ''
    monkeypatch.setattr(mk, 'convert_bytes_to_markdown_async', fake_md)
    def fake_chat(fields, llm_text, context):
        return {"numero": {"value": "12345", "confidence": 0.9}}
    monkeypatch.setattr(_llm, 'chat_json', fake_chat, raising=True)
    png = b'\x89PNG\r\n\x1a\n' + b'0'*10
    tpl = {"name":"t","fields":["numero"], "llm_text":"estrai numero"}
    c = _client()
    r = c.post('/extract', headers={"x-api-key": os.environ["API_KEY"]}, files={'file': ('img.png', png, 'image/png')}, data={'template': json.dumps(tpl)})
    assert r.status_code == 200
    rid = r.json()['request_id']
    z = c.get(f'/reports/{rid}/bundle.zip', headers={"x-api-key": os.environ["API_KEY"]})
    assert z.status_code == 200
    assert z.headers.get('content-type') == 'application/zip'
