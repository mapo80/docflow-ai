
import os, json, importlib
from fastapi.testclient import TestClient
import config, main

def test_overlays_multi_page(monkeypatch):
    os.environ['DEBUG_OVERLAY']='1'
    os.environ.setdefault('BACKENDS_MOCK','1')
    importlib.reload(config); importlib.reload(main)
    from clients import ppstructure_client as ppc
    from clients import markitdown_client as mk
    import llm as _llm
    async def fake_pp(data, filename, pages=None):
        return [
            {"page":1,"page_w":600,"page_h":800,"blocks":[
                {"type":"text","text":"Val 1","bbox":[60,60,160,110]},
                {"type":"text","text":"Val 2","bbox":[180,60,260,110]},
            ]},
            {"page":2,"page_w":600,"page_h":800,"blocks":[
                {"type":"cell","text":"Cella","bbox":[80,200,300,240]}
            ]},
        ]
    monkeypatch.setattr(ppc, 'analyze_async', fake_pp)
    async def fake_md(data, filename, mime=None): return ''
    monkeypatch.setattr(mk, 'convert_bytes_to_markdown_async', fake_md)
    async def fake_chat(messages, max_tokens=1024):
        import json as _json
        payload = {"a": {"value": "Val 1", "confidence": 0.9}, "b": {"value": "Val 2", "confidence": 0.9}}
        return {"choices":[{"message":{"content": _json.dumps(payload)}}]}
    monkeypatch.setattr(_llm, 'chat_json_async', fake_chat)
    c = TestClient(main.app)
    png = b'\x89PNG\r\n\x1a\n' + b'0'*128
    tpl = {"name":"t","fields":["a","b"], "llm_text":"estrai"}
    r = c.post('/extract', headers={"x-api-key": os.environ["API_KEY"]},
               files={'file': ('img.png', png, 'image/png')},
               data={'template': json.dumps(tpl)})
    assert r.status_code == 200
    rid = r.json()['request_id']
    # bundle must exist
    z = c.get(f'/reports/{rid}/bundle.zip', headers={"x-api-key": os.environ["API_KEY"]})
    assert z.status_code == 200
