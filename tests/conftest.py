import os, pytest

os.environ.setdefault("API_KEY","test-key")
os.environ.setdefault("MOCK_LLM","1")
os.environ.setdefault("MOCK_OCR","1")
os.environ.setdefault("REPORTS_DIR","/mnt/data/reports")
os.environ.setdefault("LOG_LEVEL","DEBUG")
os.environ.setdefault('BACKENDS_MOCK','1')
os.environ.setdefault('TEXT_LAYER_MIN_CHARS','999999')
import clients

if os.environ.get('BACKENDS_MOCK','1') == '1':
    def _fake_embed(texts):
        vecs = []
        for t in texts:
            tl = t.lower()
            vecs.append([
                float(tl.count('iban')),
                float(tl.count('codice')),
                float(len(tl)),
            ])
        return vecs
    clients.llm_embed = _fake_embed  # type: ignore
