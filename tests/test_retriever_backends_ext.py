
import os, numpy as np
from retriever import EphemeralIndex
import clients
from clients import *  # unified import

def test_tfidf_backend_default():
    os.environ["EMBEDDING_BACKEND"]="tfidf"
    chunks=[{"id":"a","text":"uno due tre","start":0},{"id":"b","text":"due tre quattro","start":100}]
    idx = EphemeralIndex(chunks)
    res = idx.search("tre", topk=2)
    assert len(res) == 2

def test_backend_st_and_gguf_monkeypatched(monkeypatch):
    def fake_embed(txts):
        return np.array([[len(t)%5, (len(t)//5)%7, 1.0] for t in txts], dtype="float32")
    monkeypatch.setattr(clients, "llm_embed", lambda texts: fake_embed(texts), raising=True)

    os.environ["EMBEDDING_BACKEND"]="st"
    idx = EphemeralIndex([{"id":"a","text":"uno due tre","start":0},{"id":"b","text":"molto molto lungo testo","start":50}])
    res = idx.search("xxx", topk=2)
    assert len(res) == 2

    os.environ["EMBEDDING_BACKEND"]="gguf"
    idx2 = EphemeralIndex([{"id":"x","text":"aaa bbb","start":0},{"id":"y","text":"ccc ddd eee fff","start":20}])
    res2 = idx2.search("ddd", topk=2)
    assert len(res2) == 2
