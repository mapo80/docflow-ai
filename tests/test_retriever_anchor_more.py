
from retriever import EphemeralIndex

def test_query_bias_towards_anchor_word():
    chunks = [
        {"id":"a","text":"contratto pagamento banca iban abc","start":0},
        {"id":"b","text":"note varie e descrizioni","start":100},
        {"id":"c","text":"iban codice e pagamenti","start":200},
    ]
    idx = EphemeralIndex(chunks)
    res_plain = idx.search("codice", topk=3)
    res_with_anchor = idx.search("iban codice", topk=3)
    top_plain = res_plain[0][0] if isinstance(res_plain[0], tuple) else res_plain[0].get("id")
    top_anchor = res_with_anchor[0][0] if isinstance(res_with_anchor[0], tuple) else res_with_anchor[0].get("id")
    assert top_anchor in ('a','c', 0, 2)
