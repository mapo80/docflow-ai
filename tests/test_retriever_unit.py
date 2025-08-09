from retriever import EphemeralIndex
import indexer

def test_anchor_boost_changes_ranking():
    md = """
# Fattura
IBAN: IT00X0000000000000000000000

Totale: 123,45 EUR
Note varie...
"""
    chunks = indexer.split_markdown_into_chunks(md)
    idx = EphemeralIndex(chunks, anchors=["iban"])
    hits = idx.search("iban", topk=3)
    # The top chunk should contain the literal "IBAN"
    top_chunk_id = hits[0][0]
    assert "IBAN" in chunks[top_chunk_id]["text"]
