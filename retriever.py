# retriever.py — in-memory vector index using local embeddings
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import numpy as np
from clients.embeddings_local import embed_texts

class EphemeralIndex:
    def __init__(self, chunks: List[Dict[str, Any]], anchors: List[str] | None = None):
        self.chunks = chunks
        self.vecs = embed_texts([c["text"] for c in chunks])

    def search(self, query: str, topk: int = 5) -> List[Tuple[int, float]]:
        qv = embed_texts([query])[0]
        sims = []
        for i, v in enumerate(self.vecs):
            s = _cosine(qv, v)
            sims.append((i, float(s)))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:topk]

def _cosine(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))
