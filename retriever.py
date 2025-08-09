from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Tuple

from config import (RAG_TOPK, RAG_W_BM25, RAG_W_VEC, RAG_W_ANCHOR, renormalize_weights,
                     EMBEDDING_BACKEND, EMBEDDING_GGUF_PATH, EMBEDDING_POOLING,
                     EMBEDDING_N_THREADS, EMBED_INSTRUCT_QUERY, EMBED_INSTRUCT_DOC,
                     EMBEDDING_MODEL_NAME, EMBEDDING_NORMALIZE)
from logger import get_logger
from config import BACKENDS_MOCK
from clients import *  # unified import
log = get_logger(__name__)

def _tokenize_for_bm25(text: str) -> List[str]:
    import re
    return [t.lower() for t in re.findall(r"\w+", text)]

class EphemeralIndex:
    def __init__(self, chunks: List[Dict[str,Any]], anchors: List[str] | None = None):
        self.chunks = chunks
        self.anchors = [a.lower() for a in (anchors or [])]
        try:
            from rank_bm25 import BM25Okapi
            self.corpus_tokens = [_tokenize_for_bm25(ch['text']) for ch in chunks]
            self.bm25 = BM25Okapi(self.corpus_tokens)
        except Exception:
            # Fallback: simple overlap score
            self.corpus_tokens = [_tokenize_for_bm25(ch['text']) for ch in chunks]
            class _BM25Fallback:
                def __init__(self, corp): self.corp = corp
                def get_scores(self, qtok):
                    scores=[]
                    qs = set(qtok)
                    for doc in self.corp:
                        scores.append(len(qs.intersection(doc)))
                    return np.array(scores, dtype='float32')
            self.bm25 = _BM25Fallback(self.corpus_tokens)
        self._init_vectors()

    def _normalize(self, X):
        if not EMBEDDING_NORMALIZE:
            return X
        denom = np.linalg.norm(X, axis=1, keepdims=True) + 1e-6
        return X / denom

    def _build_faiss(self):
        try:
            import faiss
            d = self.vecs.shape[1]
            self.index = faiss.IndexFlatIP(d)
            self.index.add(self.vecs)
            self.use_faiss = True
        except Exception:
            self.index = None
            self.use_faiss = False

    def _init_vectors(self):
        backend = EMBEDDING_BACKEND.lower().strip()
        if backend == 'gguf':
            self._init_vectors_gguf()
        elif backend == 'st':
            self._init_vectors_st()
        else:
            self._init_vectors_tfidf()

    def _init_vectors_gguf(self):
        try:
            from llama_cpp import Llama
            if BACKENDS_MOCK:
                # Use mock embeddings via clients
                texts = [c['text'] for c in self.chunks]
                V = clients.llm_embed.__wrapped__(texts) if hasattr(clients.llm_embed,'__wrapped__') else None
                if V is None:
                    import anyio
                    V = anyio.run(clients.llm_embed, texts)
                self.vecs = self._normalize(np.array(V, dtype='float32'))
                self.method = 'gguf-mock'
                return
            llm_kwargs = dict(model_path=EMBEDDING_GGUF_PATH, embedding=True, n_threads=EMBEDDING_N_THREADS)
            try:
                llm_kwargs["pooling_type"] = EMBEDDING_POOLING
            except Exception:
                pass
            self._emb_llm = Llama(**llm_kwargs)
            texts = [c['text'] for c in self.chunks]
            if EMBED_INSTRUCT_DOC:
                texts = [f"{EMBED_INSTRUCT_DOC}\n{t}" for t in texts]
            embs = self._emb_llm.create_embedding(input=texts)
            V = [row.get('embedding') for row in embs.get('data', [])]
            self.vecs = self._normalize(np.array(V, dtype='float32'))
            self.method = 'gguf'
        except Exception as e:
            log.warning("GGUF embedding failed (%s). Falling back to Sentence-Transformers.", e)
            try:
                self._init_vectors_st()
            except Exception as e2:
                log.warning("ST fallback failed (%s). Falling back to TF-IDF.", e2)
                self._init_vectors_tfidf()
        self._build_faiss()

    def _init_vectors_st(self):
        from sentence_transformers import SentenceTransformer
        self.st = SentenceTransformer(EMBEDDING_MODEL_NAME)
        texts = [c['text'] for c in self.chunks]
        if EMBED_INSTRUCT_DOC:
            texts = [f"{EMBED_INSTRUCT_DOC}\n{t}" for t in texts]
        V = self.st.encode(texts, normalize_embeddings=bool(EMBEDDING_NORMALIZE), convert_to_numpy=True)
        if not EMBEDDING_NORMALIZE:
            V = np.array(V, dtype='float32')
        self.vecs = V
        self.method = 'st'
        self._build_faiss()

    def _init_vectors_tfidf(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer().fit([c['text'] for c in self.chunks])
        mat = vec.transform([c['text'] for c in self.chunks]).astype('float32')
        V = mat.toarray()
        self.vecs = self._normalize(V)
        self.method = 'tfidf'
        self._build_faiss()

    def _anchor_boost(self, text: str, query: str) -> float:
        t = text.lower()
        boost = 0.0
        for a in self.anchors:
            if a and a in t:
                boost = max(boost, 1.0)
        ql = query.lower()
        if ql in t:
            boost = max(boost, 0.5)
        return boost

    def _embed_query(self, query: str):
        # Prefer actual method used for self.vecs to avoid shape mismatches
        method = getattr(self, 'method', EMBEDDING_BACKEND.lower().strip())
        backend = EMBEDDING_BACKEND.lower().strip()
        if method == 'tfidf':
            from sklearn.feature_extraction.text import TfidfVectorizer
            tfidf = TfidfVectorizer().fit([c['text'] for c in self.chunks])
            qv = tfidf.transform([query]).astype('float32').toarray()
            if EMBEDDING_NORMALIZE:
                denom = np.linalg.norm(qv, axis=1, keepdims=True) + 1e-6
                qv = qv / denom
            return qv
        if method == 'st' and hasattr(self, 'st'):
            q = f"{EMBED_INSTRUCT_QUERY}\n{query}" if EMBED_INSTRUCT_QUERY else query
            v = self.st.encode([q], normalize_embeddings=bool(EMBEDDING_NORMALIZE), convert_to_numpy=True)
            if not EMBEDDING_NORMALIZE:
                v = np.array(v, dtype='float32')
            return v
        # gguf path (mock or real)
        if method.startswith('gguf') and BACKENDS_MOCK:
            q = f"{EMBED_INSTRUCT_QUERY}\n{query}" if EMBED_INSTRUCT_QUERY else query
            V = clients.llm_embed.__wrapped__([q]) if hasattr(clients.llm_embed,'__wrapped__') else None
            if V is None:
                import anyio
                V = anyio.run(clients.llm_embed, [q])
            v = np.array(V[0], dtype='float32')
            if EMBEDDING_NORMALIZE:
                v = v / (np.linalg.norm(v) + 1e-6)
            return v.reshape(1,-1)
        if method == 'gguf' and hasattr(self, '_emb_llm'):
            q = f"{EMBED_INSTRUCT_QUERY}\n{query}" if EMBED_INSTRUCT_QUERY else query
            out = self._emb_llm.create_embedding(input=[q])
            v = np.array(out['data'][0]['embedding'], dtype='float32')
            if EMBEDDING_NORMALIZE:
                v = v / (np.linalg.norm(v) + 1e-6)
            return v.reshape(1,-1)
        # default fallback
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer().fit([c['text'] for c in self.chunks])
        qv = tfidf.transform([query]).astype('float32').toarray()
        if EMBEDDING_NORMALIZE:
            denom = np.linalg.norm(qv, axis=1, keepdims=True) + 1e-6
            qv = qv / denom
        return qv
    def search(self, query: str, topk: int | None = None) -> List[Tuple[int, float]]:
        topk = topk or RAG_TOPK
        qtok = _tokenize_for_bm25(query)
        bm25_scores = self.bm25.get_scores(qtok)
        qvec = self._embed_query(query)
        vec_scores = np.zeros(len(self.chunks), dtype=np.float32)
        if getattr(self, 'use_faiss', False):
            import faiss
            D, I = self.index.search(qvec.astype(np.float32), k=min(topk*4, len(self.chunks)))
            for d, i in zip(D[0], I[0]):
                if i >= 0:
                    vec_scores[i] = max(vec_scores[i], float(d))
        else:
            try:
                sims = (self.vecs @ qvec.reshape(-1))  # cosine if normalized
                order = np.argsort(-sims)[:min(topk*4, len(self.chunks))]
                for i in order:
                    vec_scores[i] = float(sims[i])
            except Exception:
                # shape mismatch or other vector error -> ignore vector scores
                pass
        w_bm25, w_vec, w_anchor = renormalize_weights(RAG_W_BM25, RAG_W_VEC, RAG_W_ANCHOR)
        combined = []
        for idx, ch in enumerate(self.chunks):
            anchor = self._anchor_boost(ch['text'], query)
            score = w_bm25*float(bm25_scores[idx]) + w_vec*float(vec_scores[idx]) + w_anchor*anchor
            combined.append((idx, score))
        combined.sort(key=lambda x: (-x[1], self.chunks[x[0]]['start']))
        return combined[:topk]
