from __future__ import annotations
from typing import List, Dict, Any, Tuple
import re
def _norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+"," ", s)
    return s
def align_value_to_tokens(value: str, chunk_tokens: List[Dict[str,Any]]) -> Tuple[List[int], float]:
    v = _norm(value)
    if not v: return ([], 0.0)
    toks = [_norm(t.get('text','')) for t in chunk_tokens]
    concat = ' '.join(toks)
    start = concat.find(v)
    if start >= 0:
        token_indices = []
        pos = 0
        for ti, t in enumerate(toks):
            if not t: continue
            t_start, t_end = pos, pos+len(t)
            v_start, v_end = start, start+len(v)
            if max(t_start, v_start) < min(t_end, v_end):
                token_indices.append(ti)
            pos = t_end + 1
        cov = len(v)/max(1,len(v))
        return token_indices, cov
    token_indices = [i for i,t in enumerate(toks) if t and t in v]
    cov = sum(len(t) for t in toks if t in v)/max(1,len(v))
    return token_indices, cov
