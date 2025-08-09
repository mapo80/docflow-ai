from __future__ import annotations
import re
from typing import List, Dict, Any
def approximate_tokens(text: str) -> int:
    words = len(re.findall(r"\S+", text))
    chars = len(text)
    return int(0.5*words + 0.25*chars/4)

def split_markdown_into_chunks(md: str) -> List[Dict[str, Any]]:
    chunks = []
    pos = 0
    lines = md.splitlines(keepends=True)
    buf = []
    kind = None
    start = 0
    def flush(k):
        nonlocal buf, kind, start, chunks
        if buf:
            txt = ''.join(buf)
            chunks.append({"id": len(chunks), "text": txt, "kind": k or "paragraph", "start": start, "end": start+len(txt)})
            buf, kind = [], None
    for ln in lines:
        if re.match(r"^\s*#", ln):
            flush(kind); kind="heading"; start=pos; buf.append(ln)
        elif ('|' in ln and re.search(r"\|", ln) and re.search(r"[-:]+\|[-:]+", ln)) or (kind=="table" and '|' in ln):
            if kind != "table":
                flush(kind); kind="table"; start=pos
            buf.append(ln)
        elif re.match(r"^\s*([\-*+]\s+|\d+\.\s+)", ln):
            if kind not in ("list", None):
                flush(kind); kind="list"; start=pos
            elif kind is None:
                kind="list"; start=pos
            buf.append(ln)
        elif ln.strip()=="":
            if kind is None: kind="paragraph"; start=pos
            buf.append(ln); flush(kind); kind=None
        else:
            if kind in ("heading","table","list"):
                buf.append(ln)
            else:
                if kind is None:
                    kind="paragraph"; start=pos
                buf.append(ln)
        pos += len(ln)
    flush(kind)
    return chunks

def sniff_tables_from_tokens(tokens: List[Dict[str,Any]], min_rows: int, min_cols: int, tol_x: float) -> bool:
    from collections import defaultdict
    rows_by_page = defaultdict(set)
    xs = []
    for t in tokens:
        page = t.get('page',1)
        line = t.get('line_id', None)
        if line is not None:
            rows_by_page[page].add(line)
        x0,y0,x1,y1 = t.get('bbox', [0,0,0,0])
        cx = (x0 + x1) / 2.0
        xs.append(round(cx, 2))
    row_count = max((len(s) for s in rows_by_page.values()), default=0)
    if row_count < min_rows:
        return False
    xs.sort()
    clusters = 1
    for i in range(1,len(xs)):
        if abs(xs[i]-xs[i-1]) > tol_x:
            clusters += 1
    return clusters >= min_cols
