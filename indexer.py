# indexer.py — simple markdown splitter + token estimate
from __future__ import annotations
from typing import List, Dict, Any
import re

def split_markdown_into_chunks(md: str, max_chars: int = 1200) -> List[Dict[str, Any]]:
    parts: List[str] = re.split(r"\n{2,}", md or "")
    chunks: List[Dict[str, Any]] = []
    buf = ""
    for p in parts:
        if len(buf) + len(p) + 2 > max_chars and buf:
            chunks.append({"text": buf, "kind": "para"})
            buf = ""
        buf += (("\n\n" if buf else "") + p)
    if buf:
        chunks.append({"text": buf, "kind": "para"})
    return chunks

def approximate_tokens(md: str) -> int:
    return max(1, int(len(md) / 4))
