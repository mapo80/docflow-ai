from __future__ import annotations
from typing import Dict, List, Any
from PIL import Image, ImageDraw
import os, io, fitz

def save_overlays(pdf_bytes: bytes, matches_per_page: Dict[int, List[dict]], out_dir: str, filename: str) -> list:
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    # Render pages as images using PyMuPDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for pno, page in enumerate(doc, start=1):
        if pno not in matches_per_page: continue
        pix = page.get_pixmap(dpi=144)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        draw = ImageDraw.Draw(img)
        # draw boxes (bbox_norm in 0..1)
        for m in matches_per_page[pno]:
            x0,y0,x1,y1 = m["bbox_norm"]
            x0*=img.width; x1*=img.width; y0*=img.height; y1*=img.height
            draw.rectangle([x0,y0,x1,y1], outline="red", width=3)
            draw.text((x0,y0-12), m.get("label","field"), fill="red")
        out_path = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(filename))[0]}_p{pno}.png")
        img.save(out_path)
        saved.append(out_path)
    return saved
