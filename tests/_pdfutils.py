import fitz

def make_pdf_text(pages: int = 1, text: str = "IBAN: IT60X0542811101000000123456\nTotale: 123,45 EUR") -> bytes:
    doc = fitz.open()
    for i in range(pages):
        p = doc.new_page()
        p.insert_text((72,72), text)
    return doc.tobytes()
