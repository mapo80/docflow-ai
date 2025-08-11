import asyncio
from typing import Optional

try:
    # pacchetto ufficiale
    from markitdown import MarkItDown
except ImportError as e:
    raise ImportError(
        "Il pacchetto 'markitdown' non è installato. "
        "Esegui: pip install markitdown"
    ) from e


async def convert_bytes_to_markdown_async(content: bytes, mime_type: Optional[str] = None) -> str:
    """
    Converte bytes (PDF/immagine/altro) in markdown usando MarkItDown.
    È eseguito in thread pool per non bloccare l'event loop.
    """
    md = MarkItDown()

    def _convert():
        # MarkItDown accetta bytes o path; qui passiamo i bytes
        # Alcune versioni usano .convert(content, mime_type=...), altre .convert(...)
        try:
            res = md.convert(content, mime_type=mime_type)
        except TypeError:
            # fallback per versioni dove l'argomento si chiama diversamente o non è supportato
            res = md.convert(content)
        # .text_content è la stringa markdown
        return getattr(res, "text_content", str(res))

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _convert)
