import os
import shutil
import tempfile
from pathlib import Path
from typing import List

from PIL import Image
from pdf2image import convert_from_path
import pytesseract

def page_needs_ocr(text: str, threshold_chars: int = 40) -> bool:
    return len(text.strip()) < threshold_chars

def ocr_with_pytesseract(pdf_path: Path, dpi: int = 300, lang: str = "eng") -> List[str]:
    pages_text: List[str] = []
    images = convert_from_path(str(pdf_path), dpi=dpi)
    for img in images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        txt = pytesseract.image_to_string(img, lang=lang)
        pages_text.append(txt or "")
    return pages_text

def ocrmypdf_available() -> bool:
    return shutil.which("ocrmypdf") is not None

def ocr_with_ocrmypdf(pdf_path: Path, lang: str = "eng") -> Path:
    tmpdir = tempfile.mkdtemp(prefix="ocrpdf_")
    out_pdf = Path(tmpdir) / (pdf_path.stem + ".searchable.pdf")
    cmd = ["ocrmypdf", "--force-ocr", "--optimize", "3", "--language", lang, str(pdf_path), str(out_pdf)]
    import subprocess
    subprocess.run(cmd, check=True)
    return out_pdf
