import os
from typing import List, Optional, Tuple

# We try PyPDF2 first (very common). If you prefer pdfminer.six, swap below.
try:
    import PyPDF2  # pip install PyPDF2
    _PDF_BACKEND = "pypdf2"
except Exception:
    PyPDF2 = None
    _PDF_BACKEND = "none"


def read_pdf_text_pages(path: str) -> List[str]:
    """
    Return a list of strings, one per PDF page.
    Uses PyPDF2 text extraction (works well for text-based PDFs).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PDF not found: {path}")

    if _PDF_BACKEND != "pypdf2":
        raise RuntimeError(
            "PyPDF2 not available. Install with `pip install PyPDF2`, "
            "or replace with pdfminer.six extraction."
        )

    pages: List[str] = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception as e:
                text = ""
                print(f"[WARN] Failed to extract text from page {i+1}: {e}")
            # normalise whitespace per page
            text = " ".join(text.split())
            pages.append(text)
    return pages


def save_pages_to_txt(pages: List[str], out_dir: str, prefix: str = "page") -> List[str]:
    """
    Save each page to a numbered .txt file. Returns the file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []
    for idx, content in enumerate(pages, start=1):
        fp = os.path.join(out_dir, f"{prefix}_{idx:03d}.txt")
        with open(fp, "w", encoding="utf-8") as f:
            f.write(content)
        paths.append(fp)
    return paths

