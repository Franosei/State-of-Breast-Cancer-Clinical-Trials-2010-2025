# taug_extraction/pdf_splitter.py
"""
Split the TAUG-BrCa PDF into page-level text files.

Default input:
  ./TAUG-BrCa/TAUG-BrCa v1.0.pdf

Default output:
  ./data/interim/taug_pages/page_001.txt, page_002.txt, ...

Usage (from project root):
  python -m taug_extraction.pdf_splitter
  python -m taug_extraction.pdf_splitter --pdf "TAUG-BrCa/TAUG-BrCa v1.0.pdf"
  python -m taug_extraction.pdf_splitter --pdf "TAUG-BrCa/TAUG-BrCa v1.0.pdf" --out-dir "data/interim/taug_pages" --overwrite
"""

import os
import argparse
from typing import List

from utils.io import ensure_dirs, INTERIM_DIR, append_run_log
from utils.pdf_utils import read_pdf_text_pages, save_pages_to_txt


def split_pdf_to_pages(
    pdf_path: str,
    out_dir: str,
    overwrite: bool = False,
) -> List[str]:
    """
    Extract text from each page of `pdf_path` and save to `out_dir` as page_XXX.txt.
    Returns the list of written file paths.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Prepare output directory
    os.makedirs(out_dir, exist_ok=True)

    # If directory already has page files and overwrite is False, skip work
    existing = [f for f in os.listdir(out_dir) if f.lower().endswith(".txt")]
    if existing and not overwrite:
        print(f"[INFO] Output directory already contains {len(existing)} .txt files. Use --overwrite to regenerate.")
        return [os.path.join(out_dir, f) for f in sorted(existing)]

    print(f"[INFO] Reading PDF: {pdf_path}")
    pages = read_pdf_text_pages(pdf_path)
    print(f"[INFO] Extracted {len(pages)} page(s) of text")

    print(f"[INFO] Writing page texts to: {out_dir}")
    written_paths = save_pages_to_txt(pages, out_dir, prefix="page")
    print(f"[INFO] Wrote {len(written_paths)} files")

    append_run_log(
        event="taug_pdf_split",
        meta={"pdf": pdf_path, "out_dir": out_dir, "pages": len(written_paths), "overwrite": overwrite},
    )
    return written_paths


def _default_pdf_path() -> str:
    # default source location alongside your repo
    return os.path.join(os.getcwd(), "TAUG-BrCa", "TAUG-BrCa v1.0.pdf")


def _default_out_dir() -> str:
    # write into the standard interim folder
    return os.path.join(INTERIM_DIR, "taug_pages")


def main():
    ensure_dirs()  # make sure ./data/* exists

    parser = argparse.ArgumentParser(description="Split TAUG-BrCa PDF into page text files.")
    parser.add_argument(
        "--pdf",
        type=str,
        default=_default_pdf_path(),
        help="Path to the TAUG-BrCa PDF (default: ./TAUG-BrCa/TAUG-BrCa v1.0.pdf)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=_default_out_dir(),
        help="Directory to write page text files (default: ./data/interim/taug_pages)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing page text files if present",
    )
    args = parser.parse_args()

    split_pdf_to_pages(pdf_path=args.pdf, out_dir=args.out_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
