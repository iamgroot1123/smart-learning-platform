# src/preprocessing/extract_text.py

import re
from PyPDF2 import PdfReader
import fitz  # PyMuPDF


def extract_with_pypdf2(pdf_path: str) -> str:
    """Extract plain text using PyPDF2 (simple backend)."""
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_with_pymupdf(pdf_path: str) -> str:
    """Extract richer content (text + structured tables) using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text_chunks = []

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks")

        for b in blocks:
            chunk = b[4].strip()

            # Skip junk: empty, mostly dots, or decorative separators
            if not chunk:
                continue
            if re.fullmatch(r"[.\s·•]+", chunk):
                continue
            if chunk.count(".") / max(len(chunk), 1) > 0.5:
                continue

            # Detect table-like content
            if "\t" in chunk or re.search(r"\d+\s+\d+", chunk):
                cleaned = re.sub(r"\s{2,}", " | ", chunk)  # replace big gaps with "|"
                text_chunks.append(f"[Table_Page{page_num}]\n{cleaned}\n[/Table_Page{page_num}]")
            else:
                text_chunks.append(chunk)

        # Add image placeholders (optional, can skip if not needed yet)
        images = page.get_images(full=True)
        for img_index, _ in enumerate(images, start=1):
            text_chunks.append(f"[Image_{page_num}_{img_index}]")

    # Join with line breaks for readability
    text = "\n".join(text_chunks)
    text = re.sub(r'\n{2,}', '\n\n', text).strip()
    return text


def extract_text_from_pdf(pdf_path: str, backend: str = "pypdf2") -> str:
    """
    Extract text from PDF using chosen backend.
    backend = "pypdf2" (simple) or "pymupdf" (rich).
    """
    if backend == "pypdf2":
        return extract_with_pypdf2(pdf_path)
    elif backend == "pymupdf":
        return extract_with_pymupdf(pdf_path)
    else:
        raise ValueError("Invalid backend. Choose 'pypdf2' or 'pymupdf'.")
