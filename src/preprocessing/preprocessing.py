
from typing import List
from .extract_text import extract_text_from_pdf
from .paragraph_splitter import split_into_paragraphs

def preprocess_material(pdf_path, backend="pypdf2"):
    text_or_paragraphs = extract_text_from_pdf(pdf_path, backend=backend)

    if backend == "pypdf2":
        # Split big string into paragraphs
        paragraphs = split_into_paragraphs(text_or_paragraphs)
    else:
        # Already a list of paragraphs
        paragraphs = text_or_paragraphs

    return paragraphs
