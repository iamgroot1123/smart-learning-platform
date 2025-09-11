from .extract_text import extract_text_from_pdf
from .paragraph_splitter import split_into_paragraphs

def preprocess_material(pdf_path, backend="pypdf2"):
    # Extract using selected backend
    text = extract_text_from_pdf(pdf_path, backend=backend)

    # Split into richer paragraphs
    paragraphs = split_into_paragraphs(text)

    return paragraphs
