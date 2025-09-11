import re
from PyPDF2 import PdfReader
from .paragraph_splitter import split_into_paragraphs

def preprocess_material(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Use our richer paragraph splitter (3–5 sentences per paragraph)
    paragraphs = split_into_paragraphs(text)

    return paragraphs
