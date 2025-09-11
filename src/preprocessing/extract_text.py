import pdfplumber

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract raw text from a PDF file using pdfplumber.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        str: Extracted text from the PDF.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
            text += "\n"
    return text.strip()
