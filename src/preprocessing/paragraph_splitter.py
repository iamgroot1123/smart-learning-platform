import re

def split_into_paragraphs(text: str, min_sentences: int = 3, max_sentences: int = 5) -> list:
    """
    Split text into richer paragraphs of 3-5 sentences each.
    Keeps enough context for question generation.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    paragraphs, current = [], []

    for sent in sentences:
        if sent:
            current.append(sent)
        # Flush paragraph if max length reached
        if len(current) >= max_sentences:
            paragraphs.append(" ".join(current))
            current = []
    
    # Add leftover sentences
    if current:
        if len(current) >= min_sentences or not paragraphs:
            paragraphs.append(" ".join(current))

    return paragraphs
