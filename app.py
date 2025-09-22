import streamlit as st
import tempfile
import os
import re
from src.preprocessing.extract_text import extract_text_from_pdf

# --------- Sentence Splitting + Sliding Window ---------
_SENT_SPLIT_RE = re.compile(r'(?<=[\.\?\!])\s+(?=[A-Z0-9\"\'\(\[])')

def split_into_sentences(text: str):
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return sents if sents else [text.strip()]

def sliding_window_chunk_sentences(text: str, chunk_size=6, overlap=2):
    sents = split_into_sentences(text)
    if len(sents) <= chunk_size:
        return [" ".join(sents)]
    chunks = []
    i = 0
    while i < len(sents):
        chunk = sents[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        if i + chunk_size >= len(sents):
            break
        i += (chunk_size - overlap)
    return chunks

# --- Placeholder question generation ---
def generate_mock_questions(paragraphs, num_mcq=3, num_short=3):
    questions = []
    for i, para in enumerate(paragraphs[: num_mcq + num_short]):
        if i < num_mcq:
            questions.append(f"MCQ {i+1}: What is a key idea from this paragraph?\n -> {para[:80]}...")
        else:
            questions.append(f"Short Q {i+1-num_mcq}: Explain briefly: {para[:80]}...")
    return questions


# --- Streamlit UI ---
st.set_page_config(page_title="Smart Learning Platform", layout="wide")

st.title("📘 Smart Learning Platform")
st.write("Upload your study material and generate questions automatically!")

# File uploader
uploaded_files = st.file_uploader(
    "Upload PDF files (you can select multiple)", type=["pdf"], accept_multiple_files=True
)

# Backend selector (👉 needs to come BEFORE we call extract_text)
backend = st.radio("Choose extraction backend", ["pypdf2", "pymupdf"])

# Options
num_mcq = st.number_input("Number of MCQs", min_value=1, max_value=20, value=5)
num_short = st.number_input("Number of Short Questions", min_value=1, max_value=20, value=5)
chunk_size = st.number_input("Chunk sentences", min_value=2, max_value=20, value=6)
overlap = st.number_input("Overlap sentences", min_value=0, max_value=10, value=2)

if uploaded_files and st.button("🔍 Extract & Generate Questions"):
    all_paragraphs = []
    for uploaded_file in uploaded_files:
        # save temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        st.info(f"Processing: {uploaded_file.name}")
        text_or_blocks = extract_text_from_pdf(temp_path, backend=backend)

        # If pypdf2 returns string, split into paragraphs
        if backend == "pypdf2":
            paras = [p.strip() for p in text_or_blocks.split("\n") if p.strip()]
        else:
            paras = text_or_blocks  # pymupdf already returns list[str]

        # tag paragraphs with filename
        paras = [f"[{uploaded_file.name}] {p}" for p in paras]
        all_paragraphs.extend(paras)

        # cleanup
        os.remove(temp_path)

    st.success(f"✅ Extracted {len(all_paragraphs)} paragraphs/blocks across {len(uploaded_files)} files")

    # --------- CHUNKING ---------
    chunks = []
    for para in all_paragraphs:
        if para.startswith("[Table_"):
            chunks.append(para)  # keep tables as-is
        else:
            if len(para.split()) < 40:  # small para, keep as is
                chunks.append(para)
            else:
                chunks.extend(sliding_window_chunk_sentences(para, chunk_size=chunk_size, overlap=overlap))

    st.success(f"✅ Prepared {len(chunks)} chunks for Question Generation")

    st.write("📖 Sample Chunks")
    for i, c in enumerate(chunks[:5], 1):
        st.markdown(f"**Chunk {i}:** {c[:300]}...")

    # Generate mock questions
    st.subheader("❓ Generated Questions")
    questions = generate_mock_questions(chunks, num_mcq, num_short)
    for q in questions:
        st.write(q)
