import streamlit as st
import tempfile
import os
import re
from transformers import T5ForConditionalGeneration, T5Tokenizer
from src.preprocessing.extract_text import extract_text_from_pdf
from src.question_generation.highlight import _SPACY_AVAILABLE
from src.question_generation.mcq import generate_mcq_with_options
from src.question_generation.short import generate_short_question


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

def clean_pdf_prefix(text: str) -> str:
    """
    Remove PDF filename tags like [Constituency Grammars.pdf] from the text.
    """
    return re.sub(r"\[[^\]]+\.pdf\]\s*", "", text)

def clean_chunk_text(text):
    # Remove PDF tags
    text = re.sub(r"\[[^\]]+\.pdf\]\s*", "", text)
    # Remove bullets, special chars
    text = re.sub(r"[•\-\*\n]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



# --- Real Question Generation using T5 ---

# Load once (can take a few seconds)
@st.cache_resource(show_spinner=True)
def load_qg_model(model_name="valhalla/t5-base-qg-hl"):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_qg_model()


def generate_questions(chunks, num_mcq=5, num_short=5):
    questions = []

    mcq_chunks = chunks[:num_mcq]
    short_chunks = chunks[num_mcq:num_mcq+num_short]

    # MCQs
    for i, chunk in enumerate(mcq_chunks, 1):
        #clean_chunk = clean_pdf_prefix(chunk)
        clean_chunk = clean_chunk_text(chunk)
        mcq = generate_mcq_with_options(clean_chunk, tokenizer, model)
        if mcq:
            mcq["id"] = i
            questions.append(mcq)

    # Short-answer
    for i, chunk in enumerate(short_chunks, 1):
        clean_chunk = clean_pdf_prefix(chunk)
        sa = generate_short_question(clean_chunk, tokenizer, model)
        if sa:
            sa["id"] = i
            questions.append(sa)

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
    questions = generate_questions(chunks, num_mcq=num_mcq, num_short=num_short)
    
    # --- Display nicely in Streamlit ---
    st.subheader("❓ Generated Questions")

    for q in questions:
        if q["type"] == "mcq":
            st.write(f"MCQ {q['id']}: {q['question']}")
            option_labels = ["A", "B", "C", "D"]
            for idx, opt in enumerate(q["options"]):
                st.write(f"{option_labels[idx]}) {opt}")
            correct_idx = q["options"].index(q["answer"])
            st.write(f"Answer: {option_labels[correct_idx]}")
        else:
            st.write(f"Short Q {q['id']}: {q['question']}")
