import streamlit as st
import tempfile
import os
import re
from transformers import T5ForConditionalGeneration, T5Tokenizer
from src.preprocessing.extract_text import extract_text_from_pdf
from src.question_generation.highlight import highlight_answer
from src.question_generation.mcq import generate_mcq_with_options

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

# --- Real Question Generation using T5 ---

# Load once (can take a few seconds)
@st.cache_resource(show_spinner=True)
def load_qg_model(model_name="valhalla/t5-base-qg-hl"):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_qg_model()


def generate_questions(chunks, num_mcq=5, num_short=5, max_input_length=512):
    questions = []
    mcq_chunks = chunks[:num_mcq]
    short_chunks = chunks[num_mcq:num_mcq + num_short]

    def qg_from_text(text):
        input_text = f"generate question: {text}"
        inputs = tokenizer.encode(
            input_text, return_tensors="pt", max_length=max_input_length, truncation=True
        )
        outputs = model.generate(inputs, max_length=64, num_beams=4, early_stopping=True)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # MCQs with options
    for i, chunk in enumerate(mcq_chunks, 1):
        mcq = generate_mcq_with_options(chunk, tokenizer, model, max_input_length=max_input_length)
        if mcq:
            questions.append({
                "type": "mcq",
                "id": i,
                "question": mcq["question"],
                "options": mcq["options"],
                "answer": mcq["answer"]
            })

    # Short-answer
    for i, chunk in enumerate(short_chunks, 1):
        highlighted = highlight_answer(chunk, use_spacy=True)
        if highlighted:
            question = qg_from_text(highlighted)
            questions.append({
                "type": "short",
                "id": i,
                "question": question
            })

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
            st.markdown(f"**MCQ {q['id']}: {q['question']}**")
            for idx, option in enumerate(q["options"], 1):
                # Use letters A, B, C, D…
                letter = chr(64 + idx)  # 65 is 'A'
                st.markdown(f"{letter}) {option}")
            st.markdown(f"**Answer:** {chr(64 + q['options'].index(q['answer']) + 1)}")
            st.markdown("---")  # separator
        elif q["type"] == "short":
            st.markdown(f"**Short Q {q['id']}: {q['question']}**")
            st.markdown("---")
