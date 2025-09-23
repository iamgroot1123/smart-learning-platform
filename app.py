import streamlit as st
import tempfile
import os
import re
import random
from transformers import T5ForConditionalGeneration, T5Tokenizer
from src.preprocessing.extract_text import extract_text_from_pdf
from src.question_generation.highlight import _SPACY_AVAILABLE
from src.question_generation.mcq import generate_mcq_with_options
from src.question_generation.short import generate_short_question
from src.question_generation.filters import apply_all_filters


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
    # Remove our custom table tags using regex
    text = re.sub(r"\[/?Table_Page\d+\]", "", text)
    # Remove PDF tags
    text = re.sub(r"\[[^\]]+\.pdf\]\s*", "", text)
    # Remove bullets, special chars, and extra whitespace
    text = re.sub(r"[•\-\*\n]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def create_context_windows(chunks, window_size=3, overlap=1):
    """
    Creates larger 'mega-chunks' by applying a sliding window over the existing chunks.
    This gives the model more context for each question.
    """
    if not chunks or window_size <= 0:
        return []

    # If there are fewer chunks than the window size, just combine them all.
    if len(chunks) <= window_size:
        return [" ".join(chunks)]

    windowed_chunks = []
    step = window_size - overlap
    for i in range(0, len(chunks) - window_size + 1, step):
        window = chunks[i:i + window_size]
        windowed_chunks.append(" ".join(window))
    
    return windowed_chunks


# --- Real Question Generation using T5 ---

# Load once (can take a few seconds)
@st.cache_resource(show_spinner=True)
def load_qg_model(model_name="valhalla/t5-base-qg-hl"):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_qg_model()


# In app.py

def generate_questions(chunks, num_mcq=5, num_short=5):
    # STEP 1: Use the new Context Windowing function.
    # The UI inputs for 'chunk_size' and 'overlap' now control paragraphs!
    context_windows = create_context_windows(chunks, window_size=st.session_state.chunk_size, overlap=st.session_state.overlap)
    if not context_windows:
        st.warning("Not enough content to generate questions. Try a smaller PDF or different settings.")
        return []
    
    st.info(f"Created {len(context_windows)} context windows for generation.")

    questions = []
    generated_questions_set = set() # STEP 2: Use a set to track and prevent duplicates.
    
    random.shuffle(context_windows)
    
    # --- Generate MCQs with Quality Filtering ---
    generated_mcqs = 0
    max_attempts = len(context_windows) * 2
    
    st.info(f"Generating {num_mcq} MCQs...")
    with st.spinner('Crafting and filtering MCQs...'):
        for i in range(max_attempts):
            if generated_mcqs >= num_mcq:
                break
            window = context_windows[i % len(context_windows)]
            mcq = generate_mcq_with_options(window, tokenizer, model)

            if mcq and apply_all_filters(mcq):
                q_text = mcq['question'].strip().lower()
                if q_text not in generated_questions_set:
                    generated_questions_set.add(q_text)
                    generated_mcqs += 1
                    mcq["id"] = generated_mcqs
                    questions.append(mcq)

    # --- Generate Short Answer Questions with Quality Filtering ---
    generated_short_qs = 0
    
    st.info(f"Generating {num_short} Short Answer questions...")
    with st.spinner('Crafting and filtering Short Answer questions...'):
        for i in range(max_attempts):
            if generated_short_qs >= num_short:
                break
            window = context_windows[i % len(context_windows)]
            sa = generate_short_question(window, tokenizer, model)

            if sa:
                sa_obj = {"type": "short", "question": sa["question"]}
                if apply_all_filters(sa_obj):
                    q_text = sa['question'].strip().lower()
                    if q_text not in generated_questions_set:
                        generated_questions_set.add(q_text)
                        generated_short_qs += 1
                        sa["id"] = generated_short_qs
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
st.number_input("Paragraphs per Chunk", min_value=1, max_value=20, value=3, key="chunk_size")
st.number_input("Overlap Paragraphs", min_value=0, max_value=10, value=1, key="overlap")


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

    # --------- CONTEXT-AWARE CHUNKING ---------
    st.info("Applying context-aware chunking...")
    chunks = []
    short_paragraph_buffer = ""
    MIN_WORDS_PER_CHUNK = 40
    MAX_WORDS_PER_CHUNK = 300 # You can adjust this value

    for para in all_paragraphs:
        # Keep tables as a single, intact chunk
        if para.startswith("[Table_"):
            # First, process any text waiting in the buffer
            if short_paragraph_buffer:
                chunks.append(short_paragraph_buffer.strip())
                short_paragraph_buffer = ""
            chunks.append(para)
            continue

        # Clean the paragraph text before processing
        clean_para = clean_chunk_text(para)
        word_count = len(clean_para.split())

        # Case 1: Paragraph is the perfect size
        if MIN_WORDS_PER_CHUNK <= word_count <= MAX_WORDS_PER_CHUNK:
            # Process the buffer first
            if short_paragraph_buffer:
                chunks.append(short_paragraph_buffer.strip())
                short_paragraph_buffer = ""
            # Add the current paragraph as its own chunk
            chunks.append(clean_para)

        # Case 2: Paragraph is too long, so we slide the window
        elif word_count > MAX_WORDS_PER_CHUNK:
            # Process the buffer first
            if short_paragraph_buffer:
                chunks.append(short_paragraph_buffer.strip())
                short_paragraph_buffer = ""
            # Add the chunks from the long paragraph
            chunks.extend(sliding_window_chunk_sentences(clean_para, chunk_size=chunk_size, overlap=overlap))

        # Case 3: Paragraph is too short, so add it to the buffer
        else:
            short_paragraph_buffer += " " + clean_para

    # After the loop, add any remaining text from the buffer as the last chunk
    if short_paragraph_buffer:
        chunks.append(short_paragraph_buffer.strip())

    st.success(f"✅ Prepared {len(chunks)} high-quality chunks for Question Generation")

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
