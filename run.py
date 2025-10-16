from flask import Flask, request, render_template, flash, redirect, url_for
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

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure secret key

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

# --- Real Question Generation using T5 ---

# Load once (can take a few seconds)
def load_qg_model(model_name="valhalla/t5-base-qg-hl"):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_qg_model()

def generate_questions(chunks, num_mcq=5, num_short=5):
    questions = []

    # Shuffle chunks to get variety if we generate fewer than total chunks
    random.shuffle(chunks)

    # --- Generate MCQs with Quality Filtering ---
    mcq_count = 0
    generated_mcqs = 0
    max_attempts = len(chunks) * 2  # Try a bit harder to find good questions

    print(f"Generating {num_mcq} MCQs...")
    for i in range(max_attempts):
        if generated_mcqs >= num_mcq:
            break
        chunk = chunks[i % len(chunks)]  # Cycle through chunks
        mcq = generate_mcq_with_options(chunk, tokenizer, model)
        if mcq and apply_all_filters(mcq):
            generated_mcqs += 1
            mcq["id"] = generated_mcqs
            questions.append(mcq)

    # --- Generate Short Answer Questions with Quality Filtering ---
    short_q_count = 0
    generated_short_qs = 0

    print(f"Generating {num_short} Short Answer questions...")
    for i in range(max_attempts):
        if generated_short_qs >= num_short:
            break
        chunk = chunks[i % len(chunks)]  # Cycle through chunks
        sa = generate_short_question(chunk, tokenizer, model)
        # We need to create a temporary object for the filter function
        if sa:
            sa_obj = {"type": "short", "question": sa["question"]}
            if apply_all_filters(sa_obj):
                generated_short_qs += 1
                sa["id"] = generated_short_qs
                questions.append(sa)

    return questions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # Check if request is AJAX (for SPA)
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

    if 'files' not in request.files:
        if is_ajax:
            return {'error': 'No file part'}, 400
        flash('No file part')
        return redirect(request.url)

    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        if is_ajax:
            return {'error': 'No selected file'}, 400
        flash('No selected file')
        return redirect(request.url)

    backend = request.form.get('backend', 'pypdf2')
    num_mcq = int(request.form.get('num_mcq', 5))
    num_short = int(request.form.get('num_short', 5))
    chunk_size = int(request.form.get('chunk_size', 6))
    overlap = int(request.form.get('overlap', 2))

    all_paragraphs = []
    for uploaded_file in files:
        if uploaded_file.filename == '':
            continue
        # save temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        print(f"Processing: {uploaded_file.filename}")
        text_or_blocks = extract_text_from_pdf(temp_path, backend=backend)

        # If pypdf2 returns string, split into paragraphs
        if backend == "pypdf2":
            paras = [p.strip() for p in text_or_blocks.split("\n") if p.strip()]
        else:
            paras = text_or_blocks  # pymupdf already returns list[str]

        # tag paragraphs with filename
        paras = [f"[{uploaded_file.filename}] {p}" for p in paras]
        all_paragraphs.extend(paras)

        # cleanup
        os.remove(temp_path)

    print(f"✅ Extracted {len(all_paragraphs)} paragraphs/blocks across {len(files)} files")

    # --------- CONTEXT-AWARE CHUNKING ---------
    print("Applying context-aware chunking...")
    chunks = []
    short_paragraph_buffer = ""
    MIN_WORDS_PER_CHUNK = 40
    MAX_WORDS_PER_CHUNK = 300  # You can adjust this value

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

    print(f"✅ Prepared {len(chunks)} high-quality chunks for Question Generation")

    # Generate questions
    questions = generate_questions(chunks, num_mcq=num_mcq, num_short=num_short)

    if is_ajax:
        # Return JSON for SPA
        return {'chunks': chunks, 'questions': questions}
    else:
        # Fallback to template rendering
        return render_template('results.html', chunks=chunks, questions=questions)

if __name__ == '__main__':
    app.run(debug=True)
