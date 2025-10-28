from flask import Flask, request, render_template, flash, redirect, url_for, jsonify
import tempfile
import os
import re
import random
from transformers import T5ForConditionalGeneration, T5Tokenizer, BartForConditionalGeneration, BartTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

from src.preprocessing.extract_text import extract_text_from_pdf
from src.question_generation.highlight import _SPACY_AVAILABLE
from src.question_generation.mcq import generate_mcq_with_options
from src.question_generation.short import generate_short_question
from src.question_generation.filters import apply_all_filters

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure secret key

# Load BART model for answer generation
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

def clean_pdf_prefix(text: str) -> str:
    """
    Remove PDF filename tags like [Constituency Grammars.pdf] from the text.
    """
    return re.sub(r"[[^]]+\.pdf]\s*", "", text)

def clean_chunk_text(text):
    # Remove our custom table tags using regex
    text = re.sub(r"[/?Table_Page\d+]", "", text)
    # Remove PDF tags
    text = re.sub(r"[[^]]+\.pdf]\s*", "", text)
    # Remove bullets, special chars, and extra whitespace
    text = re.sub(r"[•\-\*]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# --- Real Question Generation using T5 ---

# Load once (can take a few seconds)
def load_qg_model(model_name="valhalla/t5-base-qg-hl"):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_qg_model()

# Load SBERT model for evaluation
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def generate_answer_from_context(question, context):
    """
    Generate an answer to a question based on the provided context using BART.
    """
    input_text = f"question: {question} context: {context}"
    input_ids = bart_tokenizer(input_text, return_tensors='pt', max_length=1024, truncation=True)['input_ids']
    summary_ids = bart_model.generate(input_ids, num_beams=4, max_length=100, early_stopping=True)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def chunk_text_into_sentences(text, sentences_per_chunk=5):
    """
    Splits text into chunks of a specified number of sentences.
    """
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)
    return chunks

def preprocess_text(text):
    """
    Preprocess text: lowercase, remove stopwords, lemmatize, strip punctuation and whitespace.
    """
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    words = nltk.word_tokenize(text)
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    # Join back
    return ' '.join(words).strip()

def evaluate_short_answer(user_answer, correct_answer):
    """
    Evaluate short answer using SBERT and cosine similarity.
    """
    # Preprocess both answers
    user_processed = preprocess_text(user_answer)
    correct_processed = preprocess_text(correct_answer)

    # Generate embeddings
    user_embedding = sbert_model.encode([user_processed])
    correct_embedding = sbert_model.encode([correct_processed])

    # Compute cosine similarity
    similarity = cosine_similarity(user_embedding, correct_embedding)[0][0]

    # Map to percentage
    percentage = int(similarity * 100)

    # Determine status
    if similarity >= 0.85:
        status = "Correct"
        color = "green"
    elif similarity >= 0.65:
        status = "Partially correct"
        color = "orange"
    else:
        status = "Incorrect"
        color = "red"

    return {
        "similarity": percentage,
        "status": status,
        "color": color,
        "correct_answer": correct_answer
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

    if 'files' not in request.files:
        return jsonify({'error': 'No file part'}), 400 if is_ajax else redirect(request.url)

    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No selected file'}), 400 if is_ajax else redirect(request.url)

    backend = request.form.get('backend', 'pypdf2')
    num_mcq = int(request.form.get('num_mcq', 5))
    num_short = int(request.form.get('num_short', 5))

    # 1. Extracting Text from PDF Documents
    full_text = ""
    for uploaded_file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            text_or_blocks = extract_text_from_pdf(tmp.name, backend=backend)
            full_text += " " + clean_chunk_text(text_or_blocks if isinstance(text_or_blocks, str) else " ".join(text_or_blocks))
        os.remove(tmp.name)

    # 2. Chunking the Text
    chunks = chunk_text_into_sentences(full_text)

    if not chunks:
        if is_ajax:
            return jsonify({'error': 'Could not extract any text from the document.'}), 400
        else:
            flash('Could not extract any text from the document.')
            return redirect(url_for('index'))

    # 3. Generating Embeddings for Each Chunk
    chunk_embeddings = sbert_model.encode(chunks, convert_to_tensor=True)

    # 4. Indexing with FAISS
    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    index.add(chunk_embeddings.cpu().numpy())
    chunk_mapping = {i: chunk for i, chunk in enumerate(chunks)}

    questions = []
    # --- Generate MCQs ---
    mcq_count = 0
    for chunk in random.sample(chunks, min(len(chunks), num_mcq * 2)):
        if mcq_count >= num_mcq:
            break
        mcq = generate_mcq_with_options(chunk, tokenizer, model)
        if mcq and apply_all_filters(mcq):
            mcq_count += 1
            mcq["id"] = mcq_count
            questions.append(mcq)

    # --- Generate Short Answer Questions with RAG ---
    short_q_count = 0
    for chunk in random.sample(chunks, min(len(chunks), num_short * 2)):
        if short_q_count >= num_short:
            break
        sa = generate_short_question(chunk, tokenizer, model)
        if sa and apply_all_filters({"type": "short", "question": sa["question"]}):
            # 5. Query Processing and Similarity Search
            question_embedding = sbert_model.encode([sa["question"]], convert_to_tensor=True)
            k = min(3, len(chunks))
            _, top_k_indices = index.search(question_embedding.cpu().numpy(), k=k)
            context = " ".join([chunk_mapping[i] for i in top_k_indices[0]])

            # 6. Integration with LLM (BART) for Answer Generation
            generated_answer = generate_answer_from_context(sa["question"], context)
            
            short_q_count += 1
            sa["id"] = short_q_count
            sa["correct_answer"] = generated_answer
            questions.append(sa)

    if is_ajax:
        return jsonify({'questions': questions})
    else:
        return render_template('results.html', questions=questions)

@app.route('/evaluate_short', methods=['POST'])
def evaluate_short():
    data = request.get_json()
    user_answer = data.get('user_answer', '').strip()
    correct_answer = data.get('correct_answer', '').strip()
    question_id = data.get('question_id')

    if not user_answer or not correct_answer:
        return jsonify({'error': 'Missing answer data'}), 400

    result = evaluate_short_answer(user_answer, correct_answer)
    result['question_id'] = question_id
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
