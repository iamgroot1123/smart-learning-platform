from flask import Flask, request, render_template, flash, redirect, url_for, jsonify, session
import tempfile
import os
import re
import random
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    BartForConditionalGeneration, 
    BartTokenizer,
    AutoModelForSequenceClassification,
    AutoTokenizer
)
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import string
import torch
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

from src.preprocessing.extract_text import extract_text_from_pdf
from src.question_generation.highlight import _SPACY_AVAILABLE
from src.question_generation.enhanced_mcq import generate_enhanced_mcq
from src.question_generation.short import generate_short_question
from src.question_generation.filters import apply_all_filters
import logging

# Configure basic logging for the app
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure secret key

# Load enhanced models for better quality
logger.info("Loading enhanced models...")

# T5 model for question generation
def load_qg_model(model_name="valhalla/t5-base-qg-hl"):
    try:
        logger.info(f"Loading QG model: {model_name}")
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        if torch.cuda.is_available():
            model = model.cuda()
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading QG model: {e}")
        raise  # Re-raise the error since we don't have another fallback

# BART-Large for better answer generation
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
if torch.cuda.is_available():
    bart_model = bart_model.cuda()

# SBERT for semantic search and distractor generation
logger.info("Loading SBERT model...")
try:
    sbert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    if torch.cuda.is_available():
        sbert_model = sbert_model.cuda()
    logger.info("SBERT model loaded successfully")
except Exception as e:
    logger.warning(f"Failed to load mpnet model, falling back to MiniLM: {e}")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_pdf_prefix(text: str) -> str:
    """
    Remove PDF filename tags like [Constituency Grammars.pdf] from the text.
    """
    return re.sub(r"[[^]]+\.pdf]\s*", "", text)

def clean_chunk_text(text: str) -> str:
    """Enhanced text cleaning with advanced noise detection"""
    # Remove PDF artifacts and metadata
    text = re.sub(r'\[[^\]]+\.pdf\]\s*', '', text)
    text = re.sub(r'\([Pp]age \d+\)', '', text)
    
    # Remove control characters and unusual control ranges, but keep unicode characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    
    # Fix common OCR issues
    text = re.sub(r'([a-z])\.([A-Z])', r'\1. \2', text)  # Fix sentence boundaries
    text = re.sub(r'l([^a-zA-Z])', r'1\1', text)  # Common l->1 error
    text = re.sub(r'O([^a-zA-Z])', r'0\1', text)  # Common O->0 error
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Check readability
    if len(text) < 10:  # Too short to be meaningful
        return ""
        
    try:
        # Avoid TextBlob.detect_language() because it may not be available or require network access.
        # We'll rely on the entropy heuristic below to detect gibberish.
        blob = TextBlob(text)
        
        # Check for gibberish using character entropy (looser threshold)
        try:
            char_freq = {}
            for char in text.lower():
                char_freq[char] = char_freq.get(char, 0) + 1
            entropy = sum([-freq/len(text) * np.log2(freq/len(text)) for freq in char_freq.values()])
            # looser threshold: only reject extremely low-entropy text (e.g., repeated punctuation)
            if entropy < 1.5:
                return ""
        except Exception:
            pass
    except:
        pass
        
    return text

# --- Real Question Generation using T5 ---

# Initialize tokenizer and model from the enhanced loading function above
tokenizer, model = load_qg_model()

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

def chunk_text_into_sentences(text: str, sentences_per_chunk: int = 5, overlap: int = 2) -> List[str]:
    """
    Enhanced text chunking with overlap for better context preservation.
    Args:
        text: Input text to chunk
        sentences_per_chunk: Number of sentences per chunk
        overlap: Number of sentences to overlap between chunks
    Returns:
        List of text chunks
    """
    sentences = sent_tokenize(text)
    chunks = []
    
    for i in range(0, len(sentences), sentences_per_chunk - overlap):
        end_idx = min(i + sentences_per_chunk, len(sentences))
        chunk = " ".join(sentences[i:end_idx])
        if len(chunk.split()) >= 10:  # Minimum chunk size
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

def extract_keywords_and_sentences(text):
    sentences = sent_tokenize(text)
    keyword_sentence_mapping = {}
    filtered_keys = []

    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        # Part-of-speech tagging
        tagged_words = nltk.pos_tag(words)
        # Require sentence to have at least one verb to be a good source for QA
        has_verb = any(tag.startswith('V') for _, tag in tagged_words)
        if not has_verb:
            continue

        for word, tag in tagged_words:
            # Consider nouns and proper nouns as keywords
            if tag.startswith('N') and len(word) > 2 and word.lower() not in stop_words:
                # Filter out tokens with digits or many non-alpha chars (likely OCR noise)
                if not word.isalpha():
                    continue
                # Filter out tokens that are mostly repeated chars or very short after lemmatization
                lemmatized_word = lemmatizer.lemmatize(word.lower())
                if len(lemmatized_word) < 3:
                    continue
                # Reject tokens with excessive repeated characters (e.g., 'lllll')
                if re.search(r'(.)\1{3,}', lemmatized_word):
                    continue

                if lemmatized_word not in filtered_keys:
                    filtered_keys.append(lemmatized_word)
                if lemmatized_word not in keyword_sentence_mapping:
                    keyword_sentence_mapping[lemmatized_word] = []
                keyword_sentence_mapping[lemmatized_word].append(sentence)
    return filtered_keys, keyword_sentence_mapping

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

    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No file part'}), 400 if is_ajax else redirect(request.url)

        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No selected file'}), 400 if is_ajax else redirect(request.url)

        # Using PyMuPDF as the default and only PDF extraction backend
        num_mcq = int(request.form.get('num_mcq', 5))
        num_short = int(request.form.get('num_short', 5))

        # 1. Enhanced Text Extraction with Advanced Cleaning
        full_text = ""
        for uploaded_file in files:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp.flush()
                    # Extract text from PDF using PyMuPDF
                    text_or_blocks = extract_text_from_pdf(tmp.name)

                    # Log warning if no text was extracted
                    if (isinstance(text_or_blocks, str) and not text_or_blocks.strip()) or (isinstance(text_or_blocks, list) and len(text_or_blocks) == 0):
                        logger.warning("No text could be extracted from the PDF file")

                    # Combine blocks into a single text blob
                    cleaned_text = clean_chunk_text(text_or_blocks if isinstance(text_or_blocks, str) else " ".join(text_or_blocks))
                    if cleaned_text:  # Only add non-empty chunks
                        full_text += " " + cleaned_text
                os.remove(tmp.name)
            except Exception as e:
                error_msg = f"Error processing file {uploaded_file.filename}: {str(e)}"
                return jsonify({'error': error_msg}), 400 if is_ajax else redirect(url_for('index'))

        if not full_text.strip():
            return jsonify({'error': 'Could not extract readable text from the document.'}), 400 if is_ajax else redirect(url_for('index'))

        # 2. Enhanced Text Chunking with Overlap
        chunks = chunk_text_into_sentences(full_text, sentences_per_chunk=8, overlap=3)

        if not chunks:
            return jsonify({'error': 'Could not split text into meaningful chunks.'}), 400 if is_ajax else redirect(url_for('index'))

        # 3. Generate Embeddings with Enhanced Model
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
            
            try:
                # Try to generate an MCQ from this chunk
                mcq = generate_enhanced_mcq(chunk, tokenizer, model, sbert_model)
                
                # Validate the generated MCQ
                if mcq and isinstance(mcq, dict) and apply_all_filters(mcq):
                    mcq_count += 1
                    mcq["id"] = mcq_count
                    mcq["type"] = "mcq"
                    questions.append(mcq)
                    logger.info(f"Generated high-quality MCQ {mcq_count} with score: {mcq.get('metadata', {}).get('question_quality', 0):.2f}")
                else:
                    logger.debug("Generated MCQ failed validation")
            except Exception as e:
                logger.error(f"Error generating MCQ: {str(e)}")
                continue
            # --- Generate Short Answer Questions with RAG ---
        short_q_count = 0
        for chunk in random.sample(chunks, min(len(chunks), num_short * 2)):
            if short_q_count >= num_short:
                break
            
            try:
                sa = generate_short_question(chunk, tokenizer, model)
                if sa and apply_all_filters({"type": "short", "question": sa["question"]}):
                    # Query Processing and Similarity Search
                    question_embedding = sbert_model.encode([sa["question"]], convert_to_tensor=True)
                    k = min(3, len(chunks))
                    distances, top_k_indices = index.search(question_embedding.cpu().numpy().reshape(1, -1), k=k)
                    context = " ".join([chunk_mapping[i] for i in top_k_indices[0]])

                    # Integration with LLM (BART) for Answer Generation
                    generated_answer = generate_answer_from_context(sa["question"], context)
                    if generated_answer:
                        short_q_count += 1
                        sa["id"] = short_q_count
                        sa["type"] = "short"
                        sa["correct_answer"] = generated_answer
                        questions.append(sa)
                else:
                    logger.debug("Generated short answer question failed validation")
            except Exception as e:
                logger.error(f"Error generating short answer question: {str(e)}")
                continue

        if not questions:
            return jsonify({'error': 'Failed to generate any valid questions'}), 400

        # Log generated content
        logger.info(f"Generated {len(questions)} questions ({sum(1 for q in questions if q['type']=='mcq')} MCQ, {sum(1 for q in questions if q['type']=='short')} short)")
        
        # Always return JSON for XHR/fetch requests
        response_data = {
            'chunks': chunks[:5],  # Send first 5 chunks for display
            'questions': questions
        }
        logger.debug(f"Returning response with {len(response_data['chunks'])} chunks and {len(response_data['questions'])} questions")
        return jsonify(response_data)

    except Exception as e:
        error_msg = f"Error during question generation: {str(e)}"
        if is_ajax:
            return jsonify({'error': error_msg}), 500
        else:
            flash(error_msg)
            return redirect(url_for('index'))

@app.route('/evaluate_short', methods=['POST'])
def evaluate_short():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    try:
        data = request.get_json()
        user_answer = data.get('user_answer', '').strip()
        correct_answer = data.get('correct_answer', '').strip()
        question_id = data.get('question_id')

        if not user_answer or not correct_answer:
            return jsonify({'error': 'Missing answer data'}), 400

        # Generate embeddings
        user_embedding = sbert_model.encode([user_answer], convert_to_tensor=True)
        correct_embedding = sbert_model.encode([correct_answer], convert_to_tensor=True)

        # Compute similarity
        similarity = float(util.pytorch_cos_sim(user_embedding, correct_embedding)[0][0])

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

        return jsonify({
            "similarity": percentage,
            "status": status,
            "color": color,
            "correct_answer": correct_answer,
            "question_id": question_id
        })
    except Exception as e:
        logger.error(f"Error evaluating short answer: {str(e)}")
        return jsonify({'error': 'Error evaluating answer'}), 500

if __name__ == '__main__':
    app.run(debug=True)
