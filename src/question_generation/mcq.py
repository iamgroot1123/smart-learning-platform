import random
import re
from src.question_generation.highlight import highlight_answer, _SPACY_AVAILABLE, _ensure_nlp

import nltk

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
    
from nltk.corpus import wordnet


def generate_mcq_with_options(text, tokenizer, model, max_input_length=512, num_distractors=3):
    """
    Generate one MCQ with contextually relevant distractors.
    """
    highlighted = highlight_answer(text, use_spacy=True)
    if "<answer>" not in highlighted:
        return None

    # Extract the correct answer
    correct_answer = highlighted.split("<answer>")[1].split("</answer>")[0]

    # --- Generate question ---
    input_text = f"generate question: {highlighted}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=max_input_length, truncation=True)
    outputs = model.generate(inputs, max_length=64, num_beams=4, early_stopping=True)
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # --- NEW: Generate Context-Aware Distractors ---
    distractor_candidates = set()
    try:
        if _SPACY_AVAILABLE:
            nlp = _ensure_nlp()
            doc = nlp(text)
            
            # Add other named entities from the text as distractors
            for ent in doc.ents:
                ent_text = ent.text.strip()
                if ent_text.lower() != correct_answer.lower():
                    distractor_candidates.add(ent_text)

            # Add other noun chunks as distractors
            for nc in doc.noun_chunks:
                nc_text = nc.text.strip()
                if nc_text.lower() != correct_answer.lower():
                    distractor_candidates.add(nc_text)

    except Exception:
        # Fallback to simple word extraction if spaCy fails
        words = [w for w in re.findall(r'\b[A-Z][a-z]*\b|\b\w{4,}\b', text) if w.lower() != correct_answer.lower()]
        distractor_candidates.update(words)

    # Filter out candidates that are substrings of the correct answer or vice-versa
    final_candidates = [
        cand for cand in distractor_candidates
        if cand.lower() not in correct_answer.lower() and correct_answer.lower() not in cand.lower() and len(cand) > 1
    ]

    if not final_candidates:
        return None # Not enough material to create good distractors

    # Select the distractors
    distractors = random.sample(final_candidates, min(len(final_candidates), num_distractors))
    
    options = [correct_answer] + distractors
    random.shuffle(options)

    return {
        "type": "mcq",
        "question": question,
        "options": options,
        "answer": correct_answer
    }