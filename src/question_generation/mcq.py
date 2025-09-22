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
    Generate one MCQ with answer options using spaCy and WordNet for semantic distractors.
    """
    # Clean PDF prefix
    clean_text = re.sub(r"\[[^\]]+\.pdf\]\s*", "", text)

    # Highlight answer
    highlighted = highlight_answer(clean_text, use_spacy=True)
    if "<answer>" not in highlighted:
        return None

    correct = highlighted.split("<answer>")[1].split("</answer>")[0]

    # --- Generate question ---
    input_text = f"generate question: {highlighted}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=max_input_length, truncation=True)
    outputs = model.generate(inputs, max_length=64, num_beams=4, early_stopping=True)
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # --- Generate semantic distractors ---
    candidates = set()

    # spaCy noun chunks and entities
    try:
        if _SPACY_AVAILABLE:
            nlp = _ensure_nlp()
            doc = nlp(clean_text)
            for nc in doc.noun_chunks:
                chunk_text = nc.text.strip()
                if chunk_text.lower() != correct.lower() and len(chunk_text) > 2:
                    candidates.add(chunk_text)
            for ent in doc.ents:
                if ent.text.lower() != correct.lower() and ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "PRODUCT"}:
                    candidates.add(ent.text.strip())
    except Exception:
        pass

    # WordNet synonyms
    for syn in wordnet.synsets(correct):
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != correct.lower():
                candidates.add(name)

    # Fallback to words from the text
    words = [w for w in re.findall(r"\w{3,}", clean_text) if w.lower() != correct.lower()]
    candidates.update(words)

    candidates = list(candidates)
    if not candidates:
        return None

    distractors = random.sample(candidates, min(len(candidates), num_distractors))
    options = [correct] + distractors
    random.shuffle(options)

    return {
        "type": "mcq",
        "question": question,
        "options": options,
        "answer": correct
    }
