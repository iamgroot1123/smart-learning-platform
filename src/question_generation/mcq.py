import random
from src.question_generation.highlight import highlight_answer
from src.question_generation.highlight import highlight_answer, _SPACY_AVAILABLE, _ensure_nlp

def generate_mcq_with_options(text, tokenizer, model, max_input_length=512, num_distractors=3):
    """
    Generate one MCQ with answer options using spaCy for smarter distractors.
    """
    highlighted = highlight_answer(text, use_spacy=True)
    if "<answer>" not in highlighted:
        return None  # no good answer found

    # Extract the correct answer
    correct = highlighted.split("<answer>")[1].split("</answer>")[0].strip()

    # --- Generate the question with sampling ---
    input_text = f"generate question: {highlighted}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=max_input_length, truncation=True)
    outputs = model.generate(
        inputs,
        max_length=64,
        do_sample=True,       # enables stochastic output
        top_k=50,
        top_p=0.95,
        num_return_sequences=1
    )
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # --- Generate smarter distractors ---
    candidates = set()
    try:
        if _SPACY_AVAILABLE:
            nlp = _ensure_nlp()
            doc = nlp(text)

            # Noun chunks
            for nc in doc.noun_chunks:
                chunk_text = nc.text.strip()
                if chunk_text.lower() != correct.lower() and len(chunk_text) > 2 and "pdf" not in chunk_text.lower():
                    candidates.add(chunk_text)

            # Named entities
            for ent in doc.ents:
                if ent.text.lower() != correct.lower() and ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "PRODUCT"}:
                    candidates.add(ent.text.strip())

    except Exception:
        pass

    # Fallback: random words if candidates too few
    if len(candidates) < num_distractors:
        words = [w for w in text.split() if w.isalpha() and len(w) > 3 and w.lower() != correct.lower()]
        candidates.update(words)

    candidates = list(candidates)
    if len(candidates) == 0:
        distractors = []
    else:
        distractors = random.sample(candidates, min(len(candidates), num_distractors))

    options = [correct] + distractors
    random.shuffle(options)

    return {
        "type": "mcq",
        "question": question,
        "options": options,
        "answer": correct
    }
