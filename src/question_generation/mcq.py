import random
from src.question_generation.highlight import highlight_answer
from src.question_generation.highlight import highlight_answer, _SPACY_AVAILABLE, _ensure_nlp

def generate_mcq_with_options(text, tokenizer, model, max_input_length=512, num_distractors=3):
    highlighted = highlight_answer(text, use_spacy=True)
    if "<answer>" not in highlighted:
        return None

    correct = highlighted.split("<answer>")[1].split("</answer>")[0]

    # Generate the question
    input_text = f"generate question: {highlighted}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=max_input_length, truncation=True)
    outputs = model.generate(inputs, max_length=64, num_beams=4, early_stopping=True)
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Generate smarter distractors using spaCy
    distractors = []
    if _SPACY_AVAILABLE:
        nlp = _ensure_nlp()
        doc = nlp(text)
        # candidate words: noun chunks and proper nouns
        candidates = [nc.text for nc in doc.noun_chunks] + [ent.text for ent in doc.ents]
        # remove correct answer and duplicates
        candidates = list(set([c for c in candidates if correct.lower() not in c.lower()]))
        distractors = random.sample(candidates, min(len(candidates), num_distractors))

    # Fallback: random words if spaCy fails
    if len(distractors) < num_distractors:
        words = [w for w in text.split() if w.isalpha() and len(w) > 3 and correct.lower() not in w.lower()]
        distractors += random.sample(words, min(len(words), num_distractors - len(distractors)))

    options = [correct] + distractors
    random.shuffle(options)

    return {
        "question": question,
        "options": options,
        "answer": correct
    }

