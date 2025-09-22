from src.question_generation.highlight import highlight_answer

def generate_short_question(text, tokenizer, model, max_input_length=512):
    """
    Generate a short-answer type question from text using a highlighted answer.
    """
    highlighted = highlight_answer(text, use_spacy=True)
    if "<answer>" not in highlighted:
        return None

    input_text = f"generate question: {highlighted}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=max_input_length, truncation=True)
    outputs = model.generate(
        inputs,
        max_length=64,
        do_sample=True,   # enable stochastic generation
        top_k=50,
        top_p=0.95,
        num_return_sequences=1
    )
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "type": "short",
        "question": question
    }
