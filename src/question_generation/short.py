import re

def generate_short_question(text, tokenizer, model, max_input_length=512):
    """
    Generate a short-answer question and its correct answer from a text chunk.
    """
    # Clean PDF prefix
    clean_text = re.sub(r"\[[^\]]+\.pdf\]\s*", "", text)
    clean_text = re.sub(r"<answer>.*?</answer>", "", clean_text)

    # Generate question
    input_text = f"generate short question: {clean_text}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=max_input_length, truncation=True)
    outputs = model.generate(inputs, max_length=64, num_beams=4, early_stopping=True)
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if not question.strip():
        return None

    # Generate answer
    answer_input_text = f"answer the question: {question} based on: {clean_text}"
    answer_inputs = tokenizer.encode(answer_input_text, return_tensors="pt", max_length=max_input_length, truncation=True)
    answer_outputs = model.generate(answer_inputs, max_length=128, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(answer_outputs[0], skip_special_tokens=True)

    if not answer.strip():
        return None

    return {
        "type": "short",
        "question": question,
        "answer": answer
    }
