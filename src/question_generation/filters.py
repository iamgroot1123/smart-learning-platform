import re

def is_trivial_mcq(q_obj):
    """Checks if the answer is just a word-for-word copy from the question."""
    question = q_obj["question"].lower()
    answer = q_obj["answer"].lower()
    # If the answer (e.g., "syntactic constituency") is in the question
    # (e.g., "what is syntactic constituency?"), it's a bad question.
    if f" {answer} " in f" {question} ":
        return True
    return False

def has_plausible_distractors(q_obj, min_option_length=2):
    """Checks if the multiple-choice options are not just single, junk words."""
    options = q_obj["options"]
    # Count how many options are longer than a single character or a very short word.
    plausible_options = [opt for opt in options if len(opt) > min_option_length]
    # If less than half the options are plausible, fail the check.
    if len(plausible_options) < len(options) / 2:
        return False
    return True

def is_valid_question_format(q_text):
    """Performs basic structural checks on the question text."""
    # Must end with a question mark.
    if not q_text.endswith("?"):
        return False
    # Must start with a common question word (case-insensitive).
    if not re.match(r"^(what|who|where|when|why|how|which|is|are|do|does)", q_text, re.IGNORECASE):
        return False
    # Must not be excessively short.
    if len(q_text.split()) < 4:
        return False
    return True

def apply_all_filters(q_obj):
    """
    A single function to run all relevant filters on a question object.
    Returns True if the question is GOOD, False if it's BAD.
    """
    q_text = q_obj["question"]

    if not is_valid_question_format(q_text):
        return False

    if q_obj["type"] == "mcq":
        if is_trivial_mcq(q_obj):
            return False
        if not has_plausible_distractors(q_obj):
            return False
    
    # You can add more filters here for short-answer questions later.
    
    return True