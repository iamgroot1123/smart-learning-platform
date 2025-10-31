import re

def is_trivial_mcq(q_obj):
    """Checks if the answer is just a word-for-word copy from the question."""
    question = str(q_obj.get("question", "")).lower()
    answer = str(q_obj.get("answer", "")).lower()
    # If the answer (e.g., "syntactic constituency") is in the question
    # (e.g., "what is syntactic constituency?"), it's a bad question.
    if f" {answer} " in f" {question} ":
        return True
    return False

def has_plausible_distractors(q_obj, min_option_length=2):
    """Checks if the multiple-choice options are not just single, junk words."""
    # Accept either 'options' or 'Options' keys and ensure we have a list
    options = q_obj.get("options") if q_obj.get("options") is not None else q_obj.get("Options")
    if not options or not isinstance(options, (list, tuple)):
        return False

    # Count how many options are longer than a threshold
    plausible_options = [opt for opt in options if isinstance(opt, str) and len(opt.strip()) > min_option_length]
    # If less than half the options are plausible, fail the check.
    try:
        if len(plausible_options) < (len(options) / 2):
            return False
    except Exception:
        return False
    return True

def is_valid_question_format(q_text):
    """Performs basic structural checks on the question text."""
    if not isinstance(q_text, str) or not q_text:
        return False

    # Must end with a question mark.
    if not q_text.strip().endswith("?"):
        return False
    # Must start with a common question word (case-insensitive).
    if not re.match(r"^(what|who|where|when|why|how|which|is|are|do|does)", q_text.strip(), re.IGNORECASE):
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
    q_text = q_obj.get("question")
    if not is_valid_question_format(q_text):
        return False

    if q_obj.get("type") == "mcq":
        # Run MCQ-specific filters defensively
        if is_trivial_mcq(q_obj):
            return False
        if not has_plausible_distractors(q_obj):
            return False
    
    # You can add more filters here for short-answer questions later.
    
    return True