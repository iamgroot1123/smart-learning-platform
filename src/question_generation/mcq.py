import re
import pandas as pd
from gensim.models import KeyedVectors
import random
import os
from nltk.corpus import wordnet

# Placeholder for GloVe model path - User needs to provide the actual path or ensure the file exists
# The 'data/embeddings' directory needs to be created and 'glove.6B.300d.txt' placed inside it.
GLOVE_DIR = '/home/madhukiran/Desktop/mini_project/smart-learning-platform/data/embeddings'
GLOVE_FILE = os.path.join(GLOVE_DIR, 'glove.6B.300d.txt')

model = None
# Ensure the 'data/embeddings' directory exists
os.makedirs(GLOVE_DIR, exist_ok=True)

# Load GloVe embeddings directly
try:
    model = KeyedVectors.load_word2vec_format(GLOVE_FILE, binary=False, no_header=True)
    print("GloVe model loaded successfully.")
except FileNotFoundError:
    print(f"GloVe file not found at {GLOVE_FILE}. Please ensure the file exists and is correctly placed.")
except Exception as e:
    print(f"Error loading GloVe model: {e}")

def generate_distractors(answer, count=3):
    """
    Generate distractors for an answer word using GloVe nearest neighbors.
    Falls back to WordNet synonyms if GloVe is not available or returns poor candidates.
    Filters out non-alphabetic/gibberish tokens and ensures we return up to `count` items.
    """
    answer_str = str(answer).lower().strip()

    candidates = []

    # Primary: use GloVe model if available
    if model is not None:
        try:
            closest = model.most_similar(positive=[answer_str], topn=count * 10)
            candidates.extend([w for w, _ in closest])
        except Exception:
            # ignore model lookup failures and fallback
            pass

    # Secondary: WordNet synonyms + hypernyms + hyponyms
    try:
        for syn in wordnet.synsets(answer_str):
            # synonyms
            for lemma in syn.lemmas():
                candidates.append(lemma.name().replace('_', ' '))
            # hypernyms
            for h in syn.hypernyms():
                for lemma in h.lemmas():
                    candidates.append(lemma.name().replace('_', ' '))
            # hyponyms
            for h in syn.hyponyms():
                for lemma in h.lemmas():
                    candidates.append(lemma.name().replace('_', ' '))
    except Exception:
        pass

    # Final fallback: simple morphological variants
    morphs = [answer_str + 's', answer_str + 'es', answer_str + 'ing']
    candidates.extend(morphs)

    # Clean candidates: prefer multi-letter alphabetic words and remove gibberish
    seen = set()
    distractors = []
    for cand in candidates:
        cand_clean = str(cand).strip()
        # reject if same as answer
        if not cand_clean or cand_clean.lower() == answer_str:
            continue
        # remove tokens with digits or many non-alpha chars
        if re.search(r'[^a-zA-Z\s-]', cand_clean):
            continue
        # normalize spacing and lowercase for uniqueness
        cand_norm = re.sub(r"\s+", ' ', cand_clean).lower()
        if cand_norm in seen:
            continue
        # require reasonable length (>=3 characters) and alphabetic
        if len(cand_norm) < 3 or not re.match(r'^[a-zA-Z\s-]+$', cand_norm):
            continue
        seen.add(cand_norm)
        distractors.append(cand_clean)
        if len(distractors) >= count:
            break

    return distractors


def convert_blank_to_question(blank_sentence: str) -> str:
    """
    Convert a sentence with a blank ("_____") into a simple WH-question.
    Example: "a _____ detects changes in the environment." -> "What detects changes in the environment?"
    This is a heuristic approach that covers common sentence shapes.
    """
    s = blank_sentence.strip()
    # remove trailing punctuation
    s = s.rstrip('.!?')
    if '_____' not in s:
        # fallback: ensure it ends with a question mark
        return s + ('?' if not s.endswith('?') else '')

    before, after = s.split('_____', 1)
    after = after.strip()

    # Prefer using the text after the blank to form the question
    if after:
        question = f"What {after}"
    else:
        # If nothing after, try to use the part before the blank
        before = before.strip()
        # remove leading articles from before part
        before = re.sub(r'^(a|an|the)\s+', '', before)
        question = f"What {before}"

    question = question.strip()
    if not question.endswith('?'):
        question = question + '?'

    # Capitalize first letter
    if question:
        question = question[0].upper() + question[1:]

    return question


def blank_key_in_sentence(original_sentence: str, key: str) -> str:
    """
    Replace the first occurrence of `key` (case-insensitive, word-boundary) in the
    original sentence with a blank '_____' while preserving the rest of the sentence.
    Returns the blanked sentence (original casing preserved).
    """
    if not original_sentence or not key:
        return original_sentence

    # Regex for word-boundaries, case-insensitive
    pattern = re.compile(r'\b' + re.escape(key) + r'\b', flags=re.IGNORECASE)
    # Replace only the first occurrence
    new_sentence, nsub = pattern.subn('_____', original_sentence, count=1)
    return new_sentence if nsub > 0 else original_sentence


def is_noisy_sentence(s: str) -> bool:
    """Return True if sentence looks noisy (contains equations, many non-alpha chars, or many single-letter tokens)."""
    if not s or not isinstance(s, str):
        return True
    # lots of non-alpha characters
    non_alpha = re.sub(r'[A-Za-z\s]', '', s)
    if len(non_alpha) / max(1, len(s)) > 0.15:
        return True
    # contains equation-like patterns
    if re.search(r'[=<>+-/*\\^]', s):
        return True
    # parentheses with many non-alpha inside
    if re.search(r'\([^a-zA-Z]{1,}\)', s):
        return True
    # too many single-letter tokens (common in OCR noise)
    tokens = s.split()
    single_letters = sum(1 for t in tokens if len(t) == 1)
    if len(tokens) > 0 and (single_letters / len(tokens)) > 0.25:
        return True
    return False

def generate_mcq_with_options(summarized_text, filtered_keys, keyword_sentence_mapping):
    # Use lowercase column names for consistency with filters and downstream code
    mcq_question_options = pd.DataFrame(columns=["Keywords", "question", "options", "answer"])

    if not model:
        print("GloVe model not loaded. Cannot generate MCQs.")
        return mcq_question_options

    for key in filtered_keys:
        if key in keyword_sentence_mapping:
            for sentence in keyword_sentence_mapping[key]:
                # Use the original sentence casing when creating the blank
                original_sentence = sentence
                question_stem = blank_key_in_sentence(original_sentence, key)

                if '_____' in question_stem:
                    # If sentence looks noisy (equations/garbled OCR), fall back to a template question
                    noisy = is_noisy_sentence(question_stem)

                    # Generate distractors using the (lowercased) key as canonical form
                    distractors = generate_distractors(key, 3)

                    # If distractors are insufficient, try to supplement from other filtered keys
                    if len(distractors) < 3:
                        for fk in filtered_keys:
                            if fk != key and fk not in distractors:
                                distractors.append(fk)
                            if len(distractors) >= 3:
                                break

                    options = [key] + distractors[:3]
                    # Clean options: remove non-alpha and strip
                    clean_options = []
                    for opt in options:
                        if not isinstance(opt, str):
                            continue
                        opt_clean = opt.strip()
                        # drop obviously bad tokens
                        if re.search(r'[^a-zA-Z\s-]', opt_clean):
                            continue
                        if len(opt_clean) < 2:
                            continue
                        # Capitalize nicely
                        opt_clean = opt_clean.capitalize()
                        if opt_clean not in clean_options:
                            clean_options.append(opt_clean)

                    # Ensure the answer is among options and present in proper form
                    answer_label = key.capitalize() if isinstance(key, str) else str(key)
                    if answer_label not in clean_options:
                        clean_options = [answer_label] + clean_options

                    # Limit to 4 options max
                    final_options = clean_options[:4]
                    random.shuffle(final_options)

                    # Build the question text. If the sentence is noisy, use a simple template question
                    if noisy:
                        question_text = f"What is {key.capitalize()}?"
                    else:
                        question_text = convert_blank_to_question(question_stem)

                    new_row = pd.DataFrame([{
                        "Keywords": key,
                        "question": question_text,
                        "options": final_options,
                        "answer": answer_label
                    }])
                    mcq_question_options = pd.concat([mcq_question_options, new_row], ignore_index=True)

    output_path = '/home/madhukiran/Desktop/mini_project/smart-learning-platform/mcq_questions.xlsx'
    try:
        mcq_question_options.to_excel(output_path, index=False)
        print(f"MCQs saved to {output_path}")
    except Exception as e:
        print(f"Error saving MCQs to Excel: {e}")

    return mcq_question_options

if __name__ == "__main__":
    sample_summarized_text = "A sensor detects changes in the environment. Transducers convert energy. Receivers get signals."
    sample_filtered_keys = ["sensor", "transducer", "receiver"]
    sample_keyword_sentence_mapping = {
        "sensor": ["A sensor detects changes in the environment."],
        "transducer": ["Transducers convert energy."],
        "receiver": ["Receivers get signals."]
    }

    print("Running example MCQ generation...")
    # generate_mcq_with_options(sample_summarized_text, sample_filtered_keys, sample_keyword_sentence_mapping)
    print("Example MCQ generation finished. Check 'mcq_questions.xlsx' if GloVe file was present.")
