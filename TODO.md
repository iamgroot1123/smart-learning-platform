# TODO: Implement Short Answer Evaluation

- [x] Modify generate_short_question in src/question_generation/short.py to generate both question and correct answer using T5.
- [x] Update run.py to include correct answers in short question data.
- [x] Add a new Flask route /evaluate_short to handle evaluation.
- [x] Implement evaluation function: preprocess text (lowercase, remove stopwords, lemmatize), generate embeddings with all-MiniLM-L6-v2, compute cosine similarity, map to percentage, apply thresholds (≥0.85 correct, 0.65-0.85 partial, <0.65 incorrect).
- [x] Update static/js/script.js to POST user answer to /evaluate_short and display feedback with colors (green/orange/red).
- [x] Update templates/index.html if needed for feedback display.
- [ ] Test the evaluation with sample answers.
- [ ] Adjust thresholds if needed.
