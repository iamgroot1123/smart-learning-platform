import random
import re
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import logging
from textblob import TextBlob
import torch
from sentence_transformers import util
from nltk.corpus import wordnet
from src.question_generation.highlight import highlight_answer, _SPACY_AVAILABLE, _ensure_nlp

logger = logging.getLogger(__name__)

def get_semantic_distractors(
    correct_answer: str,
    context: str,
    sbert_model,
    num_candidates: int = 10
) -> List[str]:
    """
    Generate semantically similar distractors using SBERT.
    """
    # Get embeddings for the correct answer and context words
    words = [w for w in re.findall(r'\b\w+\b', context) if len(w) > 3]
    if not words:
        return []
    
    # Get embeddings
    correct_emb = sbert_model.encode([correct_answer], convert_to_tensor=True)
    words_emb = sbert_model.encode(words, convert_to_tensor=True)
    
    # Calculate similarities
    similarities = util.pytorch_cos_sim(correct_emb, words_emb)[0]
    
    # Get most similar but not identical words
    most_similar = []
    for idx in torch.argsort(similarities, descending=True):
        word = words[idx]
        if word.lower() != correct_answer.lower() and 0.3 < similarities[idx] < 0.9:
            most_similar.append(word)
            if len(most_similar) >= num_candidates:
                break
    
    return most_similar

def get_wordnet_distractors(word: str, pos: Optional[str] = None) -> List[str]:
    """
    Get distractors using WordNet relationships.
    """
    distractors = set()
    
    # Try different parts of speech if not specified
    pos_tags = [pos] if pos else ['n', 'v', 'a', 'r']
    
    for pos_tag in pos_tags:
        synsets = wordnet.synsets(word, pos=pos_tag)
        for syn in synsets:
            # Get words from synonyms
            for lemma in syn.lemmas():
                if lemma.name() != word:
                    distractors.add(lemma.name().replace('_', ' '))
            
            # Get words from hypernyms (more general terms)
            for hyper in syn.hypernyms():
                for lemma in hyper.lemmas():
                    distractors.add(lemma.name().replace('_', ' '))
            
            # Get words from hyponyms (more specific terms)
            for hypo in syn.hyponyms():
                for lemma in hypo.lemmas():
                    distractors.add(lemma.name().replace('_', ' '))
    
    return list(distractors)

def validate_question_quality(question: str) -> Tuple[bool, float]:
    """
    Validate question quality using multiple criteria.
    Returns (is_valid, quality_score).
    """
    if not question or not isinstance(question, str):
        return False, 0.0
    
    score = 0.0
    
    try:
        # Basic format checks
        if not question.endswith('?'):
            return False, 0.0
        
        # Question word check
        question_words = ['what', 'which', 'who', 'where', 'when', 'how', 'why']
        if any(question.lower().startswith(w) for w in question_words):
            score += 0.3
        
        # Length check
        words = question.split()
        if 5 <= len(words) <= 25:  # Good length
            score += 0.2
        elif len(words) > 25:  # Too long
            return False, 0.0
            
        # Check readability
        blob = TextBlob(question)
        # Avoid calling detect_language() because it may not be available or may require network access.
        # Instead, assume English for short questions and rely on other heuristics.
        try:
            if blob.sentiment.subjectivity < 0.5:  # More objective questions
                score += 0.2
        except Exception:
            pass
            
    except Exception as e:
        logger.warning(f"Error in question validation: {e}")
        return False, 0.0
    
    # Readability check using TextBlob (additional heuristics)
    try:
        blob2 = TextBlob(question)
        # Favor more objective questions
        if getattr(blob2, 'sentiment', None) and blob2.sentiment.subjectivity < 0.5:
            score += 0.1

        # Check if it starts with a Wh-word (question word)
        tags = getattr(blob2, 'tags', None)
        if tags and len(tags) > 0 and isinstance(tags[0], tuple) and tags[0][1].startswith('W'):
            score += 0.2
    except Exception:
        pass
    
    # Minimum quality threshold
    is_valid = score >= 0.5
    
    return is_valid, score

def validate_distractor_quality(
    correct: str,
    distractor: str,
    sbert_model
) -> Tuple[bool, float]:
    """
    Validate distractor quality using semantic similarity.
    Returns (is_valid, quality_score).
    """
    if not distractor or not isinstance(distractor, str):
        return False, 0.0
    
    # Length comparison
    if len(distractor) < 2 or len(distractor) > len(correct) * 3:
        return False, 0.0
    
    # Calculate semantic similarity
    try:
        embeddings = sbert_model.encode([correct, distractor], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings[0:1], embeddings[1:2])[0][0].item()
        
        # We want distractors that are related but not too similar
        # Too similar (>0.9) might be synonyms
        # Too different (<0.1) might be completely unrelated
        if 0.1 <= similarity <= 0.9:
            quality_score = 0.5 + (0.5 * (1 - abs(0.5 - similarity)))
            return True, quality_score
    except:
        pass
    
    return False, 0.0

def generate_enhanced_mcq(
    text: str,
    tokenizer,
    model,
    sbert_model,
    max_input_length: int = 512,
    num_distractors: int = 3,
    min_quality_score: float = 0.6
) -> Optional[Dict[str, Any]]:
    """
    Generate a high-quality MCQ with enhanced distractors and validation.
    Returns a dictionary with the following structure:
    {
        'type': 'mcq',
        'question': str,
        'options': List[str],
        'answer': str,
        'metadata': {
            'question_quality': float,
            'context_length': int,
            'has_semantic_distractors': bool,
            'has_wordnet_distractors': bool
        }
    }
    or None if generation fails
    """
    """
    Generate high-quality MCQ with enhanced distractors and validation.
    """
    try:
        # Clean text and highlight answer
        clean_text = re.sub(r"\[[^\]]+\.pdf\]\s*", "", text)
        highlighted = highlight_answer(clean_text, use_spacy=True)
        
        if "<answer>" not in highlighted:
            return None
            
        correct = highlighted.split("<answer>")[1].split("</answer>")[0]
        if not correct or len(correct.split()) > 5:  # Too long for MCQ answer
            return None
            
        # Generate question with improved context
        context = clean_text.replace(correct, "_____")
        input_text = f"generate multiple choice question for the gap in: {context}"
        
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=max_input_length, truncation=True)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            
        outputs = model.generate(
            inputs,
            max_length=64,
            num_beams=5,
            no_repeat_ngram_size=2,
            num_return_sequences=3,
            early_stopping=True
        )
        
        questions = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
        
        # Validate and select best question
        best_question = None
        best_score = -1
        
        for q in questions:
            is_valid, score = validate_question_quality(q)
            if is_valid and score > best_score:
                best_question = q
                best_score = score
        
        if not best_question:
            return None
        # Normalize question formatting
        best_question = best_question.strip()
        if not best_question.endswith('?'):
            best_question = best_question + '?'
        if best_score < min_quality_score:
            # allow a small relaxation: try to continue but mark lower quality
            logger.debug(f"Best question quality {best_score} below threshold {min_quality_score}")
            
        # Generate distractors using multiple strategies
        distractor_candidates = []
        
        # 1. SBERT semantic distractors
        semantic_distractors = get_semantic_distractors(correct, clean_text, sbert_model)
        distractor_candidates.extend(semantic_distractors)
        
        # 2. WordNet distractors
        wordnet_distractors = get_wordnet_distractors(correct)
        distractor_candidates.extend(wordnet_distractors)
        
        # 3. SpaCy entities and noun chunks if available
        if _SPACY_AVAILABLE:
            try:
                nlp = _ensure_nlp()
                doc = nlp(clean_text)
                
                # Add named entities of same type
                for ent in doc.ents:
                    if ent.text.strip().lower() != correct.lower():
                        distractor_candidates.append(ent.text.strip())
                
                # Add noun chunks
                for nc in doc.noun_chunks:
                    if nc.text.strip().lower() != correct.lower():
                        distractor_candidates.append(nc.text.strip())
            except:
                pass
        
        # Validate and score distractors
        valid_distractors = []
        for dist in set(distractor_candidates):
            is_valid, score = validate_distractor_quality(correct, dist, sbert_model)
            if is_valid:
                valid_distractors.append((dist, score))
        
        # Sort by quality score and take top N
        valid_distractors.sort(key=lambda x: x[1], reverse=True)
        final_distractors = [d[0] for d in valid_distractors[:num_distractors]]
        
        # If we don't have enough high-quality distractors, try to fill from raw candidates
        if len(final_distractors) < num_distractors:
            fallback = []
            for dist in distractor_candidates:
                if dist.lower() == correct.lower():
                    continue
                if dist in final_distractors:
                    continue
                is_valid, score = validate_distractor_quality(correct, dist, sbert_model)
                if is_valid:
                    fallback.append((dist, score))
                if len(final_distractors) + len(fallback) >= num_distractors:
                    break
            fallback.sort(key=lambda x: x[1], reverse=True)
            for d, _ in fallback:
                final_distractors.append(d)
                if len(final_distractors) >= num_distractors:
                    break

        if len(final_distractors) < num_distractors:
            # As a last resort, try simple lexical distractors (other nouns from context)
            extra = []
            for w in re.findall(r"\b\w{4,}\b", clean_text):
                if w.lower() == correct.lower():
                    continue
                if w in final_distractors or w in extra:
                    continue
                is_valid, score = validate_distractor_quality(correct, w, sbert_model)
                if is_valid:
                    extra.append(w)
                if len(final_distractors) + len(extra) >= num_distractors:
                    break
            final_distractors.extend(extra[:max(0, num_distractors - len(final_distractors))])

        if len(final_distractors) < num_distractors:
            return None
            
        # Create final options and shuffle
        options = [correct] + final_distractors
        random.shuffle(options)
        
        return {
            "type": "mcq",
            "question": best_question,
            "options": options,
            "answer": correct,
            "metadata": {
                "question_quality": best_score,
                "context_length": len(clean_text.split()),
                "has_semantic_distractors": len(semantic_distractors) > 0,
                "has_wordnet_distractors": len(wordnet_distractors) > 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating MCQ: {str(e)}")
        return None