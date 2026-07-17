import re
import logging

logger = logging.getLogger(__name__)

def calculate_filler_word_stats(text):
    """
    Analyzes user transcript text to count filler words/phrases,
    and returns a stats dictionary.
    """
    if not text or not text.strip():
        return {
            "filler_word_count": 0,
            "total_words": 0,
            "filler_ratio": 0.0
        }
    
    # Clean punctuation for total word count calculation
    cleaned_text = re.sub(r'[^\w\s\']', '', text.lower())
    words = cleaned_text.split()
    total_words = len(words)
    
    # Common English filler words and short phrases
    filler_pattern = r'\b(uh|um|ah|like|so|basically|actually|literally|you know)\b'
    fillers_found = re.findall(filler_pattern, text.lower())
    filler_word_count = len(fillers_found)
    
    if total_words > 0:
        filler_ratio = round(filler_word_count / total_words, 4)
    else:
        filler_ratio = 0.0
        
    return {
        "filler_word_count": filler_word_count,
        "total_words": total_words,
        "filler_ratio": filler_ratio
    }

def evaluate_understanding(similarity, filler_ratio, audio):
    """
    Computes a composite understanding score (0–100) by combining:
      - Semantic similarity (0–1 scale): up to 50 points
      - Filler word ratio: up to 20 points
      - Pause ratio: up to 15 points
      - RMS energy: up to 15 points

    Returns:
        tuple: (score, label, color_hex)
    """
    logger.info("Evaluating understanding — similarity=%.4f, filler_ratio=%.4f, audio=%s",
                similarity, filler_ratio, audio)
    score = 0
    score += 50 if similarity > 0.7 else 30 if similarity > 0.4 else 10
    score += 20 if filler_ratio < 0.05 else 10
    score += 15 if audio["pause_ratio"] < 0.25 else 5
    score += 15 if audio["rms_energy"] > 0.01 else 5
    if score >= 80:
        logger.info("Result: score=%d, level=Strong Understanding", score)
        return score, "Strong Understanding", "#2ecc71"
    elif score >= 50:
        logger.info("Result: score=%d, level=Moderate Understanding", score)
        return score, "Moderate Understanding", "#f39c12"
    logger.info("Result: score=%d, level=Poor Understanding", score)
    return score, "Poor Understanding", "#e74c3c"
