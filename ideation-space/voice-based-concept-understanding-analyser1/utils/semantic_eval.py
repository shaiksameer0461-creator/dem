import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

logger = logging.getLogger(__name__)

@st.cache_resource
def load_similarity_model():
    """
    Loads and caches the SentenceTransformer model.
    """
    logger.info("Loading SentenceTransformer 'all-MiniLM-L6-v2' (cached)...")
    m = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("SentenceTransformer model loaded successfully.")
    return m

model = load_similarity_model()

def calculate_similarity(reference_text, user_text):
    """
    Computes cosine similarity between reference_text and user_text using Sentence-BERT embeddings.
    Returns similarity score as a float between 0 and 100.
    """
    if not reference_text.strip() or not user_text.strip():
        logger.warning("Empty text provided for similarity — returning 0.0")
        return 0.0

    logger.info("Computing semantic similarity...")
    embeddings = model.encode([reference_text, user_text])

    score = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    # Clamp the score between 0.0 and 1.0, then convert to 0-100 scale
    score_clamped = max(0.0, min(1.0, float(score)))
    result = round(score_clamped * 100, 2)
    logger.info("Similarity score: %.2f%%", result)
    return result

def get_similarity(reference_text, user_text):
    """
    Alias for calculate_similarity to maintain compatibility with test scripts.
    """
    return calculate_similarity(reference_text, user_text)
