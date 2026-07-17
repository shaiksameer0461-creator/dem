"""
Task 2: Semantic Understanding & Similarity Engine
---------------------------------------------------
Uses Sentence-BERT to generate embeddings for student explanations and
reference concepts, then computes cosine similarity to quantify alignment.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ── Pre-defined reference concepts ──────────────────────────────────────────
# Each key maps to one or more canonical reference sentences that a student
# should cover when explaining that concept.

REFERENCE_CONCEPTS: dict[str, str] = {
    "Machine Learning": (
        "Machine learning is a subset of artificial intelligence that enables systems "
        "to learn and improve from experience without being explicitly programmed. "
        "It focuses on developing computer programs that can access data and use it to "
        "learn for themselves. Key techniques include supervised learning, unsupervised "
        "learning, and reinforcement learning."
    ),
    "Cloud Computing": (
        "Cloud computing is the on-demand availability of computer system resources, "
        "especially data storage and computing power, without direct active management "
        "by the user. It provides services over the internet including servers, storage, "
        "databases, networking, software, and analytics."
    ),
    "Deep Learning": (
        "Deep learning is a subset of machine learning that uses neural networks with "
        "many layers to learn representations of data. It automatically learns features "
        "from raw data and is particularly effective for image recognition, natural "
        "language processing, and speech recognition."
    ),
    "Artificial Intelligence": (
        "Artificial intelligence is the simulation of human intelligence processes by "
        "computer systems. These processes include learning, reasoning, problem-solving, "
        "perception, and language understanding. AI applications include expert systems, "
        "natural language processing, and computer vision."
    ),
    "Neural Networks": (
        "Neural networks are computing systems inspired by biological neural networks in "
        "animal brains. They consist of layers of interconnected nodes that process "
        "information using connectionist approaches. They learn by adjusting the weights "
        "of connections based on training data."
    ),
    "Natural Language Processing": (
        "Natural language processing is a branch of artificial intelligence that helps "
        "computers understand, interpret, and manipulate human language. It bridges the "
        "gap between human communication and computer understanding through tasks like "
        "sentiment analysis, translation, and text classification."
    ),
    "Data Science": (
        "Data science is an interdisciplinary field that uses scientific methods, "
        "processes, algorithms, and systems to extract knowledge and insights from "
        "structured and unstructured data. It combines statistics, mathematics, "
        "programming, and domain expertise."
    ),
    "Blockchain": (
        "Blockchain is a distributed ledger technology that records transactions across "
        "many computers in a way that is secure, transparent, and tamper-resistant. "
        "Each block contains transaction data and a cryptographic hash of the previous "
        "block, forming an immutable chain."
    ),
    "Internet of Things": (
        "The Internet of Things refers to the network of physical devices embedded with "
        "sensors, software, and connectivity that enables them to collect and exchange "
        "data. It connects everyday objects to the internet, enabling smart homes, "
        "industrial automation, and real-time monitoring."
    ),
    "Cybersecurity": (
        "Cybersecurity is the practice of protecting systems, networks, and programs "
        "from digital attacks. It encompasses measures to prevent unauthorized access, "
        "data breaches, and cyberattacks through encryption, firewalls, authentication, "
        "and security protocols."
    ),
}


# ── Model cache ──────────────────────────────────────────────────────────────
_sbert_model: SentenceTransformer | None = None


def _load_sbert(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load and cache the Sentence-BERT model."""
    global _sbert_model
    if _sbert_model is None:
        _sbert_model = SentenceTransformer(model_name)
    return _sbert_model


# ── Embedding generation ─────────────────────────────────────────────────────

def generate_embedding(text: str) -> np.ndarray:
    """
    Generate a Sentence-BERT embedding for the given text.

    Args:
        text: Input text (student explanation or reference concept).

    Returns:
        1-D numpy array (embedding vector).
    """
    model = _load_sbert()
    embedding = model.encode([text], convert_to_numpy=True, show_progress_bar=False)
    return embedding[0]


# ── Similarity computation ────────────────────────────────────────────────────

def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Returns:
        Float in [0, 1].
    """
    vec_a = vec_a.reshape(1, -1)
    vec_b = vec_b.reshape(1, -1)
    score = cosine_similarity(vec_a, vec_b)[0][0]
    return float(score)


def normalize_similarity(raw_score: float) -> float:
    """
    Normalize cosine similarity to [0, 1] for consistent cross-evaluation
    interpretation. Clips values below 0 to 0.

    Args:
        raw_score: Raw cosine similarity (may be slightly negative).

    Returns:
        Normalized float in [0, 1].
    """
    return float(np.clip(raw_score, 0.0, 1.0))


# ── High-level API ────────────────────────────────────────────────────────────

def semantic_similarity(
    student_transcript: str,
    concept: str,
    custom_reference: str | None = None,
) -> dict:
    """
    Compute semantic similarity between a student's transcript and a reference concept.

    Args:
        student_transcript: Transcribed student explanation.
        concept:            Topic name (key in REFERENCE_CONCEPTS).
        custom_reference:   Optional custom reference text (overrides built-in).

    Returns:
        dict with keys:
            - raw_score        (float)  : raw cosine similarity
            - normalized_score (float)  : clipped to [0, 1]
            - percentage       (float)  : score * 100, rounded to 1 dp
            - reference_used   (str)    : which reference text was used
            - concept          (str)    : topic name
    """
    reference_text = custom_reference or REFERENCE_CONCEPTS.get(
        concept,
        concept,   # fallback: use the concept name itself as reference
    )

    student_emb = generate_embedding(student_transcript)
    reference_emb = generate_embedding(reference_text)

    raw = compute_cosine_similarity(student_emb, reference_emb)
    normalized = normalize_similarity(raw)

    return {
        "raw_score": raw,
        "normalized_score": normalized,
        "percentage": round(normalized * 100, 1),
        "reference_used": reference_text,
        "concept": concept,
    }


def get_available_concepts() -> list[str]:
    """Return the list of built-in reference concept names."""
    return sorted(REFERENCE_CONCEPTS.keys())