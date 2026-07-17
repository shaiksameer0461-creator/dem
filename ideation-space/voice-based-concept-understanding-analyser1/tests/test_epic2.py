"""
Unit Tests — Epic 2 Modules
============================
Run with:  pytest tests/ -v
"""

import os
import sys
import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ═══════════════════════════════════════════════════════════════════════════════
# Task 1: speech_to_text.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpeechToText:
    """Tests for speech_to_text.py"""

    def test_validate_transcription_empty(self):
        from modules.speech_to_text import validate_transcription
        result = validate_transcription("")
        assert result["is_empty"] is True
        assert result["quality_flag"] == "empty"
        assert result["word_count"] == 0

    def test_validate_transcription_short(self):
        from modules.speech_to_text import validate_transcription
        result = validate_transcription("hello world")
        assert result["quality_flag"] == "short"
        assert result["word_count"] == 2

    def test_validate_transcription_good(self):
        from modules.speech_to_text import validate_transcription
        text = "Machine learning is a subset of AI that allows systems to learn from data."
        result = validate_transcription(text)
        assert result["quality_flag"] == "good"
        assert result["word_count"] == 14

    def test_validate_transcription_whitespace_only(self):
        from modules.speech_to_text import validate_transcription
        result = validate_transcription("   ")
        assert result["is_empty"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# Task 2: semantic_engine.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestSemanticEngine:
    """Tests for semantic_engine.py"""

    def test_get_available_concepts_returns_list(self):
        from modules.semantic_engine import get_available_concepts
        concepts = get_available_concepts()
        assert isinstance(concepts, list)
        assert len(concepts) > 0
        assert "Machine Learning" in concepts

    def test_generate_embedding_shape(self):
        from modules.semantic_engine import generate_embedding
        emb = generate_embedding("This is a test sentence.")
        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 1
        assert emb.shape[0] > 0

    def test_cosine_similarity_identical(self):
        from modules.semantic_engine import generate_embedding, compute_cosine_similarity
        emb = generate_embedding("Machine learning is a type of AI.")
        score = compute_cosine_similarity(emb, emb)
        assert abs(score - 1.0) < 1e-4

    def test_cosine_similarity_different(self):
        from modules.semantic_engine import generate_embedding, compute_cosine_similarity
        emb_a = generate_embedding("The cat sat on the mat.")
        emb_b = generate_embedding("Quantum entanglement describes a correlation between particles.")
        score = compute_cosine_similarity(emb_a, emb_b)
        # Dissimilar sentences should have much lower score
        assert score < 0.8

    def test_normalize_similarity_clips_negative(self):
        from modules.semantic_engine import normalize_similarity
        assert normalize_similarity(-0.5) == 0.0
        assert normalize_similarity(1.5) == 1.0
        assert normalize_similarity(0.7) == pytest.approx(0.7)

    def test_semantic_similarity_high_for_correct_explanation(self):
        from modules.semantic_engine import semantic_similarity
        good_explanation = (
            "Machine learning is a branch of artificial intelligence where computers "
            "learn from data without explicit programming. It includes supervised and "
            "unsupervised techniques."
        )
        result = semantic_similarity(good_explanation, "Machine Learning")
        assert result["normalized_score"] > 0.5
        assert result["concept"] == "Machine Learning"
        assert "percentage" in result

    def test_semantic_similarity_low_for_wrong_topic(self):
        from modules.semantic_engine import semantic_similarity
        off_topic = "The weather today is sunny with light winds from the north."
        result = semantic_similarity(off_topic, "Machine Learning")
        assert result["normalized_score"] < 0.5

    def test_semantic_similarity_custom_reference(self):
        from modules.semantic_engine import semantic_similarity
        ref = "Dogs are domesticated mammals often kept as pets."
        student = "A dog is a domestic animal that humans keep as a companion."
        result = semantic_similarity(student, "Dogs", custom_reference=ref)
        assert result["normalized_score"] > 0.6


# ═══════════════════════════════════════════════════════════════════════════════
# Task 3: audio_features.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestAudioFeatures:
    """Tests for audio_features.py"""

    # ── filler_word_ratio ─────────────────────────────────────────────────────

    def test_filler_word_ratio_no_fillers(self):
        from modules.audio_features import filler_word_ratio
        result = filler_word_ratio("The neural network processes data through layers.")
        assert result["filler_count"] == 0
        assert result["filler_percentage"] == 0.0

    def test_filler_word_ratio_with_fillers(self):
        from modules.audio_features import filler_word_ratio
        text = "So um machine learning is like basically um a subset of AI you know."
        result = filler_word_ratio(text)
        assert result["filler_count"] > 0
        assert result["filler_ratio"] > 0

    def test_filler_word_ratio_empty_string(self):
        from modules.audio_features import filler_word_ratio
        result = filler_word_ratio("")
        assert result["filler_count"] == 0
        assert result["total_words"] == 0
        assert result["filler_ratio"] == 0.0

    def test_filler_word_ratio_detected_fillers_sorted(self):
        from modules.audio_features import filler_word_ratio
        text = "um um um like like uh"
        result = filler_word_ratio(text)
        fillers = result["detected_fillers"]
        if len(fillers) > 1:
            # Should be sorted descending by count
            counts = [c for _, c in fillers]
            assert counts == sorted(counts, reverse=True)

    # ── evaluate_understanding ────────────────────────────────────────────────

    def test_evaluate_understanding_strong(self):
        from modules.audio_features import evaluate_understanding
        score, level, color = evaluate_understanding(
            semantic_score=0.90,
            filler_data={"filler_ratio": 0.0, "filler_percentage": 0.0},
            audio_features={"pause_ratio": 0.10, "rms_energy": 0.05},
        )
        assert score >= 0.75
        assert level == "Strong Understanding"
        assert color == "#28a745"

    def test_evaluate_understanding_moderate(self):
        from modules.audio_features import evaluate_understanding
        score, level, color = evaluate_understanding(
            semantic_score=0.55,
            filler_data={"filler_ratio": 0.08, "filler_percentage": 8.0},
            audio_features={"pause_ratio": 0.30, "rms_energy": 0.02},
        )
        assert 0.50 <= score < 0.75
        assert level == "Moderate Understanding"
        assert color == "#ffc107"

    def test_evaluate_understanding_poor(self):
        from modules.audio_features import evaluate_understanding
        score, level, color = evaluate_understanding(
            semantic_score=0.10,
            filler_data={"filler_ratio": 0.40, "filler_percentage": 40.0},
            audio_features={"pause_ratio": 0.70, "rms_energy": 0.001},
        )
        assert score < 0.50
        assert level == "Poor Understanding"
        assert color == "#dc3545"

    def test_evaluate_understanding_score_bounded(self):
        from modules.audio_features import evaluate_understanding
        # Even extreme inputs should stay within [0, 1]
        score, _, _ = evaluate_understanding(
            semantic_score=2.0,
            filler_data={"filler_ratio": -1.0, "filler_percentage": 0},
            audio_features={"pause_ratio": -0.5, "rms_energy": 100.0},
        )
        assert 0.0 <= score <= 1.0

    # ── get_feedback ──────────────────────────────────────────────────────────

    def test_get_feedback_returns_list(self):
        from modules.audio_features import get_feedback
        feedback = get_feedback(
            semantic_score=0.65,
            filler_data={"filler_ratio": 0.05, "filler_percentage": 5.0},
            audio_features={"pause_ratio": 0.25, "rms_energy": 0.03},
            level="Moderate Understanding",
        )
        assert isinstance(feedback, list)
        assert len(feedback) >= 3

    def test_get_feedback_no_fillers_positive(self):
        from modules.audio_features import get_feedback
        feedback = get_feedback(
            semantic_score=0.80,
            filler_data={"filler_ratio": 0.0, "filler_percentage": 0.0},
            audio_features={"pause_ratio": 0.10, "rms_energy": 0.04},
            level="Strong Understanding",
        )
        filler_feedback = [f for f in feedback if "filler" in f.lower()]
        assert any("No filler" in f or "Low filler" in f for f in filler_feedback)