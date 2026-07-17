"""
Task 3: Audio Feature Extraction & Scoring Engine
--------------------------------------------------
Extracts audio-level features (pause ratio, RMS energy) using Librosa and
SoundFile. Detects filler words from transcribed text. Combines all metrics
into a final understanding score with qualitative classification.
"""

from __future__ import annotations

import re
import numpy as np
import librosa
import soundfile as sf


# ── Filler word vocabulary ────────────────────────────────────────────────────

FILLER_WORDS: set[str] = {
    "um", "uh", "er", "ah", "like", "you know", "so", "basically",
    "actually", "literally", "kind of", "sort of", "right", "okay",
    "hmm", "well", "i mean", "you see", "anyway", "whatever",
}


# ── Audio Feature Extraction ──────────────────────────────────────────────────

def extract_audio_features(audio_path: str) -> dict:
    """
    Extract low-level acoustic features from an audio file.

    Features extracted:
        - duration_sec    : total audio length in seconds
        - rms_energy      : mean RMS energy (proxy for loudness / confidence)
        - rms_std         : std-dev of RMS (speaking variability)
        - pause_ratio     : fraction of frames considered silent
        - speaking_ratio  : 1 - pause_ratio
        - zero_crossing   : mean zero-crossing rate (high = noisy / fricatives)
        - spectral_centroid_mean : brightness / clarity of speech

    Args:
        audio_path: Path to audio file (WAV recommended).

    Returns:
        dict of feature name → value.
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    duration_sec = float(len(y) / sr)

    # ── RMS energy ───────────────────────────────────────────────────────────
    frame_len = int(sr * 0.025)   # 25 ms frames
    hop_len   = int(sr * 0.010)   # 10 ms hop

    rms_frames = librosa.feature.rms(
        y=y, frame_length=frame_len, hop_length=hop_len
    )[0]
    rms_energy = float(np.mean(rms_frames))
    rms_std    = float(np.std(rms_frames))

    # ── Pause detection via silence threshold ────────────────────────────────
    silence_threshold = max(rms_energy * 0.15, 1e-4)
    silent_frames     = np.sum(rms_frames < silence_threshold)
    pause_ratio       = float(silent_frames / len(rms_frames)) if len(rms_frames) > 0 else 0.0
    speaking_ratio    = 1.0 - pause_ratio

    # ── Zero crossing rate ────────────────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=frame_len, hop_length=hop_len
    )[0]
    zero_crossing = float(np.mean(zcr))

    # ── Spectral centroid ─────────────────────────────────────────────────────
    spec_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=hop_len
    )[0]
    spectral_centroid_mean = float(np.mean(spec_centroid))

    return {
        "duration_sec":           round(duration_sec, 2),
        "rms_energy":             round(rms_energy, 6),
        "rms_std":                round(rms_std, 6),
        "pause_ratio":            round(pause_ratio, 4),
        "speaking_ratio":         round(speaking_ratio, 4),
        "zero_crossing":          round(zero_crossing, 6),
        "spectral_centroid_mean": round(spectral_centroid_mean, 2),
    }


# ── Filler Word Detection ─────────────────────────────────────────────────────

def filler_word_ratio(transcript: str) -> dict:
    """
    Detect filler words in the transcribed text and compute filler ratio.

    Args:
        transcript: Raw transcript string from STT.

    Returns:
        dict with:
            - filler_count        : total filler occurrences
            - total_words         : total word count
            - filler_ratio        : filler_count / total_words
            - filler_percentage   : percentage (0–100)
            - detected_fillers    : list of (filler_word, count) tuples
    """
    if not transcript:
        return {
            "filler_count": 0,
            "total_words": 0,
            "filler_ratio": 0.0,
            "filler_percentage": 0.0,
            "detected_fillers": [],
        }

    text_lower = transcript.lower()
    words = re.findall(r"\b\w+\b", text_lower)
    total_words = len(words)

    # Count single-word fillers
    filler_counts: dict[str, int] = {}
    for word in words:
        if word in FILLER_WORDS:
            filler_counts[word] = filler_counts.get(word, 0) + 1

    # Count multi-word fillers (e.g., "you know", "kind of")
    multi_word_fillers = [f for f in FILLER_WORDS if " " in f]
    for mf in multi_word_fillers:
        count = len(re.findall(r"\b" + re.escape(mf) + r"\b", text_lower))
        if count > 0:
            filler_counts[mf] = filler_counts.get(mf, 0) + count

    filler_count = sum(filler_counts.values())
    ratio = filler_count / total_words if total_words > 0 else 0.0

    return {
        "filler_count": filler_count,
        "total_words": total_words,
        "filler_ratio": round(ratio, 4),
        "filler_percentage": round(ratio * 100, 1),
        "detected_fillers": sorted(filler_counts.items(), key=lambda x: -x[1]),
    }


# ── Combined Scoring Engine ───────────────────────────────────────────────────

def evaluate_understanding(
    semantic_score: float,
    filler_data: dict,
    audio_features: dict,
) -> tuple[float, str, str]:
    """
    Generate a final understanding score by combining three signal groups:

    1. Semantic similarity  → 60 % weight
    2. Filler word penalty  → 20 % weight (inverse: fewer fillers = higher score)
    3. Audio confidence     → 20 % weight (pause ratio + RMS energy)

    Classification:
        ≥ 0.75  → Strong Understanding
        ≥ 0.50  → Moderate Understanding
        <  0.50 → Poor Understanding

    Args:
        semantic_score  : normalized cosine similarity [0, 1]
        filler_data     : output of filler_word_ratio()
        audio_features  : output of extract_audio_features()

    Returns:
        (score: float, level: str, color: str)
            score in [0, 1], level is the qualitative label, color is hex.
    """
    # ── Component 1: Semantic score (60 %) ───────────────────────────────────
    semantic_component = semantic_score * 0.60

    # ── Component 2: Filler penalty (20 %) ───────────────────────────────────
    filler_ratio = filler_data.get("filler_ratio", 0.0)
    # 0 fillers → 1.0 score; ≥30 % fillers → 0.0 score (linear clamp)
    filler_score = max(0.0, 1.0 - (filler_ratio / 0.30))
    filler_component = filler_score * 0.20

    # ── Component 3: Audio confidence (20 %) ─────────────────────────────────
    pause_ratio    = audio_features.get("pause_ratio", 0.5)
    rms_energy     = audio_features.get("rms_energy", 0.01)

    # Pause score: ≤20 % pauses → 1.0; ≥70 % → 0.0
    pause_score = max(0.0, min(1.0, (0.70 - pause_ratio) / 0.50))

    # RMS score: normalize using a reasonable speech energy range
    rms_score = min(1.0, rms_energy / 0.05)  # 0.05 ≈ comfortable speaking level

    audio_confidence = (pause_score * 0.6 + rms_score * 0.4)
    audio_component  = audio_confidence * 0.20

    # ── Final score ───────────────────────────────────────────────────────────
    final_score = semantic_component + filler_component + audio_component
    final_score = float(np.clip(final_score, 0.0, 1.0))

    # ── Classification ────────────────────────────────────────────────────────
    if final_score >= 0.75:
        level = "Strong Understanding"
        color = "#28a745"   # green
    elif final_score >= 0.50:
        level = "Moderate Understanding"
        color = "#ffc107"   # amber
    else:
        level = "Poor Understanding"
        color = "#dc3545"   # red

    return round(final_score, 4), level, color


def get_feedback(
    semantic_score: float,
    filler_data: dict,
    audio_features: dict,
    level: str,
) -> list[str]:
    """
    Generate a list of human-readable feedback bullet points.

    Args:
        semantic_score  : normalized similarity score
        filler_data     : output of filler_word_ratio()
        audio_features  : output of extract_audio_features()
        level           : classification label from evaluate_understanding()

    Returns:
        List of feedback strings.
    """
    feedback: list[str] = []

    # Semantic feedback
    if semantic_score >= 0.75:
        feedback.append("✅ Excellent conceptual alignment — your explanation covers the core ideas well.")
    elif semantic_score >= 0.50:
        feedback.append("⚠️  Moderate conceptual coverage — consider elaborating on key aspects of the topic.")
    else:
        feedback.append("❌ Low conceptual alignment — review the core definition and main components of the topic.")

    # Filler feedback
    filler_pct = filler_data.get("filler_percentage", 0.0)
    if filler_pct == 0:
        feedback.append("✅ No filler words detected — excellent fluency!")
    elif filler_pct < 5:
        feedback.append(f"✅ Low filler word usage ({filler_pct:.1f}%) — good communication fluency.")
    elif filler_pct < 15:
        feedback.append(f"⚠️  Moderate filler usage ({filler_pct:.1f}%) — try to reduce hesitation words.")
    else:
        feedback.append(f"❌ High filler word usage ({filler_pct:.1f}%) — practice structured speech to improve fluency.")

    # Pause feedback
    pause_ratio = audio_features.get("pause_ratio", 0)
    if pause_ratio < 0.20:
        feedback.append("✅ Good speaking pace — minimal pausing detected.")
    elif pause_ratio < 0.40:
        feedback.append(f"⚠️  Noticeable pauses ({pause_ratio*100:.0f}% of audio) — consider improving speech flow.")
    else:
        feedback.append(f"❌ Excessive pausing ({pause_ratio*100:.0f}% of audio) — work on delivery confidence.")

    # RMS / energy feedback
    rms = audio_features.get("rms_energy", 0)
    if rms >= 0.03:
        feedback.append("✅ Clear and confident vocal energy detected.")
    elif rms >= 0.01:
        feedback.append("⚠️  Speaking volume is moderate — try to project your voice more clearly.")
    else:
        feedback.append("❌ Very low audio energy — ensure microphone quality and speaking volume.")

    return feedback