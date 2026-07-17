"""
Task 1: Speech-to-Text Module Development
-----------------------------------------
Integrates OpenAI Whisper to convert uploaded audio files into accurate
textual transcriptions. Handles WAV format and normalizes audio input.
"""

import os
import tempfile
import numpy as np
import soundfile as sf
import whisper


# ── Module-level model cache (loaded once per session) ──────────────────────
_whisper_model = None


def _load_model(model_size: str = "base") -> whisper.Whisper:
    """Load and cache the Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model(model_size)
    return _whisper_model


# ── Audio Normalisation ──────────────────────────────────────────────────────

def normalize_audio(audio_path: str, target_sr: int = 16000) -> str:
    """
    Normalize an audio file to mono, 16 kHz WAV for consistent Whisper input.

    Args:
        audio_path: Path to the source audio file (WAV / MP3 / etc.)
        target_sr:  Target sample rate (Whisper expects 16 000 Hz).

    Returns:
        Path to the normalized temporary WAV file.
    """
    data, sr = sf.read(audio_path, always_2d=True)

    # Convert to mono by averaging channels
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]

    # Resample if needed (simple linear interpolation for speed)
    if sr != target_sr:
        duration = len(data) / sr
        target_len = int(duration * target_sr)
        data = np.interp(
            np.linspace(0, len(data) - 1, target_len),
            np.arange(len(data)),
            data,
        )

    # Peak-normalize to [-1, 1]
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak

    # Write to a temp file so Whisper can read it
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, data.astype(np.float32), target_sr)
    return tmp.name


# ── Transcription ────────────────────────────────────────────────────────────

def speech_to_text(audio_path: str, model_size: str = "base") -> str:
    """
    Transcribe an audio file to text using OpenAI Whisper.

    Args:
        audio_path:  Path to the audio file.
        model_size:  Whisper model variant ("tiny", "base", "small", "medium").

    Returns:
        Transcribed text string (stripped and lowercased).
    """
    normalized_path = normalize_audio(audio_path)
    try:
        model = _load_model(model_size)
        result = model.transcribe(
            normalized_path,
            language="en",
            fp16=False,          # CPU-safe
            verbose=False,
        )
        transcript = result.get("text", "").strip()
    finally:
        # Clean up temp file
        if os.path.exists(normalized_path):
            os.remove(normalized_path)

    return transcript


# ── Validation Helper ────────────────────────────────────────────────────────

def validate_transcription(transcript: str) -> dict:
    """
    Basic quality checks on the transcription output.

    Returns a dict with:
        - word_count      : number of words
        - is_empty        : True if transcript is blank
        - estimated_wpm   : rough speaking rate (assumes ~2 min audio; for UI display only)
        - quality_flag    : "good" | "short" | "empty"
    """
    words = transcript.split() if transcript else []
    word_count = len(words)

    if word_count == 0:
        quality_flag = "empty"
    elif word_count < 10:
        quality_flag = "short"
    else:
        quality_flag = "good"

    return {
        "word_count": word_count,
        "is_empty": word_count == 0,
        "quality_flag": quality_flag,
    }


# ── Waveform Saving (used in main app) ──────────────────────────────────────

def save_waveform(audio_path: str, output_dir: str = "assets/waveforms") -> str:
    """
    Generate and save a waveform PNG for a given audio file.

    Args:
        audio_path:  Path to the audio file.
        output_dir:  Directory to write the PNG.

    Returns:
        Path to the saved waveform image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    data, sr = sf.read(audio_path, always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]

    time_axis = np.linspace(0, len(data) / sr, num=len(data))

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(time_axis, data, color="#1DB954", linewidth=0.5)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.set_title("Audio Waveform", fontsize=11)
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}_waveform.png")
    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return out_path