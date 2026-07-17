import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import os
import logging

logger = logging.getLogger(__name__)


def load_audio(audio_path, sr=None):
    """
    Single entry-point for loading audio with librosa.
    Returns (y, sr). All other functions should call this instead of
    calling librosa.load() independently to avoid redundant I/O.
    """
    logger.info("Loading audio: %s (sr=%s)", audio_path, sr)
    y, sample_rate = librosa.load(audio_path, sr=sr)
    duration = librosa.get_duration(y=y, sr=sample_rate)
    logger.info("Audio loaded: %.2fs, %d samples, sr=%d", duration, len(y), sample_rate)
    return y, sample_rate


def extract_audio_features(audio_path, top_db=30, y=None, sr=None):
    """
    Extracts audio features including duration, RMS energy, spectral centroid (as average pitch),
    zero crossing rate, and pause ratio.

    If y and sr are provided, skips the redundant librosa.load() call.
    """
    if y is None or sr is None:
        y, sr = load_audio(audio_path)

    logger.info("Extracting audio features...")

    # Duration in seconds
    duration = librosa.get_duration(y=y, sr=sr)

    # Root Mean Square Energy
    rms = librosa.feature.rms(y=y)
    energy = float(np.mean(rms))

    # Spectral Centroid (used in original code as average pitch indicator)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    pitch = float(np.mean(spectral_centroid))

    # Zero Crossing Rate
    zcr_series = librosa.feature.zero_crossing_rate(y)
    zcr = float(np.mean(zcr_series))

    # Estimate pause ratio using librosa.effects.split
    # librosa.effects.split returns intervals of non-silent sections
    intervals = librosa.effects.split(y, top_db=top_db)
    active_samples = sum(end - start for start, end in intervals)
    total_samples = len(y)
    
    if total_samples > 0:
        pause_ratio = max(0.0, min(1.0, 1.0 - (active_samples / total_samples)))
    else:
        pause_ratio = 0.0

    features = {
        "Duration (seconds)": round(duration, 2),
        "Average Energy": round(energy, 4),
        "Average Pitch": round(pitch, 2),
        "Pause Ratio": round(pause_ratio, 4),
        "Zero Crossing Rate": round(zcr, 4)
    }
    logger.info("Audio features extracted: %s", features)
    return features


def get_scoring_features(audio_path, top_db=30, y=None, sr=None):
    """
    Returns a minimal dict with machine-readable keys for the scoring engine.
    Keys: pause_ratio (float), rms_energy (float).

    If y and sr are provided, skips the redundant librosa.load() call.
    """
    if y is None or sr is None:
        y, sr = load_audio(audio_path)

    logger.info("Computing scoring features...")

    # RMS energy
    rms = librosa.feature.rms(y=y)
    rms_energy = float(np.mean(rms))

    # Pause ratio via non-silent interval detection
    intervals = librosa.effects.split(y, top_db=top_db)
    active_samples = sum(end - start for start, end in intervals)
    total_samples = len(y)

    if total_samples > 0:
        pause_ratio = max(0.0, min(1.0, 1.0 - (active_samples / total_samples)))
    else:
        pause_ratio = 0.0

    result = {
        "pause_ratio": round(pause_ratio, 4),
        "rms_energy": round(rms_energy, 4),
    }
    logger.info("Scoring features: %s", result)
    return result


def save_waveform(audio_path, output_img_name="waveform.png", y=None, sr=None):
    """
    Plots the audio waveform with a neutral theme and saves it as an image file.
    Uses transparent background so it adapts to both light and dark modes.
    Returns the path to the saved image.

    If y and sr are provided, skips the redundant librosa.load() call.
    """
    if y is None or sr is None:
        y, sr = load_audio(audio_path, sr=None)

    logger.info("Generating waveform visualization...")
    
    # Neutral colors that work on both light and dark backgrounds
    wave_color = "#4a9eff"
    label_color = "#555555"
    grid_color = "#cccccc"
    
    # Create matplotlib figure with transparent background
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    
    # Plot waveform
    librosa.display.waveshow(y, sr=sr, ax=ax, color=wave_color)
    
    # Title and axis labels
    ax.set_title("Audio Waveform", color=label_color, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Time", color=label_color, fontsize=11)
    ax.set_ylabel("Amplitude", color=label_color, fontsize=11)
    
    # Style ticks and spines
    ax.tick_params(colors=label_color, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(grid_color)
    
    # Subtle grid
    ax.grid(True, color=grid_color, linewidth=0.4, alpha=0.4)
    
    # Save figure to reports/ directory
    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", output_img_name)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15, transparent=True)
    plt.close(fig)
    
    logger.info("Waveform saved to %s", out_path)
    return out_path
