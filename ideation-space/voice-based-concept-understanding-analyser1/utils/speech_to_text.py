import os
import logging
import streamlit as st
import whisper
import librosa
import soundfile as sf
import tempfile

logger = logging.getLogger(__name__)

# Dynamically locate and add WinGet-installed Gyan.FFmpeg to path
user_profile = os.environ.get("USERPROFILE", "")
if user_profile:
    packages_dir = os.path.join(user_profile, "AppData", "Local", "Microsoft", "WinGet", "Packages")
    if os.path.exists(packages_dir):
        for item in os.listdir(packages_dir):
            if "Gyan.FFmpeg" in item:
                full_path = os.path.join(packages_dir, item)
                for root, dirs, files in os.walk(full_path):
                    if "ffmpeg.exe" in files:
                        if root not in os.environ["PATH"]:
                            os.environ["PATH"] = root + os.pathsep + os.environ["PATH"]
                        break

@st.cache_resource
def load_model():
    """
    Loads and caches the pre-trained Whisper model.
    """
    logger.info("Loading Whisper 'base' model (cached via @st.cache_resource)...")
    m = whisper.load_model("base")
    logger.info("Whisper model loaded successfully.")
    return m

# Load model (Streamlit caches it via @st.cache_resource decorator)
model = load_model()

def normalize_audio(audio_path):
    """
    Normalizes audio by converting different audio formats to a standard 16kHz, mono WAV file.
    This ensures consistent transcription accuracy across varied recording qualities.
    """
    # Load audio, automatically resampling to 16kHz and converting to mono channel
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # Save to a temporary standardized WAV file
    temp_fd, temp_wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(temp_fd) # Close file descriptor so other processes can write to it
    
    sf.write(temp_wav_path, y, sr)
    return temp_wav_path

def transcribe_audio(audio_path):
    """
    Transcribes audio file to text using the Whisper model after audio normalization.
    """
    logger.info("Starting transcription for: %s", audio_path)
    normalized_path = None
    try:
        # Step 1: Normalize audio to 16kHz mono WAV format
        normalized_path = normalize_audio(audio_path)
        logger.info("Audio normalized to: %s", normalized_path)
        
        # Step 2: Perform transcription on the standardized file
        result = model.transcribe(normalized_path)
        text = result.get("text", "").strip()

        if not text:
            logger.warning("No speech detected in audio.")
            return "⚠️ No speech detected in audio."

        logger.info("Transcription complete: %d characters", len(text))
        return text

    except FileNotFoundError:
        return "❌ Error: ffmpeg is not installed or not in the system PATH. OpenAI Whisper requires ffmpeg to transcribe audio. Please install ffmpeg and add it to your system PATH."
    except Exception as e:
        if "WinError 2" in str(e) or "cannot find the file specified" in str(e):
            return "❌ Error: ffmpeg is not installed or not in the system PATH. OpenAI Whisper requires ffmpeg to transcribe audio. Please install ffmpeg and add it to your system PATH."
        return f"❌ Error: {str(e)}"
    finally:
        # Clean up temporary normalized file
        if normalized_path and os.path.exists(normalized_path):
            try:
                os.remove(normalized_path)
            except Exception:
                pass

def speech_to_text(audio_path):
    """
    Integrates OpenAI Whisper to convert uploaded audio files into accurate textual transcriptions.
    Alias/wrapper to match workflow specifications.
    """
    return transcribe_audio(audio_path)
