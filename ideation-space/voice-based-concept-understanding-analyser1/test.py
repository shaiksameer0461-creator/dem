from utils.speech_to_text import transcribe_audio
import sys

# Set standard output encoding to utf-8 if possible
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

audio_path = "audio/harvard.wav"

text = transcribe_audio(audio_path)

print("TRANSCRIPT:")
try:
    print(text)
except UnicodeEncodeError:
    print(text.encode('utf-8', errors='ignore').decode('utf-8'))
