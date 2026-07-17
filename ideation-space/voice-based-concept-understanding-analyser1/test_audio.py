from utils.audio_utils import extract_audio_features

audio_path = "audio/sample.mp3"

features = extract_audio_features(audio_path)

print("Audio Features")
print("----------------")

for key, value in features.items():
    print(f"{key}: {value}")