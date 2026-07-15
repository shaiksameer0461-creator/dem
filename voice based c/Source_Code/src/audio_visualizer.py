import os
import tempfile
import librosa
import librosa.display
import matplotlib.pyplot as plt


class AudioVisualizer:

    def create_waveform(self, uploaded_file):

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:

            temp.write(uploaded_file.getbuffer())

            temp_path = temp.name

        # Load audio
        y, sr = librosa.load(temp_path, sr=None)

        # Create waveform
        fig, ax = plt.subplots(figsize=(12, 3))

        librosa.display.waveshow(
            y,
            sr=sr,
            ax=ax,
            color="#1f77b4"
        )

        ax.set_title("Uploaded Audio Waveform")

        ax.set_xlabel("Time (seconds)")

        ax.set_ylabel("Amplitude")

        plt.tight_layout()

        return fig


    def save_waveform(self, audio_path):

        os.makedirs("assets", exist_ok=True)

        waveform_path = os.path.join(
            "assets",
            "waveform.png"
        )

        y, sr = librosa.load(audio_path, sr=None)

        plt.figure(figsize=(10,3))

        librosa.display.waveshow(
            y,
            sr=sr
        )

        plt.title("Audio Waveform")

        plt.tight_layout()

        plt.savefig(
            waveform_path,
            dpi=300
        )

        plt.close()

        return waveform_path