import os
import tempfile
import whisper


class SpeechToText:
    """
    Converts uploaded audio into text using OpenAI Whisper.
    """

    def __init__(self):
        """
        Load the Whisper model only once.
        """
        self.model = whisper.load_model("tiny")

    def transcribe_audio(self, uploaded_file):
        """
        Transcribes the uploaded audio file.

        Parameters
        ----------
        uploaded_file : UploadedFile
            Streamlit uploaded audio object.

        Returns
        -------
        tuple
            (transcript, audio_path)
        """

        try:
            # Create temporary file
            suffix = os.path.splitext(uploaded_file.name)[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(uploaded_file.read())
                audio_path = temp_file.name

            # Transcribe
            result = self.model.transcribe(audio_path)

            transcript = result["text"].strip()

            return transcript, audio_path

        except Exception as e:
            raise Exception(
                f"Speech-to-Text Error: {str(e)}"
            )