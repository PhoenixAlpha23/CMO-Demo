import tempfile
import os

def transcribe_audio(client, audio_bytes):
    """
    Transcribes audio bytes using the provided Groq client and Whisper model.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name

    try:
        with open(temp_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=file,
                model="whisper-large-v3", # You can make this configurable if needed
                response_format="verbose_json", # Or "text" if you only need the text
                timestamp_granularities=["word", "segment"],
                temperature=0.0
            )
        return transcription.text
    finally:
        if os.path.exists(temp_path): # Check if file exists before unlinking
            os.unlink(temp_path)