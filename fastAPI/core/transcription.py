import tempfile
import os
import langid  # pip install langid
from typing import cast

SUPPORTED_LANGUAGES = {"en": "English", "hi": "Hindi", "mr": "Marathi"}

def validate_language(text):
    """Check if text is primarily in a supported language."""
    lang, _ = langid.classify(text)
    return lang in SUPPORTED_LANGUAGES

def transcribe_audio(client, audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name

    try:
        with open(temp_path, "rb") as file:
            result = client.audio.transcriptions.create(
                file=file,
                model="whisper-large-v3",
                response_format="text",
                temperature=0.0
            )

        #Groq returns a tuple (text, response), extract just the text
        if isinstance(result, tuple):
            transcription = result[0]
        else:
            transcription = result  # fallback if Groq changes behavior later

        # Validate language from transcribed text
        if not validate_language(transcription):
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.values())
            return (
                False,
                f"Sorry, I only support {supported_langs}. "
                "Please speak in one of these languages."
            )

        return (True, transcription)
    finally:
        os.unlink(temp_path) if os.path.exists(temp_path) else None

def transcribe_audio_google(audio_bytes, sample_rate=16000, language_code="en-US"):
    from google.cloud import speech
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code=language_code,
    )
    response = client.recognize(config=config, audio=audio)
    for result in response.results:
        transcription = result.alternatives[0].transcript
        return (True, transcription)
    return (False, "")

def transcribe_audio_whisper(audio_bytes, model_name="large"):
    import whisper
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name
    model = whisper.load_model(model_name)
    audio = whisper.load_audio(temp_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    probs = cast(dict[str, float], probs)
    detected_lang = max(probs, key=lambda k: probs[k])
    if detected_lang not in SUPPORTED_LANGUAGES:
        supported_langs = ", ".join(SUPPORTED_LANGUAGES.values())
        os.unlink(temp_path)
        return (
            False,
            f"Sorry, I only support {supported_langs}. Please speak in one of these languages."
        )
    result = model.transcribe(temp_path)
    os.unlink(temp_path)
    if isinstance(result, dict) and 'text' in result:
        transcription = result['text']
    else:
        transcription = ''
    return True, transcription