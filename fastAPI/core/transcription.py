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

def transcribe_audio_whisper(audio_bytes, model_name="base"):
    """
    Transcribe audio using OpenAI Whisper with proper error handling.
    Uses a smaller model by default for faster processing.
    """
    import whisper
    import tempfile
    import os
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name
    
    try:
        # Load the model (use base model for faster processing)
        model = whisper.load_model(model_name)
        
        # Transcribe directly using Whisper's transcribe method
        # This handles all the audio preprocessing internally
        result = model.transcribe(temp_path)
        
        # Extract transcription text
        if isinstance(result, dict) and 'text' in result:
            transcription = str(result['text']).strip()
        elif isinstance(result, str):
            transcription = result.strip()
        else:
            transcription = str(result).strip() if result else ''
        
        # Check if transcription is empty or too short
        if not transcription or len(transcription.strip()) < 2:
            return (False, "No speech detected. Please try speaking more clearly.")
        
        # Validate language from transcribed text
        if not validate_language(transcription):
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.values())
            return (
                False,
                f"Sorry, I only support {supported_langs}. Please speak in one of these languages."
            )
        
        return (True, transcription)
        
    except Exception as e:
        print(f"Whisper transcription error: {e}")
        return (False, f"Transcription failed: {str(e)}")
        
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception as e:
            print(f"Error cleaning up temp file: {e}")

def transcribe_audio_robust(audio_bytes, model_name="base"):
    """
    Robust transcription function that tries multiple approaches.
    """
    # First try with the base model
    success, transcription = transcribe_audio_whisper(audio_bytes, model_name)
    
    if success:
        return success, transcription
    
    # If base model fails, try with tiny model
    if model_name != "tiny":
        print("Base model failed, trying tiny model...")
        success, transcription = transcribe_audio_whisper(audio_bytes, "tiny")
        if success:
            return success, transcription
    
    # If still fails, return the original error
    return success, transcription