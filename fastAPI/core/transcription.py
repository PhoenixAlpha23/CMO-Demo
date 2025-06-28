import tempfile
import os
import langid  # pip install langid
from typing import cast
import re
import io

SUPPORTED_LANGUAGES = {"en": "English", "hi": "Hindi", "mr": "Marathi"}

def validate_language(text):
    """Check if text is primarily in a supported language."""
    if not text or len(text.strip()) < 2:
        return True  # Allow very short text to pass through
    
    # First try langid classification
    try:
        lang, confidence = langid.classify(text)
        print(f"Language detection: {lang} with confidence {confidence}")
        
        # If confidence is high and language is supported, accept it
        if confidence > 0.5 and lang in SUPPORTED_LANGUAGES:
            return True
        
        # If confidence is low, check for Indic script patterns
        if confidence < 0.5:
            return check_indic_script(text)
        
        return lang in SUPPORTED_LANGUAGES
    except Exception as e:
        print(f"Language detection error: {e}")
        # Fallback to script-based detection
        return check_indic_script(text)

def check_indic_script(text):
    """Check if text contains Indic script characters (Devanagari, etc.)"""
    # Devanagari script range (includes Hindi, Marathi, Sanskrit, etc.)
    devanagari_pattern = r'[\u0900-\u097F]'
    
    # Check if text contains Devanagari characters
    if re.search(devanagari_pattern, text):
        print(f"Detected Devanagari script in text: {text[:50]}...")
        return True
    
    # Check for Latin script (English)
    latin_pattern = r'[a-zA-Z]'
    if re.search(latin_pattern, text):
        print(f"Detected Latin script in text: {text[:50]}...")
        return True
    
    # Check for common Hindi/Marathi words in transliterated form
    hindi_marathi_words = [
        'hai', 'main', 'aap', 'kya', 'kaise', 'kahan', 'kab', 'kyun', 'kaun',
        'nahi', 'haan', 'theek', 'achha', 'badiya', 'sahi', 'galat', 'samajh',
        'karo', 'kare', 'karta', 'karti', 'raha', 'rahi', 'gaya', 'gayi',
        'aaya', 'aayi', 'jao', 'jaa', 'dekh', 'sun', 'bol', 'bata', 'batao',
        'samajh', 'samajhta', 'samajhti', 'pata', 'malum', 'kuch', 'kuchh',
        'sab', 'sabhi', 'koi', 'kuch', 'kuchh', 'yeh', 'woh', 'us', 'is',
        'mere', 'mera', 'meri', 'hamara', 'hamari', 'aapka', 'aapki', 'aapke',
        'unka', 'unki', 'unke', 'iska', 'iski', 'iske', 'uska', 'uski', 'uske',
        # Marathi specific words
        'ahe', 'ahet', 'karto', 'karte', 'kartat', 'kartat', 'gela', 'geli',
        'aala', 'aali', 'jaa', 'dekh', 'aik', 'bol', 'sang', 'sodun', 'sodla',
        'samajh', 'samajhta', 'samajhte', 'mala', 'tula', 'tyala', 'tyachi',
        'tya', 'to', 'ti', 'te', 'mi', 'tu', 'to', 'amhi', 'tumhi', 'tya',
        'he', 'hi', 'ha', 'ho', 'ka', 'ki', 'ke', 'cha', 'chi', 'che'
    ]
    
    # Convert text to lowercase for comparison
    text_lower = text.lower()
    words = text_lower.split()
    
    # Count Hindi/Marathi words
    hindi_marathi_count = sum(1 for word in words if word in hindi_marathi_words)
    
    # If more than 30% of words are Hindi/Marathi, accept it
    if len(words) > 0 and (hindi_marathi_count / len(words)) > 0.3:
        print(f"Detected Hindi/Marathi words in text: {text[:50]}...")
        return True
    
    # Check for transliterated patterns (common in Indic languages)
    transliterated_patterns = [
        r'\b[a-z]+[aeiou][a-z]*\b',  # Words with vowels (common in transliterated Hindi/Marathi)
        r'\b[a-z]*[aeiou][a-z]*[aeiou][a-z]*\b',  # Words with multiple vowels
        r'\b[a-z]*[aeiou][a-z]*[aeiou][a-z]*[aeiou][a-z]*\b'  # Words with many vowels
    ]
    
    for pattern in transliterated_patterns:
        matches = re.findall(pattern, text_lower)
        if len(matches) > len(words) * 0.5:  # If more than 50% of words match transliterated pattern
            print(f"Detected transliterated pattern in text: {text[:50]}...")
            return True
    
    # If no clear script pattern, be permissive and allow it
    print(f"No clear script pattern detected, allowing text: {text[:50]}...")
    return True

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
    
    # Convert audio to proper format first
    converted_audio = convert_audio_format(audio_bytes, 'wav')
    
    # Preprocess audio for better quality
    processed_audio = preprocess_audio(converted_audio)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(processed_audio)
        temp_path = f.name
    
    try:
        # Load the model (use base model for faster processing)
        model = whisper.load_model(model_name)
        
        # Try transcription with language hints for better Indic language support
        # First try with Hindi language hint
        try:
            result = model.transcribe(temp_path, language="hi")
            transcription = str(result.get('text', '')).strip()
            if transcription and len(transcription.strip()) >= 2:
                print(f"Transcription with Hindi hint: {transcription}")
                if validate_language(transcription):
                    return (True, transcription)
        except Exception as e:
            print(f"Hindi transcription failed: {e}")
        
        # Try with Marathi language hint
        try:
            result = model.transcribe(temp_path, language="mr")
            transcription = str(result.get('text', '')).strip()
            if transcription and len(transcription.strip()) >= 2:
                print(f"Transcription with Marathi hint: {transcription}")
                if validate_language(transcription):
                    return (True, transcription)
        except Exception as e:
            print(f"Marathi transcription failed: {e}")
        
        # Fallback to auto language detection
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
        
        print(f"Raw transcription: {transcription}")
        
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
    # Try with different models in order of preference for Indic languages
    models_to_try = ["large-v3", "large-v2", "base", "small", "tiny"]
    
    for model in models_to_try:
        print(f"Trying transcription with model: {model}")
        success, transcription = transcribe_audio_whisper(audio_bytes, model)
        
        if success:
            print(f"Successfully transcribed with {model} model")
            return success, transcription
    
    # If all models fail, return the last error
    return success, transcription

def convert_audio_format(audio_bytes, target_format='wav'):
    """
    Convert audio to the target format for better transcription.
    """
    try:
        import ffmpeg
        
        # Create input buffer
        input_buffer = io.BytesIO(audio_bytes)
        
        # Convert to target format
        out, _ = (
            ffmpeg
            .input('pipe:0')
            .output('pipe:1', 
                   format=target_format,
                   acodec='pcm_s16le',
                   ac=1,  # mono
                   ar='16000')  # 16kHz sample rate
            .run(input=input_buffer.read(), capture_stdout=True, capture_stderr=True)
        )
        
        return out
    except Exception as e:
        print(f"Audio format conversion failed: {e}")
        # Return original audio if conversion fails
        return audio_bytes

def preprocess_audio(audio_bytes):
    """
    Preprocess audio for better transcription quality.
    This helps with Indic language transcription.
    """
    try:
        import ffmpeg
        
        # Create input buffer
        input_buffer = io.BytesIO(audio_bytes)
        
        # Process audio with ffmpeg for better quality
        out, _ = (
            ffmpeg
            .input('pipe:0')
            .output('pipe:1', 
                   format='wav',
                   acodec='pcm_s16le',
                   ac=1,  # mono
                   ar='16000',  # 16kHz sample rate
                   af='highpass=f=50,lowpass=f=8000,volume=1.5')  # Filter and amplify
            .run(input=input_buffer.read(), capture_stdout=True, capture_stderr=True)
        )
        
        return out
    except Exception as e:
        print(f"Audio preprocessing failed: {e}")
        # Return original audio if preprocessing fails
        return audio_bytes