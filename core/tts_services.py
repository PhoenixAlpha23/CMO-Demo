import re
from io import BytesIO
import time

# New imports for TTS and language detection
try:
    from gtts import gTTS
    import pygame
    from langdetect import detect, DetectorFactory
    # Set seed for consistent language detection
    DetectorFactory.seed = 0
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️ TTS dependencies not installed. Install with: pip install gtts pygame langdetect")

def detect_language(text):
    """
    Auto-detect language from text
    Returns language code (e.g., 'en', 'hi', 'mr')
    """
    if not TTS_AVAILABLE:
        return 'en'
    
    try:
        # Clean text for better detection
        clean_text = re.sub(r'[^\w\s]', '', text)
        if len(clean_text.strip()) < 10:
            return 'en'  # Default for short text
        
        detected = detect(clean_text)
        
        # Map some common languages
        lang_mapping = {
            'hi': 'hi',  # Hindi
            'mr': 'mr',  # Marathi
            'en': 'en'  # English
        }
        
        return lang_mapping.get(detected, 'en')
        
    except Exception as e:
        print(f"Language detection failed: {e}")
        return 'en'  # Fallback to English

def text_to_speech(text, lang=None, auto_detect=True, speed=1.0, audio_cache=None, get_audio_hash_func=None, cache_audio_func=None, get_cached_audio_func=None):
    """
    Convert text to speech with caching and speed control
    Returns: (audio_bytes, language_used, cache_status)
    """
    if not TTS_AVAILABLE:
        return None, 'en', 'TTS not available'
    
    try:
        # Auto-detect language if not provided
        if auto_detect or not lang:
            detected_lang = detect_language(text)
            lang = detected_lang
        
        # Check cache first (including speed in hash)
        if audio_cache is not None and get_audio_hash_func and cache_audio_func and get_cached_audio_func:
            audio_hash = get_audio_hash_func(text, lang, speed)
            cached_audio = get_cached_audio_func(audio_hash)
            if cached_audio:
                return cached_audio, lang, 'cached'
        
        # Generate TTS with speed control
        # Note: gTTS doesn't directly support speed, but we can simulate it
        slow_speech = speed < 0.8 # gTTS only has slow=True/False. True is ~0.7x
        tts = gTTS(text=text, lang=lang, slow=slow_speech)
        
        # Save to BytesIO buffer
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        audio_bytes = audio_buffer.getvalue()
        
        # Cache the audio
        if audio_cache is not None and get_audio_hash_func and cache_audio_func:
            cache_audio_func(audio_hash, audio_bytes)
        
        return audio_bytes, lang, 'generated'
        
    except Exception as e:
        print(f"TTS Error: {e}")
        return None, lang or 'en', f'error: {str(e)}'

def generate_audio_response(text, lang_preference=None, speed=1.0, audio_cache=None, get_audio_hash_func=None, cache_audio_func=None, get_cached_audio_func=None):
    """
    Generate audio response for given text - updated to match rag_app.py expectations
    Returns: (audio_data, detected_lang, cache_hit) tuple
    """
    if not TTS_AVAILABLE:
        return None, 'en', False

    try:
        # Clean the input text
        clean_text = re.sub(r'\[.*?\]', '', text)
        clean_text = re.sub(r'[✅ℹ️🔍⚠️*●#]', '', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        if len(clean_text) < 5: # Min length for meaningful TTS
            return None, (lang_preference if lang_preference != 'auto' else 'en'), False

        target_lang = lang_preference if lang_preference and lang_preference != 'auto' else None
        
        audio_bytes, final_lang, cache_status = text_to_speech(
            clean_text,
            lang=target_lang, # Pass specific lang if chosen, else None for auto-detect
            auto_detect=(not target_lang), # Auto-detect only if no specific lang is preferred
            speed=speed,
            audio_cache=audio_cache,
            get_audio_hash_func=get_audio_hash_func,
            cache_audio_func=cache_audio_func,
            get_cached_audio_func=get_cached_audio_func
        )

        cache_hit = cache_status == 'cached'
        
        # If lang_preference was 'auto', final_lang is the detected one.
        # If a specific lang was preferred, final_lang is that preferred one (or fallback if error).
        return audio_bytes, final_lang, cache_hit

    except Exception as e:
        print(f"Error generating audio in generate_audio_response: {str(e)}")
        return None, (lang_preference if lang_preference != 'auto' else 'en'), False
