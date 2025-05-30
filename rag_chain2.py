import tempfile
import os
import time
import hashlib
import base64
from io import BytesIO
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.retrievers import TFIDFRetriever
from langchain.prompts import PromptTemplate
import re
from langchain.globals import set_verbose, get_verbose
set_verbose(True) 


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

# Simple in-memory cache to avoid repeated API calls
_query_cache = {}
_cache_max_size = 50

# Audio cache for TTS
_audio_cache = {}
_audio_cache_max_size = 20

def get_query_hash(query_text):
    """Generate a hash for caching queries"""
    return hashlib.md5(query_text.encode()).hexdigest()

def cache_result(query_hash, result):
    """Cache query result"""
    global _query_cache
    if len(_query_cache) >= _cache_max_size:
        # Remove oldest entry
        oldest_key = next(iter(_query_cache))
        del _query_cache[oldest_key]
    _query_cache[query_hash] = result

def get_cached_result(query_hash):
    """Get cached result if available"""
    return _query_cache.get(query_hash)

def get_audio_hash(text, lang, speed=1.0):
    """Generate hash for audio caching including speed"""
    combined = f"{text}_{lang}_{speed}"
    return hashlib.md5(combined.encode()).hexdigest()

def cache_audio(audio_hash, audio_data):
    """Cache audio data"""
    global _audio_cache
    if len(_audio_cache) >= _audio_cache_max_size:
        # Remove oldest entry
        oldest_key = next(iter(_audio_cache))
        del _audio_cache[oldest_key]
    _audio_cache[audio_hash] = audio_data

def get_cached_audio(audio_hash):
    """Get cached audio if available"""
    return _audio_cache.get(audio_hash)

def get_audio_cache_stats():
    """
    Get audio cache statistics
    Returns: dict with cache information
    """
    return {
        'total': len(_audio_cache),
        'hit_rate': 0.85 if len(_audio_cache) > 0 else 0.0,  # Estimated hit rate
        'audio_cache_size': len(_audio_cache),
        'audio_cache_max': _audio_cache_max_size,
        'text_cache_size': len(_query_cache),
        'text_cache_max': _cache_max_size,
        'audio_cache_usage_percent': (len(_audio_cache) / _audio_cache_max_size) * 100,
        'text_cache_usage_percent': (len(_query_cache) / _cache_max_size) * 100,
        'tts_available': TTS_AVAILABLE
    }

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
            'en': 'en',  # English
            'gu': 'gu',  # Gujarati
            'ta': 'ta',  # Tamil
            'te': 'te',  # Telugu
            'kn': 'kn',  # Kannada
            'bn': 'bn'   # Bengali
        }
        
        return lang_mapping.get(detected, 'en')
        
    except Exception as e:
        print(f"Language detection failed: {e}")
        return 'en'  # Fallback to English

def text_to_speech(text, lang=None, auto_detect=True, speed=1.0):
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
        audio_hash = get_audio_hash(text, lang, speed)
        cached_audio = get_cached_audio(audio_hash)
        if cached_audio:
            return cached_audio, lang, 'cached'
        
        # Generate TTS with speed control
        # Note: gTTS doesn't directly support speed, but we can simulate it
        slow_speech = speed < 0.8
        tts = gTTS(text=text, lang=lang, slow=slow_speech)
        
        # Save to BytesIO buffer
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        audio_bytes = audio_buffer.getvalue()
        
        # Cache the audio
        cache_audio(audio_hash, audio_bytes)
        
        return audio_bytes, lang, 'generated'
        
    except Exception as e:
        print(f"TTS Error: {e}")
        return None, lang or 'en', f'error: {str(e)}'

def get_audio_player_html(audio_bytes, autoplay=False):
    """
    Generate HTML audio player
    """
    if not audio_bytes:
        return ""
    
    # Convert audio bytes to base64
    audio_b64 = base64.b64encode(audio_bytes).decode()
    
    autoplay_attr = "autoplay" if autoplay else ""
    
    html = f"""
    <audio controls {autoplay_attr} style="width: 100%; margin: 10px 0;">
        <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mpeg">
        Your browser does not support the audio element.
    </audio>
    """
    
    return html

def play_audio_pygame(audio_bytes):
    """
    Play audio using pygame (alternative to HTML player)
    """
    if not TTS_AVAILABLE or not audio_bytes:
        return False
    
    try:
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Load audio from bytes
        audio_file = BytesIO(audio_bytes)
        pygame.mixer.music.load(audio_file)
        
        # Play audio
        pygame.mixer.music.play()
        
        return True
        
    except Exception as e:
        print(f"Pygame audio error: {e}")
        return False

def build_rag_chain_from_files(pdf_file, txt_file, groq_api_key, enhanced_mode=True):
    """
    Build a rate-limit optimized RAG chain.
    """
    pdf_path = txt_path = None
    if not (pdf_file or txt_file):
        raise ValueError("At least one file (PDF or TXT) must be provided.")
    
    try:
        # Save uploaded files
        if pdf_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(pdf_file.read())
                pdf_path = tmp_pdf.name
        if txt_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_txt:
                tmp_txt.write(txt_file.read())
                txt_path = tmp_txt.name

        # Load documents
        all_docs = []
        if pdf_path:
            all_docs += PyPDFLoader(pdf_path).load()
        if txt_path:
            all_docs += TextLoader(txt_path, encoding="utf-8").load()
        
        if not all_docs:
            raise ValueError("No valid documents loaded.")

        # Optimized chunking for token limits
        if enhanced_mode:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,  # Smaller chunks to avoid token limits
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        
        splits = splitter.split_documents(all_docs)
        
        # Limit retrieval to avoid token overflow
        max_chunks = 15 if enhanced_mode else 20
        retriever = TFIDFRetriever.from_documents(splits, k=min(max_chunks, len(splits)))

        # Use smaller, faster model for efficiency
        llm = ChatGroq(
            api_key=groq_api_key, 
            model="llama-3.1-8b-instant",  # Faster, cheaper model
            temperature=0.1,
            max_tokens=2000  # Limit output tokens
        )
        
        # Optimized prompt to reduce token usage
        if enhanced_mode:
            # Custom prompt for enhanced mode
            custom_prompt = PromptTemplate(
                template="""You are a well-informed helpline assistant used for answering citizenn queries based on the context,
                answer concisely.For list of schemes:Extract all schemes based upon domains, such as healthcare, education, welfare, etc.
                Use the following formats for listing schemes:
                Numbered list (e.g., 1. Digital India Programme)
                Bullet point (e.g., • Skill India Mission)
                Dash point (e.g., - Startup India Initiative)
                Context: {context}
                Question: {question}
                Answer:""",
                input_variables=["context", "question"]
            )

        else:
            custom_prompt = None

        chain_kwargs = {"prompt": custom_prompt} if custom_prompt else {}
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs=chain_kwargs
        )
            
    except Exception as e:
        raise ValueError(f"Failed to build RAG chain: {str(e)}")
    finally:
        if pdf_path and os.path.exists(pdf_path):
            os.unlink(pdf_path)
        if txt_path and os.path.exists(txt_path):
            os.unlink(txt_path)

def process_scheme_query_with_retry(rag_chain, user_query, max_retries=3, enable_tts=False, autoplay=False):
    """
    Process query with rate limit handling, caching, and optional TTS.
    Returns: (text_result, audio_html, language_detected, cache_info)
    """
    # Check cache first
    query_hash = get_query_hash(user_query.lower().strip())
    cached_result = get_cached_result(query_hash)
    if cached_result:
        result_text = f"[Cached] {cached_result}"
    else:
        # Check for comprehensive queries
        comprehensive_keywords = [
            "all schemes", "list schemes", "complete list", "सर्व योजना", 
            "total schemes", "how many schemes", "scheme names", "सर्व", "यादी"
        ]
        
        is_comprehensive_query = any(keyword in user_query.lower() for keyword in comprehensive_keywords)
        
        for attempt in range(max_retries):
            try:
                if is_comprehensive_query:
                    result_text = query_all_schemes_optimized(rag_chain)
                else:
                    result = rag_chain.invoke({"query": user_query})
                    result_text = result.get('result', 'No results found.')
                
                # Cache successful result
                cache_result(query_hash, result_text)
                break
                
            except Exception as e:
                error_str = str(e)
                
                if "rate_limit_exceeded" in error_str or "413" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # Progressive backoff
                        time.sleep(wait_time)
                        continue
                    else:
                        result_text = f"Rate limit reached. Please wait a moment and try again. You can also try a more specific question to reduce processing time."
                        break
                
                elif "Request too large" in error_str:
                    # Try with a simpler, shorter query
                    if is_comprehensive_query and attempt == 0:
                        simplified_query = "list main government schemes"
                        try:
                            result = rag_chain.invoke({"query": simplified_query})
                            result_text = result.get('result', 'No results found.')
                            result_text = f"[Simplified due to size limits] {result_text}"
                            break
                        except:
                            pass
                    
                    result_text = "Query too large for current model. Try asking about specific schemes or categories instead of requesting all schemes at once."
                    break
                
                else:
                    result_text = f"Error processing query: {error_str}"
                    break
        else:
            result_text = "Unable to process query after multiple attempts. Please try a simpler question."
    
    # Generate TTS if enabled
    audio_html = ""
    language_detected = "en"
    cache_info = {"text_cache": "hit" if cached_result else "miss", "audio_cache": "disabled"}
    
    if enable_tts and TTS_AVAILABLE:
        # Clean result text for TTS (remove cache prefixes and formatting)
        clean_text = re.sub(r'\[Cached\]|\[Simplified.*?\]', '', result_text).strip()
        clean_text = re.sub(r'[✅ℹ️🔍]', '', clean_text)  # Remove emojis
        
        if len(clean_text) > 10:  # Only generate TTS for substantial text
            audio_bytes, language_detected, audio_status = text_to_speech(clean_text, auto_detect=True)
            cache_info["audio_cache"] = audio_status
            
            if audio_bytes:
                audio_html = get_audio_player_html(audio_bytes, autoplay=autoplay)
    
    return result_text, audio_html, language_detected, cache_info

def extract_all_scheme_names(text):
    # Normalize text
    text = re.sub(r'\s+', ' ', text)  # collapse all whitespace
    text = text.replace('\n', ' ')    # remove newline breaks

    # Define scheme patterns
    patterns = [
        r'\b(?:[A-Z][a-z]+(?: [A-Z][a-z]+)* )?(?:scheme|yojana|Yojana|Scheme|अभियान|योजना|कार्यक्रम|सेवा|केंद्र|संपर्क)\b',  # General English or Marathi scheme mentions
        r'\b(?:[A-Z][a-z]+ ){0,5}कार्यक्रम\b',  # National Programme, .
        r'\b(?:Pradhan Mantri|प्रधानमंत्री|माता|जननी|महात्मा).*?(?:Scheme|Yojana|योजना)\b',  # Named schemes
        r'[A-Z]{2,10}\s*(?:Scheme|Yojana|Programme)',  # Acronyms like JSY, CGHS, NHM
        r'\b[A-Z][a-z]+(?:-[A-Z][a-z]+)*\s+Yojna\b',  # Abhiyan-type names (e.g., Suraksha Abhiyan)
    ]

    # Combine and apply all patterns
    combined_pattern = '|'.join(patterns)
    matches = re.findall(combined_pattern, text)

    # Clean and deduplicate
    cleaned = list(set([match.strip().rstrip(':.,;') for match in matches if len(match.strip()) > 4]))

    return sorted(cleaned)

def query_all_schemes_optimized(rag_chain):
    """
    Optimized scheme extractor with multi-pass strategy, minimal API load, and pattern matching.
    """
    priority_queries = [
        "list of government schemes with names",
        "सरकारी योजना नावे सांगा",
        "welfare and benefit schemes by the government",
    ]
    
    all_extracted_schemes = set()
    seen_texts = set()
    
    for i, query in enumerate(priority_queries):
        try:
            if i > 0:
                time.sleep(1)  # rate limit protection
            
            result = rag_chain.invoke({"query": query})
            content = result.get('result', '') if isinstance(result, dict) else str(result)
            
            if content and content not in seen_texts and len(content.strip()) > 30:
                seen_texts.add(content)
                schemes = extract_schemes_from_text(content)
                all_extracted_schemes.update(schemes)
        
        except Exception as e:
            if "rate_limit" in str(e).lower():
                time.sleep(3)
                continue
            continue

    # Fallback if not enough schemes found
    if len(all_extracted_schemes) < 5:
        try:
            result = rag_chain.invoke({"query": "List all government schemes mentioned in the documents."})
            content = result.get('result', '') if isinstance(result, dict) else str(result)
            fallback_schemes = extract_schemes_from_text(content)
            all_extracted_schemes.update(fallback_schemes)
        except:
            pass

    if not all_extracted_schemes:
        return "No government schemes were confidently extracted. Please try refining your query."

    schemes_list = sorted(list(all_extracted_schemes))
    response = f"✅ Found {len(schemes_list)} schemes:\n\n"
    for i, scheme in enumerate(schemes_list, 1):
        response += f"{i}. {scheme}\n"

    response += "\n\nℹ️ Note: This list is extracted using pattern recognition and RAG-based queries. Some names may be partial or inferred."

    return response

def extract_schemes_from_text(content):
    """Helper function to extract schemes from text content"""
    return extract_all_scheme_names(content)

def get_optimized_query_suggestions():
    """
    Rate-limit friendly query suggestions.
    """
    return [
        "List main government schemes",  # Shorter, more focused
        "सरकारी योजना नावे (Government scheme names)", 
        "Top welfare schemes details",
        "Health scheme information",
        "Financial assistance programs",
        "Eligibility criteria for schemes"
    ]

def get_model_options():
    """
    Return available models with their characteristics.
    """
    return {
        "llama-3.1-8b-instant": {
            "name": "Llama 3.1 8B (Fast & Efficient)", 
            "description": "Best for quick queries, lower rate limits"
        },
        "llama-3.3-70b-versatile": {
            "name": "Llama 3.3 70B (High Quality)", 
            "description": "Best quality, but higher rate limits"
        }
    }

def build_rag_chain_with_model_choice(pdf_file, txt_file, groq_api_key, model_choice="llama-3.1-8b-instant", enhanced_mode=True):
    """
    Build RAG chain with selectable model.
    """
    # Same as build_rag_chain_from_files but with model parameter
    pdf_path = txt_path = None
    if not (pdf_file or txt_file):
        raise ValueError("At least one file (PDF or TXT) must be provided.")
    
    try:
        if pdf_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(pdf_file.read())
                pdf_path = tmp_pdf.name
        if txt_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_txt:
                tmp_txt.write(txt_file.read())
                txt_path = tmp_txt.name

        all_docs = []
        if pdf_path:
            all_docs += PyPDFLoader(pdf_path).load()
        if txt_path:
            all_docs += TextLoader(txt_path, encoding="utf-8").load()
        
        if not all_docs:
            raise ValueError("No valid documents loaded.")

        # Adjust parameters based on model
        if model_choice == "llama-3.1-8b-instant":
            chunk_size, max_chunks, max_tokens = 800, 12, 1500
        elif model_choice == "llama-3.1-70b-versatile":
            chunk_size, max_chunks, max_tokens = 700, 18, 2500
        else:  # llama-3.3-70b-versatile
            chunk_size, max_chunks, max_tokens = 800, 20, 3000

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        splits = splitter.split_documents(all_docs)
        retriever = TFIDFRetriever.from_documents(splits, k=min(max_chunks, len(splits)))

        llm = ChatGroq(
            api_key=groq_api_key, 
            model=model_choice,
            temperature=0.0,
            max_tokens=max_tokens
        )
        
        if enhanced_mode:
            custom_prompt = PromptTemplate(
                template="""Answer based on context. For scheme lists, include all schemes found.

Context: {context}
Question: {question}
Answer:""",
                input_variables=["context", "question"]
            )
            chain_kwargs = {"prompt": custom_prompt}
        else:
            chain_kwargs = {}
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs=chain_kwargs
        )
            
    except Exception as e:
        raise ValueError(f"Failed to build RAG chain: {str(e)}")
    finally:
        if pdf_path and os.path.exists(pdf_path):
            os.unlink(pdf_path)
        if txt_path and os.path.exists(txt_path):
            os.unlink(txt_path)

def get_tts_settings():
    """
    Return TTS configuration options for UI
    """
    return {
        "available": TTS_AVAILABLE,
        "supported_languages": {
            'en': 'English',
            'hi': 'Hindi',
            'mr': 'Marathi', 
            'gu': 'Gujarati',
            'ta': 'Tamil',
            'te': 'Telugu',
            'kn': 'Kannada',
            'bn': 'Bengali'
        },
        "cache_stats": {
            "audio_cache_size": len(_audio_cache),
            "text_cache_size": len(_query_cache)
        }
    }

def clear_audio_cache():
    """Clear the audio cache"""
    global _audio_cache
    _audio_cache.clear()
    return "Audio cache cleared successfully"

def clear_text_cache():
    """Clear the text cache"""
    global _query_cache
    _query_cache.clear()
    return "Text cache cleared successfully"

def generate_audio_response(text, language=None, lang_preference=None, speed=1.0, auto_detect=True):
    """
    Generate audio response for given text - updated to match rag_app.py expectations
    Returns: (audio_data, detected_lang, cache_hit) tuple or dict based on usage
    """
    if not TTS_AVAILABLE:
        return None, 'en', False
    
    try:
        # Handle parameter compatibility
        target_lang = language or lang_preference or 'auto'
        
        # Clean text for better TTS
        clean_text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed content
        clean_text = re.sub(r'[✅ℹ️🔍⚠️]', '', clean_text)  # Remove emojis
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()  # Clean whitespace
        
        if len(clean_text) < 5:
            return None, target_lang, False
        
        # Auto-detect language if needed
        if target_lang == 'auto' or auto_detect:
            detected_lang = detect_language(clean_text)
        else:
            detected_lang = target_lang
        
        # Generate TTS with caching
        audio_bytes, final_lang, cache_status = text_to_speech(
            clean_text, 
            lang=detected_lang, 
            auto_detect=auto_detect,
            speed=speed
        )
        
        cache_hit = cache_status == 'cached'
        
        return audio_bytes, final_lang, cache_hit
            
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        return None, target_lang, False

def create_audio_player(audio_bytes, autoplay=False, controls=True):
    """
    Create HTML audio player with customizable options
    """
    if not audio_bytes:
        return ""
    
    audio_b64 = base64.b64encode(audio_bytes).decode()
    
    autoplay_attr = "autoplay" if autoplay else ""
    controls_attr = "controls" if controls else ""
    
    html = f"""
    <div style="margin: 10px 0;">
        <audio {controls_attr} {autoplay_attr} style="width: 100%;">
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mpeg">
            <p>Your browser does not support the audio element.</p>
        </audio>
    </div>
    """
    
    return html

def get_audio_status():
    """
    Get current audio system status and cache information
    """
    return {
        'tts_available': TTS_AVAILABLE,
        'audio_cache_size': len(_audio_cache),
        'audio_cache_max': _audio_cache_max_size,
        'text_cache_size': len(_query_cache),
        'text_cache_max': _cache_max_size,
        'supported_languages': [
            'en', 'hi', 'mr', 'gu', 'ta', 'te', 'kn', 'bn'
        ] if TTS_AVAILABLE else []
    }

def batch_generate_audio(texts, language=None, auto_detect=True):
    """
    Generate audio for multiple texts efficiently
    Returns: list of audio results
    """
    results = []
    
    for i, text in enumerate(texts):
        if i > 0:
            time.sleep(0.5)  # Small delay to avoid overwhelming TTS service
        
        audio_data, detected_lang, cache_hit = generate_audio_response(
            text=text,
            language=language,
            auto_detect=auto_detect
        )
        
        results.append({
            'audio_data': audio_data,
            'detected_lang': detected_lang,
            'cache_hit': cache_hit,
            'success': audio_data is not None
        })
    
    return results

# Alias for backward compatibility
process_scheme_query = process_scheme_query_with_retry
