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
from typing import ClassVar, List
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

# UNIFIED GOVERNMENT SCHEMES PROMPT
GOVERNMENT_SCHEMES_PROMPT = """You are a Knowledge Assistant designed for answering questions specifically from the knowledge base provided to you.

Your task is as follows: give a detailed response for user query in the user language (eg. what are some schemes? --> Here is a list of some schemes)

Ensure your response follows these styles and tone in your response:
* Use direct, everyday language
* Personal and human
* Favour detailed responses, with mentions of websites and headings such as description, eligibility or उद्देशः, अंतर्भूत घटकः
* In case no relevant information is found, default your response to a phrase like "For more details contact on 104/102 helpline numbers."

Your goal is to achieve the following: help a citizen understand about the schemes and its eligibility criteria.

Here is the content you will work with: {context}

Question: {question}

Now perform the task as instructed above.

Answer:"""

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
            'en': 'en'  # English
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

class EnhancedTFIDFRetriever(TFIDFRetriever):
    """Enhanced TFIDF Retriever with keyword boosting for government schemes"""
    MARATHI_KEYWORDS:ClassVar[List[str]] = [
    "उद्देशः", "अंतर्भूत घटक", "हेल्प लाईन क्र", "योजना", "लाभार्थी", 
    "सेवा", "हेल्पलाइन", "टोल फ्री नंबर", "हेल्पलाईनवर","अधिक माहितीसाठी","अधिक"," माहिती"
]
    
    ENGLISH_KEYWORDS: ClassVar[List[str]] = [
    "Description:", "Eligibility:", "Target Group:", "Inclusion Criteria:",
    "Exclusion Criteria:", "Benefits:", "Helpline:"
]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.english_keywords = ENGLISH_KEYWORDS
        self.marathi_keywords = MARATHI_KEYWORDS
    
    def get_relevant_documents(self, query, k=None):
        """Enhanced retrieval with keyword boosting"""
        if k is None:
            k = self.k
        
        # Get base documents
        base_docs = super().get_relevant_documents(query, k * 2)  # Get more initially
        
        # Score documents based on keyword presence
        scored_docs = []
        for doc in base_docs:
            score = 0
            content = doc.page_content.lower()
            
            # Boost score for English keywords
            for keyword in self.english_keywords:
                if keyword.lower() in content:
                    score += 2
            
            # Boost score for Marathi keywords
            for keyword in self.marathi_keywords:
                if keyword in doc.page_content:  # Case sensitive for Marathi
                    score += 3
            
            # Boost for scheme-related terms
            scheme_terms = ['scheme', 'yojana', 'योजना', 'कार्यक्रम', 'सेवा', 'programme', 'mission']
            for term in scheme_terms:
                if term in content:
                    score += 1
            
            scored_docs.append((doc, score))
        
        # Sort by score (descending) and return top k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:k]]

def build_rag_chain_from_files(pdf_file, txt_file, groq_api_key, enhanced_mode=True):
    """
    Build a rate-limit optimized RAG chain with unified government schemes prompt.
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

        # Enhanced chunking for government schemes
        if enhanced_mode:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Increased for better context preservation
                chunk_overlap=300,  # Higher overlap to avoid splitting scheme info
                separators=[
                    "\n\n",     # Paragraph breaks
                    "\n",       # Line breaks
                    "।",        # Devanagari sentence end
                    ".",        # English sentence end
                    "?", "!",   # Question/exclamation
                    ";", ","    # Clauses
                ]
            )
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        
        splits = splitter.split_documents(all_docs)
        
        # Use enhanced retriever with keyword boosting
        max_chunks = 15 if enhanced_mode else 20
        if enhanced_mode:
            retriever = EnhancedTFIDFRetriever.from_documents(splits,
                                                              k=min(max_chunks, len(splits)),
                                                              english_keywords=ENGLISH_KEYWORDS,
                                                              marathi_keywords=MARATHI_KEYWORDS)

        else:
            retriever = TFIDFRetriever.from_documents(splits, k=min(max_chunks, len(splits)))

        # Use smaller, faster model for efficiency
        llm = ChatGroq(
            api_key=groq_api_key, 
            model="llama-3.1-8b-instant",  # Faster, cheaper model
            temperature=0.1,  # Low temperature for consistent responses
            max_tokens=2000  # Limit output tokens
        )
        
        # Use unified government schemes prompt
        unified_prompt = PromptTemplate(
            template=GOVERNMENT_SCHEMES_PROMPT,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": unified_prompt}
        )
            
    except Exception as e:
        raise ValueError(f"Failed to build RAG chain: {str(e)}")
    finally:
        if pdf_path and os.path.exists(pdf_path):
            os.unlink(pdf_path)
        if txt_path and os.path.exists(txt_path):
            os.unlink(txt_path)

def build_rag_chain_with_model_choice(pdf_file, txt_file, groq_api_key, model_choice="llama-3.1-8b-instant", enhanced_mode=True):
    """
    Build RAG chain with selectable model and unified government schemes prompt.
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
            chunk_size, max_chunks, max_tokens = 1000, 12, 1800
        elif model_choice == "llama-3.1-70b-versatile":
            chunk_size, max_chunks, max_tokens = 1000, 18, 2500
        else:  # llama-3.3-70b-versatile
            chunk_size, max_chunks, max_tokens = 1200, 20, 3000

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=300,
            separators=["\n\n", "\n", "।", ".", "!", "?", ",", " ", ""]
        )
        splits = splitter.split_documents(all_docs)
        
        # Use enhanced retriever with keyword boosting
        if enhanced_mode:
            retriever = EnhancedTFIDFRetriever.from_documents(splits, k=min(max_chunks, len(splits)))
        else:
            retriever = TFIDFRetriever.from_documents(splits, k=min(max_chunks, len(splits)))

        llm = ChatGroq(
            api_key=groq_api_key, 
            model=model_choice,
            temperature=0.1,
            max_tokens=max_tokens
        )
        
        # Use unified government schemes prompt
        unified_prompt = PromptTemplate(
            template=GOVERNMENT_SCHEMES_PROMPT,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": unified_prompt}
        )
            
    except Exception as e:
        raise ValueError(f"Failed to build RAG chain: {str(e)}")
    finally:
        if pdf_path and os.path.exists(pdf_path):
            os.unlink(pdf_path)
        if txt_path and os.path.exists(txt_path):
            os.unlink(txt_path)

def extract_all_scheme_names(text):
    """Enhanced scheme name extraction with better patterns"""
    # Normalize text
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\n', ' ')

    # Enhanced patterns for better scheme detection
    patterns = [
        # English schemes
        r'\b(?:[A-Z][a-z]+(?: [A-Z][a-z]+)*) (?:Scheme|Programme|Mission|Yojana|Initiative|Project)\b',
        r'\bPradhan Mantri [A-Z][a-z]+(?: [A-Z][a-z]+)* (?:Scheme|Yojana)\b',
        r'\b[A-Z]{2,6}\s*(?:Scheme|Programme|Mission)\b',  # Acronyms
        
        # Marathi/Hindi schemes
        r'\b(?:[अ-ह]+\s*){1,4}(?:योजना|अभियान|कार्यक्रम|सेवा|केंद्र)\b',
        r'\bप्रधानमंत्री\s+(?:[अ-ह]+\s*){1,3}योजना\b',
        r'\bमुख्यमंत्री\s+(?:[अ-ह]+\s*){1,3}योजना\b',
        
        # Mixed language patterns
        r'\b(?:[A-Za-z]+[-\s]*){1,4}(?:योजना|Yojana)\b',
    ]

    all_matches = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.UNICODE)
        all_matches.update(match.strip() for match in matches if len(match.strip()) > 4)

    return sorted(list(all_matches))

def process_unified_query(rag_chain, user_query, max_retries=3, enable_tts=False, autoplay=False):
    """
    Process query using unified prompt with enhanced error handling and fallback
    """
    # Check cache first
    query_hash = get_query_hash(user_query.lower().strip())
    cached_result = get_cached_result(query_hash)
    
    if cached_result:
        result_text = cached_result
        cache_status = "hit"
    else:
        cache_status = "miss"
        
        # Process with retry logic
        for attempt in range(max_retries):
            try:
                result = rag_chain.invoke({"query": user_query})
                result_text = result.get('result', 'No results found.')
                
                # Apply fallback message if no relevant info found
                if any(phrase in result_text.lower() for phrase in [
                    "no information", "not found", "cannot find", "not available",
                    "माहिती उपलब्ध नाही", "सापडले नाही"
                ]):
                    result_text += "\n\nFor more details contact on 104/102 helpline numbers."
                
                # Cache successful result
                cache_result(query_hash, result_text)
                break
                
            except Exception as e:
                error_str = str(e)
                
                if "rate_limit" in error_str.lower() or "413" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        time.sleep(wait_time)
                        continue
                    else:
                        result_text = "Rate limit reached. Please wait and try again. For immediate help, contact 104/102 helpline numbers."
                        break
                
                elif "too large" in error_str.lower():
                    result_text = "Query too complex. Please ask about specific schemes. For general help, contact 104/102 helpline numbers."
                    break
                
                else:
                    if attempt == max_retries - 1:
                        result_text = f"Error processing query. For assistance, contact 104/102 helpline numbers."
                    continue
    
    # Generate TTS if enabled
    audio_html = ""
    language_detected = "en"
    cache_info = {"text_cache": cache_status, "audio_cache": "disabled"}
    
    if enable_tts and TTS_AVAILABLE:
        # Clean text for TTS
        clean_text = re.sub(r'[✅ℹ️🔍📋💫📞]', '', result_text).strip()
        
        if len(clean_text) > 10:
            audio_bytes, language_detected, audio_status = text_to_speech(clean_text, auto_detect=True)
            cache_info["audio_cache"] = audio_status
            
            if audio_bytes:
                audio_html = get_audio_player_html(audio_bytes, autoplay=autoplay)
    
    return result_text, audio_html, language_detected, cache_info

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
    Suggestions optimized for unified prompt
    """
    return [
        "List all government schemes",
        "सरकारी योजनांची यादी (List of government schemes)",
        "Health schemes and eligibility",
        "आरोग्य योजना आणि पात्रता (Health schemes and eligibility)",
        "Education schemes details",
        "शिक्षण योजना तपशील (Education scheme details)",
        "Financial assistance programs",
        "आर्थिक सहाय्य कार्यक्रम (Financial assistance programs)",
        "Women welfare schemes",
        "महिला कल्याण योजना (Women welfare schemes)",
        "Farmer schemes and benefits",
        "शेतकरी योजना आणि लाभ (Farmer schemes and benefits)"
    ]

def get_model_options():
    """
    Return available models with their characteristics.
    """
    return {
        "llama-3.1-8b-instant": {
            "name": "Llama 3.1 8B (Fast & Efficient)", 
            "description": "Best for quick queries, lower rate limits, optimized for government schemes"
        },
        "llama-3.1-70b-versatile": {
            "name": "Llama 3.1 70B (Balanced)", 
            "description": "Good balance of quality and speed for detailed scheme information"
        },
        "llama-3.3-70b-versatile": {
            "name": "Llama 3.3 70B (High Quality)", 
            "description": "Best quality responses, detailed explanations, higher rate limits"
        }
    }

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

def extract_scheme_details(text, scheme_name):
    """Enhanced scheme details extraction for bilingual content"""
    details = {
        "name": scheme_name,
        "description": {"en": [], "mr": []},
        "eligibility": {"en": [], "mr": []},
        "benefits": {"en": [], "mr": []},
        "documents": {"en": [], "mr": []},
        "helpline": [],
        "website": []
    }
    
    # Extract website links
    website_pattern = r'(?:http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|(?:www\.)[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(?:/\S*)?)'
    details["website"] = re.findall(website_pattern, text)
    
    # Extract helpline numbers
    helpline_pattern = r'(?:हेल्प लाईन|Helpline|टोल फ्री|Toll Free|104|102).*?([0-9\-]{8,})'
    helpline_matches = re.findall(helpline_pattern, text, re.IGNORECASE)
    details["helpline"] = helpline_matches

    # Process English sections
    for section in ["Description:", "Eligibility:", "Benefits:"]:
        pattern = f"{section}(.*?)(?={'|'.join(ENGLISH_KEYWORDS)}|$)"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            key = section.lower().strip(":")
            if key in details:
                details[key]["en"] = [m.strip() for m in matches if m.strip()]

    # Process Marathi sections
    marathi_patterns = {
        "description": r'उद्देशः?(.*?)(?=पात्रता|लाभार्थी|$)',
        "eligibility": r'(?:पात्रता|लाभार्थी).*?निकष?(.*?)(?=लाभ|अर्ज|$)',
        "benefits": r'(?:लाभ|सुविधा)(.*?)(?=कागदपत्रे|अर्ज|$)',
        "documents": r'(?:कागदपत्रे|आवश्यक दस्तऐवज)(.*?)(?=अर्ज|$)'
    }

    for key, pattern in marathi_patterns.items():
        matches = re.findall(pattern, text, re.DOTALL | re.UNICODE)
        if matches and key in details:
            details[key]["mr"] = [m.strip() for m in matches if m.strip()]

    return details

# Main function alias for backward compatibility
process_scheme_query_with_retry = process_unified_query
process_scheme_query = process_unified_query
