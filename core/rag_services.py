import tempfile
import os
import time
import hashlib
import re
import langid
from core.transcription import SUPPORTED_LANGUAGES  
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.retrievers import TFIDFRetriever
from langchain.prompts import PromptTemplate
from langchain.globals import set_verbose

TTS_AVAILABLE = True # added this to use older query processing function pre-break up
# For language detection of the query
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0 # for consistent results
except ImportError:
    print("Warning: langdetect not installed. Query language validation will be skipped. pip install langdetect")

set_verbose(True) # Optional: for Langchain verbosity

_query_cache = {}
_cache_max_size = 50

# Enhanced context storage for conversational flow
_conversation_context = {
    'previous_question': None,
    'last_response': None,
    'suggested_followup': None,
    'scheme_mentioned': None
}

def get_query_hash(query_text):
    """Generate a hash for caching queries"""
    return hashlib.md5(query_text.encode()).hexdigest()

def cache_result(query_hash, result):
    """Cache query result"""
    global _query_cache
    if len(_query_cache) >= _cache_max_size:
        # Remove oldest entry (FIFO)
        try:
            oldest_key = next(iter(_query_cache))
            del _query_cache[oldest_key]
        except StopIteration: # Should not happen if cache_max_size > 0
            pass
    _query_cache[query_hash] = result

def get_cached_result(query_hash):
    """Get cached result if available"""
    return _query_cache.get(query_hash)

def clear_query_cache():
    """Clear the RAG query cache"""
    global _query_cache
    _query_cache.clear()
    print("RAG query cache cleared.")

def set_previous_question_context(question):
    """Set the previous question to be used as context for the next query"""
    global _conversation_context
    _conversation_context['previous_question'] = question

def get_previous_question_context():
    """Get the previous question context"""
    global _conversation_context
    return _conversation_context['previous_question']

def clear_previous_question_context():
    """Clear the previous question context"""
    global _conversation_context
    _conversation_context = {
        'previous_question': None,
        'last_response': None,
        'suggested_followup': None,
        'scheme_mentioned': None
    }
    print("Previous question context cleared.")

def update_conversation_context(question, response):
    """Update the full conversation context including extracting follow-up suggestions"""
    global _conversation_context
    
    _conversation_context['previous_question'] = question
    _conversation_context['last_response'] = response
    
    # Extract suggested follow-up and scheme mentioned
    followup_info = extract_followup_suggestion(response)
    _conversation_context['suggested_followup'] = followup_info.get('suggestion')
    _conversation_context['scheme_mentioned'] = followup_info.get('scheme')

def extract_followup_suggestion(response_text):
    """Extract follow-up suggestion and scheme name from AI response"""
    # Patterns to match follow-up questions in different languages
    patterns = [
        # English patterns
        r'(?:Would you like to know|Do you want to know|Want to know more about)\s+(?:about\s+)?(?:the\s+)?(eligibility criteria?|benefits|details|application process)\s+(?:of\s+|for\s+)?([A-Z][A-Za-z\s\-\.]+?)(?:\?|$)',
        r'(?:Would you like to know|Do you want to know)\s+([A-Z][A-Za-z\s\-\.]+?)\s+(eligibility|benefits|details)(?:\?|$)',
        
        # Hindi patterns  
        r'(?:क्या आपको|आपको)\s+([A-Za-z\s\-\.]+?)\s+(?:की\s+|के\s+)?(पात्रता|लाभ|विवरण|जानकारी)\s+(?:के\s+बारे\s+में\s+)?जानना चाहिए(?:\?|$)',
        r'(?:क्या आपको|आपको)\s+([A-Za-z\s\-\.]+?)\s+(?:eligibility|benefits)\s+(?:के\s+बारे\s+में\s+)?जानकारी चाहिए(?:\?|$)',
        
        # Marathi patterns
        r'(?:तुम्हाला|तुला)\s+([A-Za-z\s\-\.]+?)\s+(?:च्या\s+|चे\s+)?(पात्रता|लाभ|तपशील)(?:बद्दल|बाबत)\s+जाणून घ्यायचे आहे का(?:\?|$)',
        r'(?:तुम्हाला|तुला)\s+([A-Za-z\s\-\.]+?)\s+(?:eligibility|benefits)(?:बद्दल|बाबत)\s+जाणून घ्यायचे आहे का(?:\?|$)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) >= 2:
                scheme = groups[-1].strip()  # Last group is usually the scheme name
                suggestion_type = groups[-2].strip() if len(groups) > 1 else "details"
            else:
                scheme = groups[0].strip()
                suggestion_type = "details"
            
            # Clean up scheme name
            scheme = re.sub(r'[^\w\s\-\.]', '', scheme).strip()
            
            return {
                'suggestion': suggestion_type,
                'scheme': scheme,
                'full_match': match.group(0)
            }
    
    return {'suggestion': None, 'scheme': None, 'full_match': None}

def detect_affirmative_response(user_input):
    """Detect if user input is an affirmative response"""
    user_input = user_input.lower().strip()
    
    # Affirmative responses in different languages
    affirmative_patterns = [
        # English
        r'^(yes|yeah|yep|sure|okay|ok|alright|right|correct|exactly)$',
        r'^(tell me|show me|i want to know|please)$',
        
        # Hindi  
        r'^(हाँ|हा|जी|हां|सही|ठीक|बताओ|बताइए|जानना चाहता हूं|जानना चाहती हूं)$',
        
        # Marathi
        r'^(होय|हो|बरं|ठीक|सांगा|सांगू|जाणून घ्यायचं आहे|जाणून घेऊ इच्छितो)$'
    ]
    
    for pattern in affirmative_patterns:
        if re.match(pattern, user_input):
            return True
    
    return False

def process_contextual_query(user_query):
    """Process user query considering conversation context and affirmative responses"""
    global _conversation_context
    
    # Check if this is an affirmative response to a previous suggestion
    if detect_affirmative_response(user_query):
        suggestion = _conversation_context.get('suggested_followup')
        scheme = _conversation_context.get('scheme_mentioned')
        
        if suggestion and scheme:
            # Convert affirmative response to explicit query
            if suggestion in ['eligibility', 'पात्रता']:
                processed_query = f"What are the eligibility criteria for {scheme}?"
            elif suggestion in ['benefits', 'लाभ']:
                processed_query = f"What are the benefits of {scheme}?"
            elif suggestion in ['details', 'विवरण', 'तपशील']:
                processed_query = f"Tell me more details about {scheme}"
            elif suggestion in ['application', 'application process']:
                processed_query = f"How to apply for {scheme}?"
            else:
                processed_query = f"Tell me about {suggestion} of {scheme}"
            
            print(f"[Context] Converted '{user_query}' to '{processed_query}'")
            return processed_query, True
    
    return user_query, False

def detect_language(text):
    """Return the ISO 639-1 language code and full language name, if supported."""
    lang, _ = langid.classify(text)
    return lang, SUPPORTED_LANGUAGES.get(lang, "Unsupported")

# --- RAG Chain Building ---
def build_rag_chain_from_files(pdf_file, txt_file, groq_api_key, enhanced_mode=True, model_choice="llama-3.1-8b-instant"):
    """
    Build a RAG chain from PDF and/or TXT files.
    This function is kept for potential direct use or as a base.
    `build_rag_chain_with_model_choice` is generally preferred.
    """
    pdf_path = txt_path = None
    if not (pdf_file or txt_file):
        raise ValueError("At least one file (PDF or TXT) must be provided.")
    
    temp_files_to_clean = []
    try:
        if pdf_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(pdf_file.getvalue()) # Use getvalue() for UploadedFile
                pdf_path = tmp_pdf.name
                temp_files_to_clean.append(pdf_path)
        if txt_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_txt:
                tmp_txt.write(txt_file.getvalue()) # Use getvalue() for UploadedFile
                txt_path = tmp_txt.name
                temp_files_to_clean.append(txt_path)

        all_docs = []
        if pdf_path:
            all_docs.extend(PyPDFLoader(pdf_path).load())
        if txt_path:
            all_docs.extend(TextLoader(txt_path, encoding="utf-8").load())
        
        if not all_docs:
            raise ValueError("No valid documents loaded or documents are empty.")

        # Adjust parameters based on model and mode
        if model_choice == "llama-3.1-8b-instant":
            chunk_size = 700 if enhanced_mode else 800
            max_chunks = 10 if enhanced_mode else 12
            max_tokens = 1500
        else: # "llama-3.3-70b-versatile"
            chunk_size = 600 if enhanced_mode else 700
            max_chunks = 15 if enhanced_mode else 18
            max_tokens = 2500

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=max(100, int(chunk_size * 0.2)), # Overlap as a percentage of chunk_size
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
            length_function=len
        )
        splits = splitter.split_documents(all_docs)
        
        if not splits:
            raise ValueError("Document splitting resulted in no chunks. Check document content and splitter settings.")
            
        retriever = TFIDFRetriever.from_documents(splits, k=min(max_chunks, len(splits)))

        llm = ChatGroq(
            api_key=groq_api_key, 
            model=model_choice,
            temperature=0.05, # Slightly more deterministic
            max_tokens=max_tokens
        )
        
        # Updated prompt to include previous question context
        template = """You are a Knowledge Assistant designed for answering questions specifically from the knowledge base provided to you.

Your task is as follows: give a detailed response for the user query in the user language (e.g., "what are some schemes?" --> "Here is a list of some schemes").

{previous_context}

Ensure your response follows these styles and tone:
* Every answer should be in the **same language** as the user query.
* Use direct, everyday language.
* Maintain a personal and friendly tone, aligned with the user's language.
* Provide detailed responses, with **toll free numbers** and website links wherever applicable. Use section headers like "Description", "Eligibility", or for Marathi: "उद्देशः", "अंतर्भूत घटकः".
* If no relevant information is found, reply with: "For more details contact on 104/102 helpline numbers."
* **Remove duplicate information and provide only one consolidated answer.**

Your goal is to help a citizen understand schemes and their eligibility criteria clearly.

Here is the content you will work with: {context}

Question: {question}

Now perform the task as instructed above.

Answer:"""

        custom_prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question", "previous_context"]
        )
        chain_kwargs = {"prompt": custom_prompt}
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False, # Set to True if you need to inspect sources
            chain_type_kwargs=chain_kwargs
        )
            
    finally:
        for f_path in temp_files_to_clean:
            if os.path.exists(f_path):
                os.unlink(f_path)

def build_rag_chain_with_model_choice(pdf_file, txt_file, groq_api_key, model_choice="llama-3.1-8b-instant", enhanced_mode=True):
    """
    Build RAG chain with selectable model. This is the primary RAG chain builder.
    """
    return build_rag_chain_from_files(pdf_file, txt_file, groq_api_key, enhanced_mode, model_choice)

def detect_language(text):
    """
    Auto-detect language from text
    Returns language code (e.g., 'en', 'hi', 'mr')
    """
    if not TTS_AVAILABLE:
        return 'en'
    
    try:
        clean_text = re.sub(r'[^\w\s]', '', text)
        if len(clean_text.strip()) < 10:
            return 'en'
        
        detected = detect(clean_text)
        
        lang_mapping = {
            'hi': 'hi',
            'mr': 'mr',
            'en': 'en'
        }
        
        return lang_mapping.get(detected, 'en')
        
    except Exception as e:
        print(f"Language detection failed: {e}")
        return 'en'

# --- Query Processing ---
def process_scheme_query_with_retry(rag_chain, user_query, max_retries=3, enable_tts=False, autoplay=False):
    """
    Process query with rate limit handling, caching, and optional TTS.
    Returns: (text_result, audio_html, language_detected, cache_info)
    """
    # Process contextual query (handle "yes" responses)
    processed_query, was_converted = process_contextual_query(user_query)
    
    # Detect language and enforce allowed languages for text processing
    supported_languages = {"en", "hi", "mr"}
    detected_lang = detect_language(processed_query)
    if detected_lang not in supported_languages:
        return (
            "⚠️ Sorry, only Marathi, Hindi, and English are supported. कृपया मराठी, हिंदी अथवा इंग्रजी भाषेत विचारा.",
            "",
            detected_lang,
            {"text_cache": "skipped", "audio_cache": "not_generated"}
        )

    # Check cache first (use processed query for cache)
    query_hash = get_query_hash(processed_query.lower().strip())
    cached_result = get_cached_result(query_hash)
    if cached_result:
        result_text = f"[Cached] {cached_result}"
        cache_status = "cached"
        # Still update context even for cached results
        update_conversation_context(user_query, result_text)
    else:
        cache_status = "not_cached"
        # Check for comprehensive queries
        comprehensive_keywords = [
            "all schemes", "list schemes", "complete list", "सर्व योजना", 
            "total schemes", "how many schemes", "scheme names", "सर्व", "यादी"
        ]
        
        is_comprehensive_query = any(keyword in processed_query.lower() for keyword in comprehensive_keywords)
        
        for attempt in range(max_retries):
            try:
                if is_comprehensive_query:
                    result_text = query_all_schemes_optimized(rag_chain)
                else:
                    # Get previous question context
                    previous_question = get_previous_question_context()
                    previous_context = ""
                    if previous_question and not was_converted:  # Don't show previous context if we converted "yes"
                        previous_context = f"Previous question context: {previous_question}\nPlease consider this context when answering the current question if it's related.\n"
                    
                    # Prepare the query with context
                    query_data = {
                        "question": processed_query,
                        "previous_context": previous_context
                    }
                    
                    result = rag_chain.invoke(query_data)
                    if isinstance(result, dict):
                        result_text = result.get('result', 'No results found.')
                    elif isinstance(result, tuple) and len(result) > 0:
                        # If tuple, take first element as string
                        result_text = str(result[0])
                    else:
                        result_text = str(result)
                
                # Cache successful result
                cache_result(query_hash, result_text)
                cache_status = "cached"
                
                # Update conversation context with original user query and AI response
                update_conversation_context(user_query, result_text)
                break
                
            except Exception as e:
                error_str = str(e)
                
                if "rate_limit_exceeded" in error_str or "413" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # Progressive backoff
                        time.sleep(wait_time)
                        continue
                    else:
                        result_text = "Rate limit reached. Please wait a moment and try again. You can also try a more specific question to reduce processing time."
                        break
                
                elif "Request too large" in error_str:
                    if is_comprehensive_query and attempt == 0:
                        simplified_query = "list main government schemes"
                        try:
                            previous_question = get_previous_question_context()
                            previous_context = ""
                            if previous_question and not was_converted:
                                previous_context = f"Previous question context: {previous_question}\nPlease consider this context when answering the current question if it's related.\n"
                            
                            query_data = {
                                "question": simplified_query,
                                "previous_context": previous_context
                            }
                            result = rag_chain.invoke(query_data)
                            result_text = result.get('result', 'No results found.')
                            result_text = f"[Simplified due to size limits] {result_text}"
                            # Update conversation context
                            update_conversation_context(user_query, result_text)
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
    
    return (result_text, "", detected_lang, {"text_cache": cache_status, "audio_cache": "not_generated"})

# --- Scheme Extraction & Specialized Queries ---
def extract_schemes_from_text(text_content):
    """Helper function to extract schemes from text content using regex patterns."""
    # Normalize text: collapse multiple whitespaces, handle newlines
    text = re.sub(r'\s+', ' ', str(text_content)).replace('\n', ' ')

    # More comprehensive patterns, including common Marathi and English scheme indicators
    # Order matters: more specific patterns first
    patterns = [
        r'\b(?:[A-Z][\w\'-]+(?: [A-Z][\w\'-]+)* )?(?:योजना|कार्यक्रम|अभियान|मिशन|धोरण|निधी|कार्ड|Scheme|Yojana|Programme|Abhiyan|Mission|Initiative|Program|Policy|Fund|Card)\b',
        r'\b(?:Pradhan Mantri|Mukhyamantri|CM|PM|National|Rashtriya|State|Rajya|प्रधानमंत्री|मुख्यमंत्री|राष्ट्रीय|राज्य) (?:[A-Z][\w\'-]+ ?)+', # Schemes starting with titles
        r'\b[A-Z]{2,}(?:-[A-Z]{2,})? Scheme\b', # Acronym schemes like JSY Scheme
        r'\b(?:[०-९]+|[0-9]+)\.\s+([A-Z][\w\s\'-]+(?:योजना|Scheme|कार्यक्रम|Karyakram|अभियान|Abhiyan))', # Numbered list items
        r'•\s+([A-Z][\w\s\'-]+(?:योजना|Scheme|कार्यक्रम|Karyakram|अभियान|Abhiyan))' # Bulleted list items
    ]
    
    extracted_schemes = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE) # IGNORECASE can be useful
        for match in matches:
            # If pattern captures a group (like for numbered/bulleted lists), use the group
            scheme_name = match[1] if isinstance(match, tuple) else match
            # Clean up: strip whitespace, remove trailing punctuation, title case
            cleaned_name = scheme_name.strip().rstrip('.,:;-').title()
            if len(cleaned_name) > 4 and len(cleaned_name.split()) < 10: # Basic sanity checks
                extracted_schemes.add(cleaned_name)

    return sorted(list(extracted_schemes))

def query_all_schemes_optimized(rag_chain):
    """Optimized scheme extractor using targeted queries and regex."""
    try:
        context_query = "Provide a comprehensive list of all government schemes, programs, and yojana mentioned in the documents."
        
        previous_question = get_previous_question_context()
        previous_context = ""
        if previous_question:
            previous_context = f"Previous question context: {previous_question}\nPlease consider this context when answering the current question if it's related.\n"
        
        query_data = {
            "question": context_query,
            "previous_context": previous_context
        }
        
        response = rag_chain.invoke(query_data)
        
        # Handle different response types
        if isinstance(response, dict):
            content_to_parse = response.get('result', '')
        elif isinstance(response, str):
            content_to_parse = response
        elif isinstance(response, tuple) and len(response) > 0:
            content_to_parse = str(response[0])
        else:
            content_to_parse = str(response)
        
        all_extracted_schemes = extract_schemes_from_text(content_to_parse)

        if not all_extracted_schemes:
            return "No government schemes were confidently extracted. The documents might not contain a clear list, or the format is not recognized."

        response_text = f"✅ Found {len(all_extracted_schemes)} potential schemes:\n\n"
        for i, scheme in enumerate(all_extracted_schemes, 1):
            response_text += f"{i}. {scheme}\n"
        response_text += "\n\nℹ️ Note: This list is extracted based on document content. Some names may be partial or inferred."
        return response_text
    except Exception as e:
        return f"Error during optimized scheme query: {str(e)}"

# --- Configuration/Helpers ---
def get_model_options():
    """Return available models with their characteristics."""
    return {
        "llama-3.1-8b-instant": {
            "name": "Llama 3.1 8B (Fast & Efficient)", 
            "description": "Best for quick queries, good for most tasks."
        },
        "llama-3.3-70b-versatile": {
            "name": "Llama 3.3 70B (High Quality)", 
            "description": "Better quality for complex queries, higher latency."
        }
    }
