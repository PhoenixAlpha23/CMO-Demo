import tempfile
import os
import time
import hashlib
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.retrievers import TFIDFRetriever
from langchain.prompts import PromptTemplate
import re
# Simple in-memory cache to avoid repeated API calls
_query_cache = {}
_cache_max_size = 50


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
            custom_prompt = PromptTemplate(
                template="""Based on the context, answer concisely, to the point and comprehensively.do not mention anything irrelevant(eg. don't mention 'based upon given context')

For list of schemes:Extract all schemes or program names from the in the documents, both English and Marathi names. A valid name typically starts with a capital letter and includes terms like Scheme, Yojana, Program, Programme, or योजना. Also match if presented as:

Numbered list (e.g., 1. Digital India Programme)

Bullet point (e.g., • Skill India Mission)

Dash point (e.g., - Startup India Initiative)

Use regex patterns to ensure accurate extraction, but remove unrequired punctuation marks and present a clean output.
Instructions: Always reply in the input language(eg. Marathi,English or Hindi).
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


def process_scheme_query_with_retry(rag_chain, user_query, max_retries=3):
    """
    Process query with rate limit handling and caching.
    """
    # Check cache first
    query_hash = get_query_hash(user_query.lower().strip())
    cached_result = get_cached_result(query_hash)
    if cached_result:
        return f"[Cached] {cached_result}"
    
    # Check for comprehensive queries
    comprehensive_keywords = [
        "all schemes", "list schemes", "complete list", "सर्व योजना", 
        "total schemes", "how many schemes", "scheme names", "सर्व", "यादी"
    ]
    
    is_comprehensive_query = any(keyword in user_query.lower() for keyword in comprehensive_keywords)
    
    for attempt in range(max_retries):
        try:
            if is_comprehensive_query:
                result = query_all_schemes_optimized(rag_chain)
            else:
                result = rag_chain.invoke({"query": user_query})
                result = result.get('result', 'No results found.')
            
            # Cache successful result
            cache_result(query_hash, result)
            return result
            
        except Exception as e:
            error_str = str(e)
            
            if "rate_limit_exceeded" in error_str or "413" in error_str:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Progressive backoff
                    time.sleep(wait_time)
                    continue
                else:
                    return f"Rate limit reached. Please wait a moment and try again. You can also try a more specific question to reduce processing time."
            
            elif "Request too large" in error_str:
                # Try with a simpler, shorter query
                if is_comprehensive_query and attempt == 0:
                    simplified_query = "list main government schemes"
                    try:
                        result = rag_chain.invoke({"query": simplified_query})
                        result = result.get('result', 'No results found.')
                        return f"[Simplified due to size limits] {result}"
                    except:
                        pass
                
                return "Query too large for current model. Try asking about specific schemes or categories instead of requesting all schemes at once."
            
            else:
                return f"Error processing query: {error_str}"
    
    return "Unable to process query after multiple attempts. Please try a simpler question."

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


# Additional helper functions for your Streamlit app
def get_model_options():
    """
    Return available models with their characteristics.
    """
    return {
        "llama-3.1-8b-instant": {
            "name": "Llama 3.1 8B (Fast & Cheap)", 
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


# Alias for backward compatibility
process_scheme_query = process_scheme_query_with_retry
