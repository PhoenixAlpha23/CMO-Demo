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
                template="""Based on the context, answer concisely but comprehensively.

For scheme lists: List ALL schemes found, both English and Marathi names.

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


def query_all_schemes_optimized(rag_chain):
    """
    Optimized comprehensive scheme search to avoid rate limits.
    """
    # Use fewer, more targeted queries
    priority_queries = [
        "government schemes list names",
        "सरकारी योजना नावे",  # Government scheme names in Marathi
        "welfare programs benefits"
    ]
    
    results = []
    unique_content = set()
    
    for i, query in enumerate(priority_queries):
        try:
            # Add delay between queries to respect rate limits
            if i > 0:
                time.sleep(1)
            
            result = rag_chain.invoke({"query": query})
            content = result.get('result', '')
            
            if content and content not in unique_content and len(content) > 30:
                results.append(content)
                unique_content.add(content)
                
        except Exception as e:
            if "rate_limit" in str(e):
                time.sleep(3)  # Wait longer on rate limit
                break
            continue
    
    if not results:
        return "Unable to retrieve comprehensive scheme list due to rate limits. Try asking about specific schemes."
    
    # Simple combination without additional API call
    if len(results) == 1:
        return results[0]
    else:
        combined = "COMPREHENSIVE SCHEME INFORMATION:\n\n"
        for i, result in enumerate(results, 1):
            combined += f"=== Search Result {i} ===\n{result}\n\n"
        return combined


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
        "llama-3.1-70b-versatile": {
            "name": "Llama 3.1 70B (Balanced)", 
            "description": "Good balance of quality and speed"
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
            chunk_size, max_chunks, max_tokens = 500, 12, 1500
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
            temperature=0.1,
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
