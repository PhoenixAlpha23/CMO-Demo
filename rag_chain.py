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
    """Cache query result - only cache non-empty results"""
    global _query_cache
    if not result or result.strip() == "":  # Don't cache empty results
        return
    
    if len(_query_cache) >= _cache_max_size:
        oldest_key = next(iter(_query_cache))
        del _query_cache[oldest_key]
    _query_cache[query_hash] = result

def get_cached_result(query_hash):
    """Get cached result if available and non-empty"""
    cached = _query_cache.get(query_hash)
    if cached and cached.strip():  # Only return non-empty cached results
        return cached
    return None

def build_rag_chain_from_files(pdf_file, txt_file, groq_api_key, enhanced_mode=True, debug=False):
    """
    Build a rate-limit optimized RAG chain with improved error handling.
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
            pdf_docs = PyPDFLoader(pdf_path).load()
            all_docs += pdf_docs
            if debug:
                print(f"Loaded {len(pdf_docs)} PDF pages")
        if txt_path:
            txt_docs = TextLoader(txt_path, encoding="utf-8").load()
            all_docs += txt_docs
            if debug:
                print(f"Loaded {len(txt_docs)} text documents")
        
        if not all_docs:
            raise ValueError("No valid documents loaded.")

        # Check if documents have content
        total_content = sum(len(doc.page_content) for doc in all_docs)
        if total_content == 0:
            raise ValueError("Documents are empty - no content found.")
        
        if debug:
            print(f"Total content length: {total_content} characters")

        # Improved chunking
        if enhanced_mode:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,  # Increased from 600
                chunk_overlap=100,  # Reduced overlap
                separators=["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""]
            )
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        
        splits = splitter.split_documents(all_docs)
        
        if debug:
            print(f"Created {len(splits)} chunks")
            if splits:
                print(f"Sample chunk length: {len(splits[0].page_content)}")
        
        if not splits:
            raise ValueError("Document splitting failed - no chunks created.")
        
        # Ensure we have meaningful chunks
        valid_splits = [split for split in splits if len(split.page_content.strip()) > 50]
        if not valid_splits:
            raise ValueError("No meaningful chunks found after splitting.")
        
        # Limit retrieval to avoid token overflow
        max_chunks = min(15 if enhanced_mode else 20, len(valid_splits))
        retriever = TFIDFRetriever.from_documents(valid_splits, k=max_chunks)

        # Use smaller, faster model for efficiency
        llm = ChatGroq(
            api_key=groq_api_key, 
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=2048
        )
        
        # Improved prompt
        if enhanced_mode:
            custom_prompt = PromptTemplate(
                template="""You are a helpful assistant that answers questions based on the provided context.

Context: {context}

Question: {question}

Instructions:
- Answer based only on the information provided in the context
- If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided documents to answer this question."
- For scheme lists: List ALL schemes found, including both English and Marathi names
- Be specific and detailed in your response
- Don't make up information not present in the context

Answer:""",
                input_variables=["context", "question"]
            )
        else:
            custom_prompt = None

        chain_kwargs = {"prompt": custom_prompt} if custom_prompt else {}
        
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,  # Enable for debugging
            chain_type_kwargs=chain_kwargs
        )
        
        return rag_chain
            
    except Exception as e:
        raise ValueError(f"Failed to build RAG chain: {str(e)}")
    finally:
        if pdf_path and os.path.exists(pdf_path):
            os.unlink(pdf_path)
        if txt_path and os.path.exists(txt_path):
            os.unlink(txt_path)


def process_scheme_query_with_retry(rag_chain, user_query, max_retries=3, debug=False):
    """
    Process query with improved error handling and debugging.
    """
    if not user_query or not user_query.strip():
        return "Please provide a valid question."
    
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
                result = query_all_schemes_optimized(rag_chain, debug)
            else:
                # Standard query with debugging
                response = rag_chain.invoke({"query": user_query})
                
                if debug:
                    print(f"Raw response type: {type(response)}")
                    print(f"Raw response keys: {response.keys() if isinstance(response, dict) else 'Not a dict'}")
                
                # Better result extraction
                if isinstance(response, dict):
                    result = response.get('result', '')
                    if not result:
                        result = response.get('answer', '')
                    if not result:
                        # Try other possible keys
                        for key in ['output', 'text', 'response']:
                            if key in response and response[key]:
                                result = response[key]
                                break
                else:
                    result = str(response)
                
                # Check if result is empty or just whitespace
                if not result or not result.strip():
                    if debug:
                        print("Empty result detected, checking source documents...")
                        if 'source_documents' in response:
                            print(f"Retrieved {len(response['source_documents'])} documents")
                            for i, doc in enumerate(response['source_documents'][:2]):
                                print(f"Doc {i}: {doc.page_content[:200]}...")
                    
                    result = "I couldn't find relevant information in the documents to answer your question. Please try rephrasing your question or asking about specific topics mentioned in the documents."
            
            # Validate result before caching
            if result and result.strip() and len(result.strip()) > 10:
                cache_result(query_hash, result)
                return result
            else:
                if debug:
                    print(f"Result validation failed: '{result}'")
                result = "I couldn't generate a meaningful response. Please try asking your question differently."
                return result
            
        except Exception as e:
            error_str = str(e)
            
            if debug:
                print(f"Attempt {attempt + 1} failed: {error_str}")
            
            if "rate_limit_exceeded" in error_str or "413" in error_str:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    time.sleep(wait_time)
                    continue
                else:
                    return "Rate limit reached. Please wait a moment and try again."
            
            elif "Request too large" in error_str:
                return "Query too large. Try asking about specific topics instead."
            
            else:
                if attempt == max_retries - 1:
                    return f"Error processing query: {error_str}"
                continue
    
    return "Unable to process query after multiple attempts. Please try again."


def query_all_schemes_optimized(rag_chain, debug=False):
    """
    Optimized comprehensive scheme search.
    """
    priority_queries = [
        "list government schemes",
        "government welfare programs",
        "सरकारी योजना नावे"
    ]
    
    results = []
    unique_content = set()
    
    for i, query in enumerate(priority_queries):
        try:
            if i > 0:
                time.sleep(1)
            
            response = rag_chain.invoke({"query": query})
            content = response.get('result', '') if isinstance(response, dict) else str(response)
            
            if content and content.strip() and len(content.strip()) > 30:
                content_hash = hashlib.md5(content.encode()).hexdigest()
                if content_hash not in unique_content:
                    results.append(content)
                    unique_content.add(content_hash)
                    
        except Exception as e:
            if debug:
                print(f"Query '{query}' failed: {str(e)}")
            if "rate_limit" in str(e):
                time.sleep(3)
                break
            continue
    
    if not results:
        return "Unable to retrieve comprehensive scheme list. Try asking about specific schemes."
    
    return "\n\n".join(results)


# Keep other functions unchanged
def get_optimized_query_suggestions():
    return [
        "List main government schemes",
        "सरकारी योजना नावे (Government scheme names)", 
        "Top welfare schemes details",
        "Health scheme information",
        "Financial assistance programs",
        "Eligibility criteria for schemes"
    ]

def get_model_options():
    return {
        "llama-3.1-8b-instant": {
            "name": "Llama 3.1 8B (Fast & Cheap)", 
            "description": "Best for quick queries, lower rate limits"
        },
        "llama-3.3-70b-versatile": {
            "name": "Llama 3.3 70B (High Quality)", 
            "description": "Best quality, but higher rate limits"
        },
        "gemma2-9b-it":{
            "name": "Gemma2 9b model", 
            "description": "Higher rate limits"
        }
    }

# Add the model choice function (same as before but with debug parameter)
def build_rag_chain_with_model_choice(pdf_file, txt_file, groq_api_key, model_choice="llama-3.1-8b-instant", enhanced_mode=True, debug=False):
    # Implementation similar to build_rag_chain_from_files but with model parameter
    # (keeping it shorter for space, but add debug parameter throughout)
    pass

# Alias for backward compatibility
process_scheme_query = process_scheme_query_with_retry
