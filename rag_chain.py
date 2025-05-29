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

# Keywords for enhanced scheme extraction
MARATHI_KEYWORDS = [
    "उद्देशः", "अंतर्भूत घटक", "हेल्प लाईन क्र", "योजना", "लाभार्थी", 
    "पात्रता", "निकष", "अर्ज", "कागदपत्रे", "माहिती"
]

ENGLISH_KEYWORDS = [
    "Description:", "Eligibility:", "Target Group:", "Inclusion Criteria:",
    "Exclusion Criteria:", "Benefits:", "Helpline:", "Documents Required:"
]

# Default prompt template for the LLM
DEFAULT_PROMPT_TEMPLATE = """Based on the provided context, answer the question concisely and comprehensively. Focus on the specific information requested without unnecessary elaborations.

For scheme searches, extract relevant information including:
- English documents: Description, Eligibility, Target Group, Inclusion Criteria, Exclusion Criteria, Benefits
- Marathi documents: उद्देशः (Purpose), अंतर्भूत घटकः (Included Components), हेल्प लाईन क्र (Helpline Number), website links, and toll-free numbers

Always respond in the same language as the input question.

Context: {context}

Question: {question}

Answer:"""

def get_query_hash(query_text):
    """Generate a hash for caching queries"""
    return hashlib.md5(query_text.encode()).hexdigest()

def cache_result(query_hash, result):
    """Cache query result"""
    global _query_cache
    if len(_query_cache) >= _cache_max_size:
        oldest_key = next(iter(_query_cache))
        del _query_cache[oldest_key]
    _query_cache[query_hash] = result

def get_cached_result(query_hash):
    """Get cached result if available"""
    return _query_cache.get(query_hash)

def build_rag_chain_from_files(pdf_file, txt_file, groq_api_key, enhanced_mode=True):
    """Build a rate-limit optimized RAG chain."""
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

        # Optimized chunking
        chunk_size = 600 if enhanced_mode else 800
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        splits = splitter.split_documents(all_docs)
        
        # Limit retrieval to avoid token overflow
        max_chunks = 15 if enhanced_mode else 20
        retriever = TFIDFRetriever.from_documents(splits, k=min(max_chunks, len(splits)))

        # Use smaller, faster model for efficiency
        llm = ChatGroq(
            api_key=groq_api_key, 
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=2000
        )
        
        # Use default prompt template
        custom_prompt = PromptTemplate(
            template=DEFAULT_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        ) if enhanced_mode else None

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

def build_enhanced_response(query_result, context=""):
    """Build enhanced response with fallback for out-of-context queries"""
    
    # Check if query is about something not in documents
    irrelevant_patterns = [
        r"(?i)not (found|available|provided|mentioned|present)",
        r"(?i)no information",
        r"(?i)cannot (find|locate|see)",
        r"माहिती उपलब्ध नाही",
        r"सापडले नाही"
    ]
    
    for pattern in irrelevant_patterns:
        if re.search(pattern, query_result):
            return ("⚠️ This information was not provided in the documents. "
                   "Please ask about schemes and details mentioned in the uploaded files.")

    # Extract and structure the response
    schemes = extract_all_scheme_names(query_result)
    if schemes:
        enhanced_response = ""
        for scheme in schemes:
            details = extract_scheme_details(context + query_result, scheme)
            
            enhanced_response += f"\n🔷 {scheme}\n"
            
            # Add description
            if details["description"]["en"] or details["description"]["mr"]:
                enhanced_response += "\n📋 Description/उद्देश:\n"
                for desc in details["description"]["en"]:
                    enhanced_response += f"- {desc}\n"
                for desc in details["description"]["mr"]:
                    enhanced_response += f"- {desc}\n"
            
            # Add eligibility
            if details["eligibility"]["en"] or details["eligibility"]["mr"]:
                enhanced_response += "\n✅ Eligibility/पात्रता:\n"
                for elig in details["eligibility"]["en"]:
                    enhanced_response += f"- {elig}\n"
                for elig in details["eligibility"]["mr"]:
                    enhanced_response += f"- {elig}\n"
            
            # Add benefits
            if details["benefits"]["en"] or details["benefits"]["mr"]:
                enhanced_response += "\n💫 Benefits/लाभ:\n"
                for benefit in details["benefits"]["en"]:
                    enhanced_response += f"- {benefit}\n"
                for benefit in details["benefits"]["mr"]:
                    enhanced_response += f"- {benefit}\n"
            
            # Add contact information
            if details["helpline"] or details["website"]:
                enhanced_response += "\n📞 Contact Information:\n"
                if details["helpline"]:
                    enhanced_response += f"Helpline: {', '.join(details['helpline'])}\n"
                if details["website"]:
                    enhanced_response += f"Website: {', '.join(details['website'])}\n"
            
            enhanced_response += "\n" + "-"*50 + "\n"
        
        return enhanced_response
    
    return query_result

def process_scheme_query_with_retry(rag_chain, user_query, max_retries=3):
    """Process query with rate limit handling, caching, and enhanced response building."""
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
                result_text = query_all_schemes_optimized(rag_chain)
            else:
                result = rag_chain.invoke({"query": user_query})
                result_text = result.get('result', 'No results found.')
            
            # Enhance the response with structured information
            enhanced_result = build_enhanced_response(result_text)
            
            cache_result(query_hash, enhanced_result)
            return enhanced_result
            
        except Exception as e:
            error_str = str(e)
            
            if "rate_limit_exceeded" in error_str or "413" in error_str:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    time.sleep(wait_time)
                    continue
                else:
                    return "Rate limit reached. Please wait and try again with a more specific question."
            
            elif "Request too large" in error_str:
                if is_comprehensive_query and attempt == 0:
                    try:
                        result = rag_chain.invoke({"query": "list main government schemes"})
                        enhanced_result = build_enhanced_response(result.get('result', 'No results found.'))
                        return f"[Simplified] {enhanced_result}"
                    except:
                        pass
                return "Query too large. Try asking about specific schemes instead."
            
            else:
                return f"Error processing query: {error_str}"
    
    return "Unable to process query. Please try a simpler question."

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
    helpline_pattern = r'(?:हेल्प लाईन|Helpline|टोल फ्री|Toll Free).*?([0-9\-]{8,})'
    helpline_matches = re.findall(helpline_pattern, text, re.IGNORECASE)
    details["helpline"] = helpline_matches

    # Process English sections
    for section in ENGLISH_KEYWORDS:
        pattern = f"{section}(.*?)(?={'|'.join(ENGLISH_KEYWORDS)}|$)"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            key = section.lower().strip(":")
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
        if matches:
            details[key]["mr"] = [m.strip() for m in matches if m.strip()]

    return details

def extract_all_scheme_names(text):
    """Extract scheme names using enhanced search patterns."""
    # Enhanced patterns for English documents using keywords
    english_patterns = [
        r'\b[A-Z][a-zA-Z\s]*(?:Scheme|Programme?|Mission|Yojana|Initiative)\b',
        r'\b(?:PM|Pradhan Mantri|National|State)\s+[A-Z][a-zA-Z\s]*(?:Scheme|Programme?|Mission)\b'
    ]
    
    # Add patterns for English keywords
    for keyword in ENGLISH_KEYWORDS:
        english_patterns.append(f'{keyword}\\s*([^\\n]+)')
    
    # Enhanced patterns for Marathi documents using keywords
    marathi_patterns = [
        r'\b[अ-ह][अ-ह\s]*(?:योजना|कार्यक्रम|अभियान|सेवा|केंद्र)\b',
        r'(?:www\.|https?://)[^\s]+',  # Website links
        r'\b(?:1800|18[0-9]{2})[0-9\-\s]+\b'  # Toll-free numbers
    ]
    
    # Add patterns for Marathi keywords
    for keyword in MARATHI_KEYWORDS:
        marathi_patterns.append(f'{keyword}[:\\s]*([^\\n]+)')
    
    all_patterns = english_patterns + marathi_patterns
    matches = []
    
    for pattern in all_patterns:
        matches.extend(re.findall(pattern, text, re.IGNORECASE))
    
    # Clean and deduplicate
    cleaned = list(set([match.strip().rstrip(':.,;') for match in matches if len(match.strip()) > 3]))
    return sorted(cleaned)

def query_all_schemes_optimized(rag_chain):
    """Optimized scheme extractor with enhanced keyword search."""
    priority_queries = [
        "Extract Description, Eligibility, Target Group, Inclusion Criteria, Exclusion Criteria, Benefits from schemes",
        "उद्देशः, अंतर्भूत घटकः, हेल्प लाईन क्र सह योजना माहिती",
        "List government schemes with eligibility and benefits",
        "सरकारी योजना नावे व अर्हता निकष"
    ]
    
    all_extracted_schemes = set()
    seen_texts = set()
    
    for i, query in enumerate(priority_queries):
        try:
            if i > 0:
                time.sleep(1)
            
            result = rag_chain.invoke({"query": query})
            content = result.get('result', '') if isinstance(result, dict) else str(result)
            
            if content and content not in seen_texts and len(content.strip()) > 30:
                seen_texts.add(content)
                schemes = extract_all_scheme_names(content)
                all_extracted_schemes.update(schemes)
        
        except Exception as e:
            if "rate_limit" in str(e).lower():
                time.sleep(3)
            continue

    if not all_extracted_schemes:
        return "No schemes found. Please refine your query."

    schemes_list = sorted(list(all_extracted_schemes))
    response = f"Found {len(schemes_list)} schemes/details:\n\n"
    for i, scheme in enumerate(schemes_list, 1):
        response += f"{i}. {scheme}\n"

    return response

def get_optimized_query_suggestions():
    """Rate-limit friendly query suggestions."""
    return [
        "List main government schemes",
        "सरकारी योजना नावे",
        "Scheme eligibility criteria",
        "योजना अर्हता निकष",
        "Benefits and target groups",
        "हेल्पलाइन आणि संपर्क माहिती"
    ]

def get_model_options():
    """Return available models with their characteristics."""
    return {
        "llama-3.1-8b-instant": {
            "name": "Llama 3.1 8B (Fast & Cheap)", 
            "description": "Best for quick queries"
        },
        "llama-3.3-70b-versatile": {
            "name": "Llama 3.3 70B (High Quality)", 
            "description": "Best quality, higher rate limits"
        }
    }

def build_rag_chain_with_model_choice(pdf_file, txt_file, groq_api_key, model_choice="llama-3.1-8b-instant", enhanced_mode=True):
    """Build RAG chain with selectable model."""
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

        # Model-specific parameters
        model_params = {
            "llama-3.1-8b-instant": {"chunk_size": 800, "max_chunks": 12, "max_tokens": 1500},
            "llama-3.1-70b-versatile": {"chunk_size": 700, "max_chunks": 18, "max_tokens": 2500},
            "llama-3.3-70b-versatile": {"chunk_size": 800, "max_chunks": 20, "max_tokens": 3000}
        }
        
        params = model_params.get(model_choice, model_params["llama-3.1-8b-instant"])
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=params["chunk_size"],
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        splits = splitter.split_documents(all_docs)
        retriever = TFIDFRetriever.from_documents(splits, k=min(params["max_chunks"], len(splits)))

        llm = ChatGroq(
            api_key=groq_api_key, 
            model=model_choice,
            temperature=0.0,
            max_tokens=params["max_tokens"]
        )
        
        custom_prompt = PromptTemplate(
            template=DEFAULT_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        ) if enhanced_mode else None
        
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

# Helper function for extracting scheme information
def extract_schemes_from_text(text):
    """Wrapper function for backward compatibility."""
    return extract_all_scheme_names(text)

# Alias for backward compatibility
process_scheme_query = process_scheme_query_with_retry
