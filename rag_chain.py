import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.retrievers import TFIDFRetriever
from langchain.prompts import PromptTemplate

# Try to import BM25 and Ensemble retrievers, fallback if not available
try:
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    ADVANCED_RETRIEVERS_AVAILABLE = True
except ImportError:
    ADVANCED_RETRIEVERS_AVAILABLE = False

def build_rag_chain_from_files(pdf_file, txt_file, groq_api_key, enhanced_mode=True):
    """
    Build a RAG chain for comprehensive scheme retrieval.
    
    Args:
        pdf_file: PDF file object from Streamlit
        txt_file: TXT file object from Streamlit  
        groq_api_key: API key for Groq
        enhanced_mode: If True, uses enhanced retrieval for better coverage
    
    Returns:
        RAG chain object for querying documents
    """
    pdf_path = txt_path = None
    if not (pdf_file or txt_file):
        raise ValueError("At least one file (PDF or TXT) must be provided.")
    
    try:
        # Save uploaded files to temporary files
        if pdf_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(pdf_file.read())
                pdf_path = tmp_pdf.name
        if txt_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_txt:
                tmp_txt.write(txt_file.read())
                txt_path = tmp_txt.name

        # Load and process documents
        all_docs = []
        if pdf_path:
            all_docs += PyPDFLoader(pdf_path).load()
        if txt_path:
            all_docs += TextLoader(txt_path, encoding="utf-8").load()
        
        if not all_docs:
            raise ValueError("No valid documents loaded from PDF or TXT.")

        if enhanced_mode and ADVANCED_RETRIEVERS_AVAILABLE:
            # Enhanced chunking strategy
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=300,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            splits = splitter.split_documents(all_docs)
            
            # Ensemble retriever for comprehensive results
            tfidf_retriever = TFIDFRetriever.from_documents(splits, k=40)
            bm25_retriever = BM25Retriever.from_documents(splits, k=40)
            retriever = EnsembleRetriever(
                retrievers=[tfidf_retriever, bm25_retriever],
                weights=[0.5, 0.5]
            )
            
        elif enhanced_mode:
            # Enhanced mode without ensemble (fallback)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=300,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            splits = splitter.split_documents(all_docs)
            retriever = TFIDFRetriever.from_documents(splits, k=40)
            
        else:
            # Original simple approach
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = splitter.split_documents(all_docs)
            retriever = TFIDFRetriever.from_documents(splits, k=30)

        # Custom prompt for comprehensive results
        if enhanced_mode:
            custom_prompt = PromptTemplate(
                template="""You are an expert assistant for government scheme information with knowledge of both English and Marathi schemes.

Based on the provided context, answer the user's question comprehensively and accurately.

IMPORTANT GUIDELINES:
- For queries about "all schemes", "सर्व योजना", or "list schemes": Provide ALL schemes mentioned in the documents
- Include both English and Marathi scheme names
- Provide clear, organized information
- If you find partial scheme information, include it
- Be thorough and don't limit your response
- Maintain accuracy and cite specific details when available

Context from documents:
{context}

User Question: {question}

Detailed Answer:""",
                input_variables=["context", "question"]
            )
        else:
            custom_prompt = None

        # Initialize LLM
        llm = ChatGroq(
            api_key=groq_api_key, 
            model="llama-3.3-70b-versatile",
            temperature=0.2
        )
        
        # Build RAG chain
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
        # Clean up temp files
        if pdf_path and os.path.exists(pdf_path):
            os.unlink(pdf_path)
        if txt_path and os.path.exists(txt_path):
            os.unlink(txt_path)


def process_scheme_query(rag_chain, user_query):
    """
    Process user query with automatic enhancement for comprehensive results.
    
    Args:
        rag_chain: RAG chain object
        user_query: User's input query
    
    Returns:
        Enhanced query results
    """
    # Check if user is asking for comprehensive listing
    comprehensive_keywords = [
        "all schemes", "list schemes", "complete list", "सर्व योजना", 
        "total schemes", "how many schemes", "scheme names", "सर्व", "यादी"
    ]
    
    is_comprehensive_query = any(keyword in user_query.lower() for keyword in comprehensive_keywords)
    
    if is_comprehensive_query:
        # Use multiple query strategy for comprehensive results
        return query_all_schemes(rag_chain)
    else:
        # Regular query processing
        try:
            result = rag_chain.invoke({"query": user_query})
            return result.get('result', 'No results found.')
        except Exception as e:
            return f"Error processing query: {str(e)}"


def query_all_schemes(rag_chain):
    """
    Comprehensive scheme search using multiple query strategies.
    
    Args:
        rag_chain: The RAG chain object
    
    Returns:
        Comprehensive list of all schemes
    """
    # Multiple targeted queries for maximum coverage
    queries = [
        "list all government schemes available",
        "complete list of schemes and programs", 
        "सर्व सरकारी योजना नावे आणि तपशील",  # All govt schemes names and details
        "scheme names with eligibility criteria",
        "welfare schemes and benefits programs",
        "योजना यादी आणि फायदे",  # Scheme list and benefits
        "all 72 schemes mentioned in documents"
    ]
    
    results = []
    unique_content = set()
    
    for query in queries:
        try:
            result = rag_chain.invoke({"query": query})
            content = result.get('result', '')
            
            # Add only unique content
            if content and content not in unique_content and len(content) > 50:
                results.append(content)
                unique_content.add(content)
        except Exception as e:
            continue
    
    if not results:
        return "No schemes found. Please check if your documents contain scheme information."
    
    # Combine results
    combined_results = "\n\n--- Additional Search Results ---\n\n".join(results)
    
    # Final synthesis for comprehensive output
    synthesis_query = f"""
    Based on all the search results below, provide a comprehensive, organized list of ALL government schemes mentioned. 
    Remove any duplicates and present the information clearly:
    
    SEARCH RESULTS:
    {combined_results}
    
    Please provide:
    1. Complete list of all unique schemes
    2. Organize by categories if possible
    3. Include both English and Marathi scheme names
    4. Mention total count if possible
    
    COMPREHENSIVE SCHEME LIST:
    """
    
    try:
        final_result = rag_chain.invoke({"query": synthesis_query})
        final_content = final_result.get('result', '')
        
        if final_content and len(final_content) > len(max(results, key=len)):
            return final_content
        else:
            # Fallback to combined results if synthesis didn't improve
            return f"COMPREHENSIVE SCHEME INFORMATION:\n\n{combined_results}"
            
    except Exception:
        return f"COMPREHENSIVE SCHEME INFORMATION:\n\n{combined_results}"


def get_optimized_query_suggestions():
    """
    Returns suggested queries optimized for comprehensive scheme retrieval.
    """
    return [
        "List all government schemes available",
        "सर्व सरकारी योजना दाखवा (Show all government schemes)", 
        "Complete list of 72 schemes from documents",
        "Welfare schemes and eligibility criteria",
        "Financial assistance programs details",
        "आरोग्य योजना माहिती (Health scheme information)"
    ]
