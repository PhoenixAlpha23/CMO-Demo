import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.retrievers import TFIDFRetriever, BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate

def build_rag_chain_from_files(pdf_file, txt_file, groq_api_key, enhanced_mode=True):
    """
    Build a RAG chain for Streamlit app with enhanced retrieval capabilities.
    
    Args:
        pdf_file: PDF file object from Streamlit file uploader
        txt_file: TXT file object from Streamlit file uploader  
        groq_api_key: API key for Groq
        enhanced_mode: If True, uses enhanced retrieval for better scheme coverage
    
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

        if enhanced_mode:
            # Enhanced chunking strategy for better scheme coverage
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
            
            # Custom prompt for comprehensive scheme listing
            custom_prompt = PromptTemplate(
                template="""You are an expert assistant for government scheme information.

Based on the provided context, answer the user's question comprehensively.

IMPORTANT: For queries about "all schemes" or "list of schemes":
- List ALL schemes mentioned in the documents
- Include both English and Marathi schemes  
- Provide clear scheme names
- Don't limit your response - be as complete as possible
- If you see partial scheme information, include it

Context:
{context}

Question: {question}

Answer:""",
                input_variables=["context", "question"]
            )
            
            llm = ChatGroq(
                api_key=groq_api_key, 
                model="llama-3.3-70b-versatile",
                temperature=0.2
            )
            
            return RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=False,
                chain_type_kwargs={"prompt": custom_prompt}
            )
        
        else:
            # Original simple approach (fallback)
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = splitter.split_documents(all_docs)
            retriever = TFIDFRetriever.from_documents(splits, k=30)  # Increased from 20
            llm = ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile")
            
            return RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff",
                retriever=retriever, 
                return_source_documents=False
            )
            
    except Exception as e:
        raise ValueError(f"Failed to build RAG chain: {str(e)}")
    finally:
        # Clean up temp files
        if pdf_path and os.path.exists(pdf_path):
            os.unlink(pdf_path)
        if txt_path and os.path.exists(txt_path):
            os.unlink(txt_path)


def query_all_schemes(rag_chain, language_hint="both"):
    """
    Helper function to get comprehensive scheme list using multiple query strategies.
    
    Args:
        rag_chain: The RAG chain object
        language_hint: "english", "marathi", or "both"
    
    Returns:
        Comprehensive list of schemes
    """
    # Multiple targeted queries for better coverage
    base_queries = [
        "list all government schemes",
        "complete list of schemes and programs", 
        "सर्व सरकारी योजना नावे",  # All government scheme names in Marathi
        "scheme names with eligibility criteria",
        "welfare schemes and benefits programs"
    ]
    
    if language_hint == "english":
        queries = [q for q in base_queries if not any(ord(char) > 127 for char in q)]
    elif language_hint == "marathi":
        queries = [q for q in base_queries if any(ord(char) > 127 for char in q)] + ["योजना", "अनुदान"]
    else:
        queries = base_queries
    
    results = []
    for query in queries:
        try:
            result = rag_chain.invoke({"query": query})
            if result and 'result' in result:
                results.append(result['result'])
        except Exception as e:
            continue
    
    # Combine and deduplicate results
    if results:
        combined = "\n\n=== COMPREHENSIVE SCHEME LIST ===\n\n" + "\n\n".join(results)
        
        # Final synthesis
        synthesis_query = f"""
        Based on all the information below, provide a final consolidated list of ALL unique government schemes mentioned. 
        Remove duplicates and organize clearly:
        
        {combined}
        
        Provide the complete consolidated list:
        """
        
        try:
            final_result = rag_chain.invoke({"query": synthesis_query})
            return final_result.get('result', combined)
        except:
            return combined
    
    return "No schemes found. Please check your documents."


# Streamlit app integration helpers
def get_optimized_query_suggestions():
    """
    Returns suggested queries optimized for comprehensive scheme retrieval.
    """
    return [
        "List all government schemes available",
        "Complete list of schemes in both languages", 
        "सर्व सरकारी योजना नावे (All government schemes)",
        "Show me all 72 schemes from the documents",
        "Welfare schemes and eligibility criteria",
        "Financial assistance programs list"
    ]

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
        "total schemes", "how many schemes", "scheme names"
    ]
    
    is_comprehensive_query = any(keyword in user_query.lower() for keyword in comprehensive_keywords)
    
    if is_comprehensive_query:
        # Use the specialized comprehensive search
        return query_all_schemes(rag_chain)
    else:
        # Regular query processing
        try:
            result = rag_chain.invoke({"query": user_query})
            return result.get('result', 'No results found.')
        except Exception as e:
            return f"Error processing query: {str(e)}"
