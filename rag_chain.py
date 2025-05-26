import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.retrievers import TFIDFRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.prompts import PromptTemplate

def build_enhanced_rag_chain_from_files(pdf_file, txt_file, groq_api_key):
    """
    Enhanced RAG chain with better retrieval for comprehensive scheme listing.
    """
    pdf_path = txt_path = None
    if not (pdf_file or txt_file):
        raise ValueError("At least one file (PDF or TXT) must be provided.")
    
    try:
        # Save to temp files if they exist
        if pdf_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(pdf_file.read())
                pdf_path = tmp_pdf.name
        if txt_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_txt:
                tmp_txt.write(txt_file.read())
                txt_path = tmp_txt.name

        # Load and split documents
        all_docs = []
        if pdf_path:
            all_docs += PyPDFLoader(pdf_path).load()
        if txt_path:
            all_docs += TextLoader(txt_path, encoding="utf-8").load()
        
        if not all_docs:
            raise ValueError("No valid documents loaded from PDF or TXT.")

        # Strategy 1: Smaller chunks with more overlap for better granularity
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Smaller chunks
            chunk_overlap=300,  # More overlap to preserve context
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        splits = splitter.split_documents(all_docs)
        
        # Strategy 2: Use ensemble retriever with multiple retrieval methods
        # TF-IDF Retriever with higher k
        tfidf_retriever = TFIDFRetriever.from_documents(splits, k=40)
        
        # BM25 Retriever (better for keyword matching)
        bm25_retriever = BM25Retriever.from_documents(splits, k=40)
        
        # Ensemble retriever combining both methods
        ensemble_retriever = EnsembleRetriever(
            retrievers=[tfidf_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )
        
        # Strategy 3: Custom prompt for comprehensive listing
        custom_prompt = PromptTemplate(
            template="""You are an expert assistant helping users find information about government schemes.
            
Based on the provided documents, answer the user's question comprehensively.

For queries asking for "all schemes" or "list of schemes":
1. List ALL schemes mentioned in the documents
2. Include both English and Marathi schemes
3. Provide scheme names clearly
4. If you find references to schemes but don't have complete information, mention them anyway
5. Organize by language/document if helpful

Context from documents:
{context}

Question: {question}

Comprehensive Answer:""",
            input_variables=["context", "question"]
        )
        
        # LLM with higher temperature for more comprehensive responses
        llm = ChatGroq(
            api_key=groq_api_key, 
            model="llama-3.3-70b-versatile",
            temperature=0.3  # Slightly higher for more comprehensive responses
        )
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=ensemble_retriever,
            return_source_documents=True,  # Helpful for debugging
            chain_type_kwargs={"prompt": custom_prompt}
        )
        
    except Exception as e:
        raise ValueError(f"Failed to build enhanced RAG chain: {str(e)}")
    finally:
        # Clean up temp files
        if pdf_path and os.path.exists(pdf_path):
            os.unlink(pdf_path)
        if txt_path and os.path.exists(txt_path):
            os.unlink(txt_path)


def build_multi_query_rag_chain(pdf_file, txt_file, groq_api_key):
    """
    Alternative approach: Use multiple targeted queries to ensure comprehensive coverage.
    """
    # Build the basic chain first
    rag_chain = build_enhanced_rag_chain_from_files(pdf_file, txt_file, groq_api_key)
    
    def comprehensive_scheme_search(query):
        """
        Performs multiple searches with different query variations to maximize coverage.
        """
        # Multiple query variations to capture different schemes
        query_variations = [
            query,  # Original query
            "government schemes programs benefits",
            "scheme name list अनुदान योजना",  # Include Marathi terms
            "eligibility criteria schemes",
            "financial assistance programs",
            "welfare schemes benefits"
        ]
        
        all_results = []
        seen_content = set()
        
        for variant_query in query_variations:
            try:
                result = rag_chain.invoke({"query": variant_query})
                content = result.get('result', '')
                
                # Avoid duplicate content
                if content and content not in seen_content:
                    all_results.append(content)
                    seen_content.add(content)
            except Exception as e:
                print(f"Error with query '{variant_query}': {e}")
                continue
        
        # Combine all results
        combined_result = "\n\n--- Additional Search Results ---\n\n".join(all_results)
        
        # Final synthesis query
        synthesis_prompt = f"""
        Based on all the following search results, provide a comprehensive list of ALL schemes mentioned:
        
        {combined_result}
        
        Please consolidate this information into a complete, organized list of all government schemes, 
        removing any duplicates and organizing them clearly.
        """
        
        try:
            final_result = rag_chain.invoke({"query": synthesis_prompt})
            return final_result.get('result', combined_result)
        except Exception:
            return combined_result
    
    return comprehensive_scheme_search


# Alternative: Direct document parsing approach
def build_direct_parsing_rag_chain(pdf_file, txt_file, groq_api_key):
    """
    Alternative approach: Parse documents more directly to ensure no schemes are missed.
    """
    pdf_path = txt_path = None
    
    try:
        # Save and load documents (same as before)
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

        # Create larger chunks to keep schemes together
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Larger chunks
            chunk_overlap=500,
            separators=["\n\n", "\n", ".", "!"]
        )
        splits = splitter.split_documents(all_docs)
        
        # Use very high k value to retrieve most relevant chunks
        retriever = TFIDFRetriever.from_documents(splits, k=min(50, len(splits)))
        
        # LLM
        llm = ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile")
        
        # Custom prompt emphasizing completeness
        complete_listing_prompt = PromptTemplate(
            template="""You are tasked with providing a COMPLETE and COMPREHENSIVE list of schemes.

IMPORTANT INSTRUCTIONS:
- You MUST list ALL schemes mentioned in the provided context
- Do NOT limit your response - include every single scheme you find
- Include schemes in both English and Marathi
- If a scheme is mentioned partially, include it anyway
- Organize the output clearly but prioritize completeness over formatting

Context (contains information about multiple government schemes):
{context}

Question: {question}

COMPLETE LIST OF ALL SCHEMES (do not limit or summarize):""",
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": complete_listing_prompt}
        )
        
    finally:
        if pdf_path and os.path.exists(pdf_path):
            os.unlink(pdf_path)
        if txt_path and os.path.exists(txt_path):
            os.unlink(txt_path)
