import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.retrievers import TFIDFRetriever

def build_rag_chain_from_files(pdf_file, txt_file, groq_api_key):
    """
    Build a RAG chain using the simplest possible retrieval method (TF-IDF)
    to avoid dependency issues with more complex embedding models.
    """
    # Save to temp files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(pdf_file.read())
        pdf_path = tmp_pdf.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_txt:
        tmp_txt.write(txt_file.read())
        txt_path = tmp_txt.name
    
    try:
        # Load and split documents
        eng_docs = PyPDFLoader(pdf_path).load()
        mar_docs = TextLoader(txt_path, encoding="utf-8").load()
        all_docs = eng_docs + mar_docs
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = splitter.split_documents(all_docs)
        
        # Use TF-IDF Retriever - the simplest possible option that doesn't require ML models
        retriever = TFIDFRetriever.from_documents(splits, k=5)
        
        # LLM & RAG
        llm = ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile")
        
        # Clean up temp files
        os.unlink(pdf_path)
        os.unlink(txt_path)
        
        return RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff",  # Simplest chain type
            retriever=retriever, 
            return_source_documents=False
        )
    
    except Exception as e:
        # Clean up temp files even if there's an error
        try:
            os.unlink(pdf_path)
            os.unlink(txt_path)
        except:
            pass
        
        print(f"Error building RAG chain: {e}")
        raise ValueError(f"Failed to build RAG chain: {str(e)}")
