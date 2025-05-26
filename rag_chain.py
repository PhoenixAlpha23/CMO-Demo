import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.retrievers import TFIDFRetriever

def build_rag_chain_from_files(pdf_file, txt_file, groq_api_key):
    """
    Build a RAG chain using the simplest possible retrieval method (TF-IDF)
    to avoid dependency issues with more complex embedding models.
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

        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = splitter.split_documents(all_docs)

        # Use TF-IDF Retriever - the simplest possible option that doesn't require ML models
        retriever = TFIDFRetriever.from_documents(splits, k=20)

        # LLM & RAG
        llm = ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant")

        return RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff",  # Simplest chain type
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
