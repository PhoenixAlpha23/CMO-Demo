import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os

def build_rag_chain_from_files(pdf_file, txt_file, groq_api_key):
    # Save to temp files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(pdf_file.read())
        pdf_path = tmp_pdf.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_txt:
        tmp_txt.write(txt_file.read())
        txt_path = tmp_txt.name

    # Load and split
    eng_docs = PyPDFLoader(pdf_path).load()
    mar_docs = TextLoader(txt_path, encoding="utf-8").load()
    all_docs = eng_docs + mar_docs

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = splitter.split_documents(all_docs)

    # Embedding & FAISS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectordb = FAISS.from_documents(splits, embeddings)

    # LLM & RAG
    llm = ChatGroq(api_key=groq_api_key, model="llama-3-3-70b-versatile")
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5})

    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
