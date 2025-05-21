import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os
import torch

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
    
    # Embedding & FAISS - try multiple approaches
    def create_embeddings():
        # Try approach 1: Set PyTorch device globally first
        torch.set_default_device('cpu')
        try:
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"  # Different model that might work better
            )
        except Exception as e1:
            print(f"First embedding attempt failed: {e1}")
            
            # Try approach 2: Explicitly use a simpler model
            try:
                return HuggingFaceEmbeddings(
                    model_name="distilbert-base-uncased"
                )
            except Exception as e2:
                print(f"Second embedding attempt failed: {e2}")
                
                # Try approach 3: Fall back to an older version of the API if available
                try:
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
                    
                    # Create a simple wrapper class that implements the required interface
                    class CustomEmbeddings:
                        def embed_documents(self, texts):
                            return model.encode(texts)
                        
                        def embed_query(self, text):
                            return model.encode(text)
                    
                    return CustomEmbeddings()
                except Exception as e3:
                    print(f"All embedding attempts failed. Last error: {e3}")
                    raise ValueError("Could not initialize any embedding model. Please check your environment setup.")
    
    embeddings = create_embeddings()
    vectordb = FAISS.from_documents(splits, embeddings)
    
    # LLM & RAG
    llm = ChatGroq(api_key=groq_api_key, model="llama-3-3-70b-versatile")
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    
    # Clean up temp files
    os.unlink(pdf_path)
    os.unlink(txt_path)
    
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
