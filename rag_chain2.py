from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.retrievers import TFIDFRetriever
from langchain.prompts import PromptTemplate
import tempfile, os


def build_rag_chain_from_files(pdf_file, txt_file, groq_api_key):
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

        docs = []
        if pdf_path:
            docs += PyPDFLoader(pdf_path).load()
        if txt_path:
            docs += TextLoader(txt_path, encoding="utf-8").load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        splits = splitter.split_documents(docs)
        retriever = TFIDFRetriever.from_documents(splits, k=min(15, len(splits)))

        llm = ChatGroq(
            api_key=groq_api_key,
            model="deepseek-r1-distill-llama-70b",
            temperature=0.0,
            max_tokens=2000
        )

        prompt = PromptTemplate(
            template=prompt = PromptTemplate(
    template="""
You are a government scheme assistant. You must answer based ONLY on the context provided below.
Always structure your output like this if relevant information is available:
Schemes:
[scheme names and summaries]

Websites (if any):
[list any URLs related to the scheme]

Helpline (if any):
[mention phone numbers if present]

If no relevant information is found, say:
"I don't have relevant information for this. You can contact 102/104 helpline numbers for details."

Use language that matches the user's question. Be helpful, concise, and complete.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)


        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt}
        )

    finally:
        if pdf_path and os.path.exists(pdf_path):
            os.unlink(pdf_path)
        if txt_path and os.path.exists(txt_path):
            os.unlink(txt_path)


def process_scheme_query_with_retry(rag_chain, user_query, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = rag_chain.invoke({"query": user_query})
            return result.get("result", "No results found.")
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            return f"Error processing query: {str(e)}"

# Alias for continuity
process_scheme_query = process_scheme_query_with_retry
