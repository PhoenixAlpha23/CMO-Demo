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
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=2000
        )

        prompt = PromptTemplate(
            template="""You are a Knowledge Assistant designed for answering questions specifically from the knowledge base provided to you.
Your task is as follows: give a detailed response for user query in the user language (eg. what are some schemes? --> Here is a list of some schemes)
Ensure your response follows these styles and tone in your response:
* Use direct, everyday language
* Personal and human
* Favour detailed responses, with mentions of websites and headings such as description, eligibility or उद्देशः, अंतर्भूत घटकः
* In case no relevant information is found, default your response to a phrase like "For more details contact on 104/102 helpline numbers."

Your goal is to achieve the following: help a citizen understand about the schemes and its eligibility criteria.
Here is the content you will work with: {context}
Question: {question}
Now perform the task as instructed above.
Answer:""",
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
