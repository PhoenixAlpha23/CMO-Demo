import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from groq import Groq
import sys
import pysqlite3
from chromadb.config import Settings

sys.modules["sqlite3"] = pysqlite3
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate


# Load environment variables
load_dotenv()

# ---------------------- Initialization ---------------------- #
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a Helpline assistant. Use the following context to answer the user's question.

Context:
{context}

Question:
{question}

Answer in the user's language as appropriate:
Eg: Marathi, Hindi, or English.
"""
)

@st.cache_resource
def load_rag_chain():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectordb = Chroma(persist_directory="chroma_db", embedding_function=embedding)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.1-8b-instant")
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": custom_prompt}
    )

def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def transcribe_audio(client, audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name
    try:
        with open(temp_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=file,
                model="whisper-large-v3",
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
                temperature=0.0
            )
        return transcription.text
    finally:
        os.unlink(temp_path)

# ---------------------- Streamlit UI ---------------------- #
def main():
    st.set_page_config(page_title="💬 RAG Assistant", layout="wide")
    st.markdown("<h1 style='text-align: center;'>🤖 RAG Assistant – English & Marathi Support</h1>", unsafe_allow_html=True)

    if not os.getenv("GROQ_API_KEY"):
        st.error("Missing GROQ_API_KEY. Please set it in your environment.")
        st.stop()

    # Load components
    whisper_client = Groq(api_key=os.getenv("GROQ_API_KEY_2"))
    rag_chain = load_rag_chain()
    init_session_state()

    # Chat Area
    st.markdown("""
        <style>
            .chat-container {
                max-height: 500px;
                overflow-y: auto;
                padding: 15px;
                background-color: #f9f9f9;
                border-radius: 10px;
                border: 1px solid #ddd;
                margin-bottom: 1rem;
            }
            .user-bubble, .bot-bubble {
                padding: 10px 15px;
                border-radius: 10px;
                margin-bottom: 10px;
                max-width: 80%;
                font-size: 16px;
            }
            .user-bubble {
                background-color: #d1ecf1;
                text-align: left;
                align-self: flex-start;
            }
            .bot-bubble {
                background-color: #d4edda;
                text-align: left;
                align-self: flex-end;
            }
        </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

        if st.session_state.chat_history:
            for entry in st.session_state.chat_history:
                st.markdown(f"<div class='user-bubble'><b>🧑 You:</b><br>{entry['user']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='bot-bubble'><b>🤖 Assistant:</b><br>{entry['assistant']}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<p>No messages yet. Ask something!</p>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Chat Input
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input("Ask your question", placeholder="e.g. माहिती द्या...")
        with col2:
            submit_btn = st.form_submit_button("Send")

    # Handle Audio Input
    audio_value = st.audio_input("🎤 Or record your query")
    user_text = None
    if audio_value is not None:
        try:
            user_text = transcribe_audio(whisper_client, audio_value.getvalue())
            st.success(f"🎧 Transcribed: {user_text}")
        except Exception as e:
            st.error(f"Transcription Error: {str(e)}")

    # Process Input
    if submit_btn or user_text:
        input_text = user_text if user_text else user_input.strip()
        if input_text:
            try:
                assistant_reply = rag_chain.invoke(input_text)["result"]
                st.session_state.chat_history.append({"user": input_text, "assistant": assistant_reply})
            except Exception as e:
                st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
