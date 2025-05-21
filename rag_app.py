import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from groq import Groq
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# ----------------- Initialization ----------------- #
@st.cache_resource
def load_rag_chain():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectordb = Chroma(persist_directory="chroma_db", embedding_function=embedding)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        max_tokens=1048
    )
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)

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

# ----------------- Main App ----------------- #
def main():
    st.set_page_config(page_title="RAG Assistant", layout="wide")
    st.title("🤖 RAG Assistant – English & Marathi Support")

    # Validate API Key
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        st.error("Missing GROQ_API_KEY. Please set it in your environment.")
        st.stop()

    # Load components
    whisper_client = Groq(api_key=GROQ_API_KEY)
    init_session_state()
    rag_chain = load_rag_chain()

    # Input Section
    st.markdown("### Ask a question by typing or using audio input")
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            user_input = st.text_input("Enter your question", key="text_input", placeholder="e.g. माहिती द्या...")

        with col2:
            audio_value = st.audio_input("🎤 Record your query")

    # Handle audio
    user_text = None
    if audio_value is not None:
        try:
            user_text = transcribe_audio(whisper_client, audio_value.getvalue())
            st.success(f"🎧 Transcribed: {user_text}")
        except Exception as e:
            st.error(f"Transcription Error: {str(e)}")

    # Handle text submission
    if st.button("🔍 Get Answer") or user_text:
        input_text = user_text if user_text else user_input.strip()

        if input_text:
            try:
                assistant_reply = rag_chain.invoke(input_text)["result"]
                st.session_state.chat_history.insert(0, {"user": input_text, "assistant": assistant_reply})
            except Exception as e:
                st.error(f"Error generating response: {e}")

    # Display Chat History
    with st.expander("📜 Chat History", expanded=True):
        if st.session_state.chat_history:
            for idx, entry in enumerate(st.session_state.chat_history):
                st.markdown(
                    f"""<div style='background-color:#E3F2FD; padding:10px; border-radius:8px; margin-bottom:5px;'>
                    <strong>🧑 You:</strong> {entry['user']}
                    </div>""", unsafe_allow_html=True
                )
                st.markdown(
                    f"""<div style='background-color:#E8F5E9; padding:10px; border-radius:8px; margin-bottom:15px;'>
                    <strong>🤖 Assistant:</strong> {entry['assistant']}
                    </div>""", unsafe_allow_html=True
                )
        else:
            st.info("No chat history yet. Ask your first question!")

if __name__ == "__main__":
    main()
