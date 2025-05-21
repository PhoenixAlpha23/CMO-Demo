import streamlit as st
from dotenv import load_dotenv
import os
from groq import Groq
from rag_chain import build_rag_chain_from_files
import tempfile
import pandas as pd
load_dotenv()

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

def main():
    st.set_page_config(page_title="CMRF RAG ", layout="wide")
    st.markdown("<h1 style='text-align: center;'>🤖 CMRF RAG Assistant</h1>", unsafe_allow_html=True)
    

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        st.error("Missing GROQ_API_KEY. Please set it in your .env file.")
        st.stop()

    init_session_state()

    # Upload files
    st.markdown("<h4 style='text-align: center;'>📄 कृपया आरोग्य योजना आणि स्कीम तपशील फाईल्स खाली अपलोड करा</h4>",unsafe_allow_html=True)
    uploaded_pdf = st.file_uploader("Upload Scheme Details PDF", type=["pdf"])
    uploaded_txt = st.file_uploader("Upload Arogya Yojna booklet file", type=["txt"])

    if not (uploaded_pdf and uploaded_txt):
        st.warning("Please upload at least one file (PDF or TXT) to continue.")
        st.stop()

    # Load Whisper Client and RAG
    whisper_client = Groq(api_key=GROQ_API_KEY)
    rag_chain = build_rag_chain_from_files(uploaded_pdf, uploaded_txt, GROQ_API_KEY)

    # Input Section
    st.markdown("### Ask a question by typing or using audio input")
    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_input("Enter your question", key="text_input", placeholder="e.g. योजना माहिती द्या...")
    with col2:
        audio_value = st.audio_input("🎤 Record your query")

    user_text = None
    if audio_value is not None:
        try:
            user_text = transcribe_audio(whisper_client, audio_value.getvalue())
            st.success(f"🎧 Transcribed: {user_text}")
        except Exception as e:
            st.error(f"Transcription Error: {str(e)}")

    if st.button("🔍 Get Answer") or user_text:
        input_text = user_text if user_text else user_input.strip()
        if input_text:
            try:
                assistant_reply = rag_chain.invoke(input_text)["result"]
                st.session_state.chat_history.insert(0, {"user": input_text, "assistant": assistant_reply})
            except Exception as e:
                st.error(f"Error generating response: {e}")

    # Chat history
    with st.expander("📜 Chat History", expanded=True):
        if st.session_state.chat_history:
            for entry in st.session_state.chat_history:
                st.markdown(
                    f"""<div style='background-color:#E3F2FD; padding:10px; border-radius:8px; margin-bottom:5px;'>
                    <strong>🧑 Citizen:</strong> {entry['user']}
                    </div>""", unsafe_allow_html=True
                )
                st.markdown(
                    f"""<div style='background-color:#E8F5E9; padding:10px; border-radius:8px; margin-bottom:15px;'>
                    <strong>🤖 Assistant:</strong> {entry['assistant']}
                    </div>""", unsafe_allow_html=True
                )
        else:
            st.info("No chat history yet. Ask your first question!")
            
        # Add download button for chat history
        if st.session_state.chat_history:
            import pandas as pd
            import io
            
            # Create DataFrame from chat history
            df = pd.DataFrame(st.session_state.chat_history)
            df.columns = ['Query', 'Response']
            
            # Create download button
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Assistant History', index=True)
            
            st.download_button(
                label="📥 Download History",
                data=buffer.getvalue(),
                file_name="chat_history.xlsx",
                mime="application/vnd.ms-excel"
            )

if __name__ == "__main__":
    main()
