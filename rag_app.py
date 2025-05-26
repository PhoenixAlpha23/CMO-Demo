import streamlit as st
from dotenv import load_dotenv
import os
from groq import Groq
from rag_chain import build_rag_chain_from_files, process_scheme_query, get_optimized_query_suggestions
import tempfile
import pandas as pd
import io
load_dotenv()

def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "suggested_query" not in st.session_state:
        st.session_state.suggested_query = ""

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
    st.set_page_config(page_title="CMRF RAG Assistant", layout="wide")
    st.markdown("<h1 style='text-align: center;'>🤖 CMRF RAG Assistant</h1>", unsafe_allow_html=True)
    
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        st.error("Missing GROQ_API_KEY. Please set it in your .env file.")
        st.stop()

    init_session_state()

    # Upload files
    st.markdown("<h4 style='text-align: center;'>📄 कृपया आरोग्य योजना आणि स्कीम तपशील फाईल्स खाली अपलोड करा</h4>", unsafe_allow_html=True)
    uploaded_pdf = st.file_uploader("Upload Scheme Details PDF", type=["pdf"])
    uploaded_txt = st.file_uploader("Upload Arogya Yojna booklet file", type=["txt"])

    if not (uploaded_pdf or uploaded_txt):
        st.warning("Please upload at least one file (PDF or TXT) to continue.")
        st.stop()

    # Load Whisper Client and Enhanced RAG
    whisper_client = Groq(api_key=GROQ_API_KEY)
    
    # Build enhanced RAG chain
    with st.spinner("🔧 Building enhanced RAG system for comprehensive scheme search..."):
        rag_chain = build_rag_chain_from_files(
            uploaded_pdf, 
            uploaded_txt, 
            GROQ_API_KEY, 
            enhanced_mode=True  # Enable enhanced mode for better scheme coverage
        )
    
    st.success("✅ RAG system ready! Enhanced for comprehensive scheme retrieval.")
    
    # Add suggested queries section
    with st.expander("💡 Quick Query Suggestions (Click to use)", expanded=False):
        st.markdown("**For comprehensive scheme information, try these optimized queries:**")
        suggestions = get_optimized_query_suggestions()
        
        col1, col2 = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            col = col1 if i % 2 == 0 else col2
            with col:
                if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                    st.session_state.suggested_query = suggestion
                    st.rerun()

    # Input Section
    st.markdown("### Ask a question by typing or using audio input")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Use suggested query if available
        default_value = st.session_state.suggested_query if st.session_state.suggested_query else ""
        user_input = st.text_input(
            "Enter your question", 
            key="text_input", 
            placeholder="e.g. सर्व योजना दाखवा / List all schemes / योजना माहिती द्या...",
            value=default_value
        )
        # Clear suggested query after use
        if st.session_state.suggested_query:
            st.session_state.suggested_query = ""
    
    with col2:
        audio_value = st.audio_input("🎤 Record your query")

    user_text = None
    if audio_value is not None:
        try:
            with st.spinner("🎧 Transcribing audio..."):
                user_text = transcribe_audio(whisper_client, audio_value.getvalue())
            st.success(f"🎧 Transcribed: {user_text}")
        except Exception as e:
            st.error(f"Transcription Error: {str(e)}")

    # Enhanced query processing
    if st.button("🔍 Get Answer", type="primary") or user_text:
        input_text = user_text if user_text else user_input.strip()
        if input_text:
            try:
                with st.spinner("🔍 Searching through all schemes..."):
                    # Use enhanced query processing
                    assistant_reply = process_scheme_query(rag_chain, input_text)
                
                # Add to chat history
                st.session_state.chat_history.insert(0, {
                    "user": input_text, 
                    "assistant": assistant_reply
                })
                
                # Show immediate result
                st.markdown("### 📋 Answer:")
                st.markdown(
                    f"""<div style='background-color:#E8F5E9; padding:15px; border-radius:8px; margin-bottom:15px; border-left: 4px solid #4CAF50;'>
                    <strong>🤖 Assistant:</strong><br><br>{assistant_reply}
                    </div>""", 
                    unsafe_allow_html=True
                )
                
            except Exception as e:
                st.error(f"Error generating response: {e}")
        else:
            st.warning("Please enter a question or record audio.")

    # Enhanced Chat History with better formatting
    with st.expander("📜 Chat History", expanded=len(st.session_state.chat_history) > 0):
        if st.session_state.chat_history:
            st.markdown(f"**Total conversations: {len(st.session_state.chat_history)}**")
            
            for i, entry in enumerate(st.session_state.chat_history):
                st.markdown(f"---")
                st.markdown(f"**Conversation #{len(st.session_state.chat_history) - i}**")
                
                st.markdown(
                    f"""<div style='background-color:#E3F2FD; padding:10px; border-radius:8px; margin-bottom:5px; border-left: 4px solid #2196F3;'>
                    <strong>🧑 Citizen:</strong><br>{entry['user']}
                    </div>""", 
                    unsafe_allow_html=True
                )
                
                st.markdown(
                    f"""<div style='background-color:#E8F5E9; padding:10px; border-radius:8px; margin-bottom:15px; border-left: 4px solid #4CAF50;'>
                    <strong>🤖 Assistant:</strong><br>{entry['assistant']}
                    </div>""", 
                    unsafe_allow_html=True
                )
            
            # Enhanced download options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Excel download
                df = pd.DataFrame(st.session_state.chat_history)
                df.columns = ['Query', 'Response']
                df.index = range(1, len(df) + 1)
                
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df.to_excel(writer, sheet_name='Chat History', index=True)
                
                st.download_button(
                    label="📥 Download Excel",
                    data=buffer.getvalue(),
                    file_name="cmrf_chat_history.xlsx",
                    mime="application/vnd.ms-excel",
                    use_container_width=True
                )
            
            with col2:
                # CSV download
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=True)
                
                st.download_button(
                    label="📄 Download CSV",
                    data=csv_buffer.getvalue(),
                    file_name="cmrf_chat_history.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col3:
                # Clear history button
                if st.button("🗑️ Clear History", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
                    
        else:
            st.info("No chat history yet. Ask your first question!")
            st.markdown("""
            **💡 Tips for better results:**
            - Use the suggested queries above for comprehensive scheme lists
            - Ask specific questions about eligibility, benefits, or application process
            - You can ask in English, Marathi, or mix both languages
            - Try queries like "सर्व योजना दाखवा" or "List all 72 schemes"
            """)

    # Add footer with usage statistics
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown(f"**📊 Session Stats:** {len(st.session_state.chat_history)} questions asked")

if __name__ == "__main__":
    main()
