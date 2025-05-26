import streamlit as st
from dotenv import load_dotenv
import os
from groq import Groq
from rag_chain import (
    build_rag_chain_with_model_choice, 
    process_scheme_query_with_retry, 
    get_optimized_query_suggestions,
    get_model_options
)
import tempfile
import pandas as pd
import io
import time
load_dotenv()

def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "suggested_query" not in st.session_state:
        st.session_state.suggested_query = ""
    if "last_query_time" not in st.session_state:
        st.session_state.last_query_time = 0
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

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

def check_rate_limit_delay():
    """Check if we need to wait before making another query"""
    current_time = time.time()
    time_since_last = current_time - st.session_state.last_query_time
    min_delay = 2  # Minimum 2 seconds between queries
    
    if time_since_last < min_delay:
        wait_time = min_delay - time_since_last
        return wait_time
    return 0

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

    # Model selection for rate limit optimization
    st.sidebar.markdown("### ⚙️ Settings")
    model_options = get_model_options()
    
    selected_model = st.sidebar.selectbox(
        "Choose Model (for rate limit management):",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x]["name"],
        index=0  # Default to fastest model
    )
    
    st.sidebar.markdown(f"**Selected:** {model_options[selected_model]['description']}")
    
    # Enhanced mode toggle
    enhanced_mode = st.sidebar.checkbox(
        "Enhanced Mode", 
        value=True, 
        help="Better scheme coverage but uses more tokens"
    )
    
    # Rate limit info
    st.sidebar.markdown("### 📊 Rate Limit Info")
    queries_count = len(st.session_state.chat_history)
    st.sidebar.metric("Queries Made", queries_count)
    
    if queries_count > 10:
        st.sidebar.warning("⚠️ High query count. Consider shorter breaks between queries.")

    # Load Whisper Client and RAG (with caching)
    whisper_client = Groq(api_key=GROQ_API_KEY)
    
    # Build RAG chain only once or when model changes
    current_model_key = f"{selected_model}_{enhanced_mode}"
    if (st.session_state.rag_chain is None or 
        getattr(st.session_state, 'current_model_key', '') != current_model_key):
        
        with st.spinner("🔧 Building optimized RAG system..."):
            try:
                st.session_state.rag_chain = build_rag_chain_with_model_choice(
                    uploaded_pdf, 
                    uploaded_txt, 
                    GROQ_API_KEY,
                    model_choice=selected_model,
                    enhanced_mode=enhanced_mode
                )
                st.session_state.current_model_key = current_model_key
                st.success("✅ RAG system ready!")
            except Exception as e:
                st.error(f"Failed to build RAG system: {e}")
                st.stop()
    
    # Rate-limit friendly suggestions
    with st.expander("💡 Optimized Query Suggestions", expanded=False):
        st.markdown("**Rate-limit friendly queries:**")
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
        default_value = st.session_state.suggested_query if st.session_state.suggested_query else ""
        user_input = st.text_input(
            "Enter your question", 
            key="text_input", 
            placeholder="e.g. मुख्य योजना दाखवा / Show main schemes...",
            value=default_value
        )
        if st.session_state.suggested_query:
            st.session_state.suggested_query = ""
    
    with col2:
        audio_value = st.audio_input("🎤 Record your query")

    # Audio transcription
    user_text = None
    if audio_value is not None:
        try:
            with st.spinner("🎧 Transcribing audio..."):
                user_text = transcribe_audio(whisper_client, audio_value.getvalue())
            st.success(f"🎧 Transcribed: {user_text}")
        except Exception as e:
            st.error(f"Transcription Error: {str(e)}")

    # Query processing with rate limit handling
    if st.button("🔍 Get Answer", type="primary") or user_text:
        input_text = user_text if user_text else user_input.strip()
        if input_text:
            # Check rate limit
            wait_time = check_rate_limit_delay()
            if wait_time > 0:
                st.warning(f"⏳ Please wait {wait_time:.1f} seconds to avoid rate limits...")
                time.sleep(wait_time)
            
            try:
                with st.spinner("🔍 Processing query..."):
                    st.session_state.last_query_time = time.time()
                    
                    # Use the optimized query processor
                    assistant_reply = process_scheme_query_with_retry(
                        st.session_state.rag_chain, 
                        input_text
                    )
                
                # Add to chat history
                st.session_state.chat_history.insert(0, {
                    "user": input_text, 
                    "assistant": assistant_reply,
                    "model": selected_model,
                    "timestamp": time.strftime("%H:%M:%S")
                })
                
                # Show result
                st.markdown("### 📋 Answer:")
                
                # Show if result was cached
                is_cached = assistant_reply.startswith("[Cached]")
                if is_cached:
                    st.info("🚀 This result was retrieved from cache (faster response)")
                    assistant_reply = assistant_reply.replace("[Cached] ", "")
                
                st.markdown(
                    f"<div style='background-color:#E8F5E9; padding:15px; border-radius:8px; margin-bottom:15px; border-left: 4px solid #4CAF50;'>"
                    f"<b>🤖 Assistant:<br><br>{assistant_reply}</div>", 
                    unsafe_allow_html=True
                )
                
            except Exception as e:
                st.error(f"Error: {e}")
                if "rate_limit" in str(e).lower():
                    st.info("💡 **Tips to avoid rate limits:**\n- Wait 10-15 seconds between queries\n- Use simpler, more specific questions\n- Try the faster model (8B) for basic queries")
        else:
            st.warning("Please enter a question or record audio.")

    # Enhanced Chat History
    with st.expander("📜 Chat History", expanded=len(st.session_state.chat_history) > 0):
        if st.session_state.chat_history:
            st.markdown(f"**Total conversations: {len(st.session_state.chat_history)}**")
            
            for i, entry in enumerate(st.session_state.chat_history):
                st.markdown(f"---")
                
                # Show metadata
                model_used = entry.get('model', 'Unknown')
                timestamp = entry.get('timestamp', 'Unknown time')
                st.caption(f"#{len(st.session_state.chat_history) - i} | Model: {model_used} | Time: {timestamp}")
                
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
            
            # Download and management options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Excel download with metadata
                df = pd.DataFrame(st.session_state.chat_history)
                df = df[['user', 'assistant', 'model', 'timestamp']]
                df.columns = ['Query', 'Response', 'Model Used', 'Time']
                df.index = range(1, len(df) + 1)
                
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df.to_excel(writer, sheet_name='Chat History', index=True)
                
                st.download_button(
                    label="📥 Download Excel",
                    data=buffer.getvalue(),
                    file_name=f"cmrf_chat_history_{time.strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.ms-excel",
                    use_container_width=True
                )
            
            with col2:
                if st.button("🗑️ Clear History", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
            
            with col3:
                if st.button("🔄 Reset RAG", use_container_width=True, help="Reset RAG system and cache"):
                    st.session_state.rag_chain = None
                    st.rerun()
                    
        else:
            st.info("No chat history yet. Ask your first question!")
            st.markdown("""
            **💡 Rate-limit friendly tips:**
            - Start with specific questions rather than "list all schemes"
            - Wait 2-3 seconds between queries
            - Use the 8B model for simple questions
            - Cached results (marked with 🚀) don't count against rate limits
            """)

    # Footer with tips
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.chat_history:
            st.markdown(f"📊 **Session:** {len(st.session_state.chat_history)} queries | Model: {selected_model}")
    with col2:
        st.markdown("💡 **Tip:** Use specific questions to avoid rate limits")

if __name__ == "__main__":
    main()
