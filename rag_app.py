import streamlit as st
from dotenv import load_dotenv
import os
from groq import Groq
from rag_chain import (
    build_rag_chain_from_files, 
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
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False

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

    # Sidebar Settings
    st.sidebar.markdown("### ⚙️ Settings")
    
    # Enhanced mode toggle
    enhanced_mode = st.sidebar.checkbox(
        "Enhanced Mode", 
        value=True, 
        help="Better scheme coverage but uses more tokens"
    )
    
    # Debug mode toggle
    debug_mode = st.sidebar.checkbox(
        "Debug Mode", 
        value=False, 
        help="Show detailed processing information"
    )
    st.session_state.debug_mode = debug_mode
    
    # Rate limit info
    st.sidebar.markdown("### 📊 Rate Limit Info")
    queries_count = len(st.session_state.chat_history)
    st.sidebar.metric("Queries Made", queries_count)
    
    if queries_count > 10:
        st.sidebar.warning("⚠️ High query count. Consider shorter breaks between queries.")
    
    # Model information (for display only since we're using the fixed model from rag_chain)
    st.sidebar.markdown("### 🤖 Model Info")
    st.sidebar.info("Using: Llama 3.1 8B Instant (optimized for rate limits)")

    # Load Whisper Client and RAG (with caching)
    whisper_client = Groq(api_key=GROQ_API_KEY)
    
    # Build RAG chain only once or when settings change
    current_settings_key = f"{enhanced_mode}_{debug_mode}"
    if (st.session_state.rag_chain is None or 
        getattr(st.session_state, 'current_settings_key', '') != current_settings_key):
        
        with st.spinner("🔧 Building optimized RAG system..."):
            try:
                # Reset file pointers to beginning
                if uploaded_pdf:
                    uploaded_pdf.seek(0)
                if uploaded_txt:
                    uploaded_txt.seek(0)
                
                st.session_state.rag_chain = build_rag_chain_from_files(
                    uploaded_pdf, 
                    uploaded_txt, 
                    GROQ_API_KEY,
                    enhanced_mode=enhanced_mode,
                    debug=debug_mode
                )
                st.session_state.current_settings_key = current_settings_key
                st.success("✅ RAG system ready!")
                
                if debug_mode:
                    st.info("🔧 Debug mode enabled - detailed processing info will be shown")
                    
            except Exception as e:
                st.error(f"Failed to build RAG system: {e}")
                if debug_mode:
                    st.exception(e)
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
            if debug_mode:
                st.exception(e)

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
                    
                    # Use the optimized query processor with debug option
                    assistant_reply = process_scheme_query_with_retry(
                        st.session_state.rag_chain, 
                        input_text,
                        max_retries=3,
                        debug=debug_mode
                    )
                
                # Add to chat history
                st.session_state.chat_history.insert(0, {
                    "user": input_text, 
                    "assistant": assistant_reply,
                    "model": "llama-3.1-8b-instant",
                    "timestamp": time.strftime("%H:%M:%S"),
                    "enhanced_mode": enhanced_mode
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
                    f"<b>🤖 Assistant:</b><br><br>{assistant_reply}</div>", 
                    unsafe_allow_html=True
                )
                
                if debug_mode:
                    st.markdown("### 🔧 Debug Information:")
                    st.json({
                        "query_length": len(input_text),
                        "response_length": len(assistant_reply),
                        "was_cached": is_cached,
                        "enhanced_mode": enhanced_mode,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                
            except Exception as e:
                st.error(f"Error: {e}")
                if debug_mode:
                    st.exception(e)
                if "rate_limit" in str(e).lower():
                    st.info("💡 **Tips to avoid rate limits:**\n- Wait 10-15 seconds between queries\n- Use simpler, more specific questions\n- Try asking about specific schemes instead of 'all schemes'")
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
                enhanced = entry.get('enhanced_mode', False)
                mode_text = "Enhanced" if enhanced else "Standard"
                
                st.caption(f"#{len(st.session_state.chat_history) - i} | Model: {model_used} | Mode: {mode_text} | Time: {timestamp}")
                
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
                df = df[['user', 'assistant', 'model', 'enhanced_mode', 'timestamp']]
                df.columns = ['Query', 'Response', 'Model Used', 'Enhanced Mode', 'Time']
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
                    # Clear the internal cache from rag_chain module
                    from rag_chain import _query_cache
                    _query_cache.clear()
                    st.success("RAG system and cache cleared!")
                    st.rerun()
                    
        else:
            st.info("No chat history yet. Ask your first question!")
            st.markdown("""
            **💡 Rate-limit friendly tips:**
            - Start with specific questions rather than "list all schemes"
            - Wait 2-3 seconds between queries
            - Cached results (marked with 🚀) don't count against rate limits
            - Use Enhanced Mode for better scheme coverage
            """)

    # Footer with tips and system info
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.chat_history:
            mode_text = "Enhanced" if enhanced_mode else "Standard"
            st.markdown(f"📊 **Session:** {len(st.session_state.chat_history)} queries | Mode: {mode_text}")
    with col2:
        st.markdown("💡 **Tip:** Use specific questions to avoid rate limits")
    
    # System status
    if debug_mode:
        st.markdown("### 🔧 System Status")
        cache_size = len(getattr(__import__('rag_chain'), '_query_cache', {}))
        st.metric("Cache Size", cache_size)
        
        if st.button("Clear Cache Only"):
            from rag_chain import _query_cache
            _query_cache.clear()
            st.success("Cache cleared!")
            st.rerun()

if __name__ == "__main__":
    main()
