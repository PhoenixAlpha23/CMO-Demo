import streamlit as st
from groq import Groq
import pandas as pd
import io
import time
import functools # Added for partial function application

# Core service imports
from core.rag_services import build_rag_chain_with_model_choice, process_scheme_query_with_retry, get_model_options
from core.tts_services import generate_audio_response, TTS_AVAILABLE 
from core.cache_manager import (
    get_audio_cache_stats, 
    clear_audio_cache,
    _audio_cache, 
    get_audio_hash,
    cache_audio, 
    get_cached_audio 
)
from core.transcription import transcribe_audio

# UI module imports
from ui.main_panel import render_file_uploaders, render_query_input, render_answer_section, render_chat_history, render_footer
from ui.components import create_audio_player_html

# Utility imports
from utils.config import load_env_vars, GROQ_API_KEY
from utils.helpers import init_session_state, check_rate_limit_delay, safe_get_cache_stats, LANG_CODE_TO_NAME, ALLOWED_TTS_LANGS

load_env_vars()

def main():
    st.set_page_config(page_title="RAGhu", layout="wide")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col2:
        st.image("assets/cmrf logo.jpg", width=250)
    st.markdown("<h1 style='text-align: center;'>🤖CMRF AI AGENT </h1>", unsafe_allow_html=True)
    
    if not GROQ_API_KEY:
        st.error("Missing GROQ_API_KEY. Please set it in your .env file.")
        st.stop()

    init_session_state()

    # Upload files (now returns a list)
    uploaded_pdf, uploaded_txt = render_file_uploaders(st)

    # Separate PDF and TXT files for downstream logic
    uploaded_pdf = next((f for f in uploaded_files if f.name.lower().endswith(".pdf")), None)
    uploaded_txt = next((f for f in uploaded_files if f.name.lower().endswith(".txt")), None)

    if not (uploaded_pdf or uploaded_txt):
        st.warning("कृपया पुढे जानेसाठी किमान एक फाइल (PDF किंवा TXT) अपलोड करा.")
        st.stop()

#hardcode Sidebar values
    selected_model = "llama-3.3-70b-versatile"  # Use the model key for Llama 3.3 Versatile
    enhanced_mode = True
    voice_lang_pref = "auto"
    st.session_state.auto_play_tts = True

    # Load Whisper Client
    whisper_client = Groq(api_key=GROQ_API_KEY)
    
    # Build RAG chain only once or when model/files change
    # More robust key: includes file names to rebuild if files change.
    pdf_name = uploaded_pdf.name if uploaded_pdf else "None"
    txt_name = uploaded_txt.name if uploaded_txt else "None"
    current_model_key = f"{selected_model}_{enhanced_mode}_{pdf_name}_{txt_name}"

    if (st.session_state.rag_chain is None or 
        st.session_state.get('current_model_key', '') != current_model_key):
        
        with st.spinner("🔧 Building optimized RAG system... Please wait."):
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

    # Input Section
    user_input, user_text = render_query_input(st, whisper_client, transcribe_audio)

    # Create a partially configured TTS function to pass to UI elements
    # This keeps UI elements unaware of cache internals.
    if TTS_AVAILABLE:
        partial_generate_audio = functools.partial(
            generate_audio_response,
            audio_cache=_audio_cache,
            get_audio_hash_func=get_audio_hash,
            cache_audio_func=cache_audio,
            get_cached_audio_func=get_cached_audio
        )
    else:
        # Provide a dummy function if TTS is not available
        # It should match the expected return signature (audio_data, lang_used, cache_hit)
        def dummy_tts(*args, **kwargs):
            return None, (kwargs.get('lang_preference') if kwargs.get('lang_preference') != 'auto' else 'en'), False
        partial_generate_audio = dummy_tts


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
                    
                    result = process_scheme_query_with_retry(
                        st.session_state.rag_chain, 
                        input_text
                    )
                    
                    if isinstance(result, tuple):
                        assistant_reply = result[0] if result else "No response received"
                    else:
                        assistant_reply = result if result else "No response received"
                    
                    if not isinstance(assistant_reply, str):
                        assistant_reply = str(assistant_reply)
                
                st.session_state.chat_history.insert(0, {
                    "user": input_text, 
                    "assistant": assistant_reply,
                    "model": selected_model,
                    "timestamp": time.strftime("%H:%M:%S")
                })
                
                render_answer_section(
                    st, 
                    assistant_reply, 
                    partial_generate_audio, # Pass the pre-configured TTS function
                    create_audio_player_html, 
                    voice_lang_pref,
                    LANG_CODE_TO_NAME,
                    ALLOWED_TTS_LANGS,
                    TTS_AVAILABLE # Pass TTS availability status
                )
            except Exception as e:
                st.error(f"Error: {e}")
                if "rate_limit" in str(e).lower():
                    st.info("💡 **Tips to avoid rate limits:**\n- Wait 10-15 seconds between queries\n- Use simpler, more specific questions\n- Try the faster model (8B) for basic queries")
        else:
            st.warning("Please enter a question or record audio.")

    # Enhanced Chat History
    render_chat_history(
        st, 
        pd, 
        io, 
        time, 
        partial_generate_audio, # Pass the pre-configured TTS function
        create_audio_player_html, 
        voice_lang_pref,
        TTS_AVAILABLE, # Pass TTS availability status
        LANG_CODE_TO_NAME,
        ALLOWED_TTS_LANGS
    )

    # Footer with tips
    render_footer(st, selected_model)

if __name__ == "__main__":
    main()
