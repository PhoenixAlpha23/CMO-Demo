import streamlit as st
from dotenv import load_dotenv
import os
from groq import Groq
from rag_chain2 import (
    build_rag_chain_with_model_choice, 
    process_scheme_query_with_retry, 
    get_optimized_query_suggestions,
    get_model_options,
    generate_audio_response,
    get_audio_cache_stats
)
import tempfile
import pandas as pd
import io
import time
import base64

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
    if "auto_play_tts" not in st.session_state:
        st.session_state.auto_play_tts = False

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

def create_audio_player_html(audio_data, auto_play=False):
    """Create custom HTML audio player with auto-play option"""
    audio_base64 = base64.b64encode(audio_data).decode()
    autoplay_attr = "autoplay" if auto_play else ""
    
    html = f"""
    <audio controls {autoplay_attr} style="width: 100%; margin: 10px 0;">
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    """
    return html

def safe_get_cache_stats():
    """Safely get cache stats with fallback values"""
    try:
        cache_stats = get_audio_cache_stats()
        # Ensure the expected keys exist
        if not isinstance(cache_stats, dict):
            return {"total": 0, "hit_rate": 0.0}
        
        # Set default values for missing keys
        return {
            "total": cache_stats.get("total", 0),
            "hit_rate": cache_stats.get("hit_rate", 0.0)
        }
    except Exception as e:
        # If the function fails entirely, return default values
        st.warning(f"Cache stats unavailable: {e}")
        return {"total": 0, "hit_rate": 0.0}

def main():
    st.set_page_config(page_title="CMRF RAG Assistant", layout="wide")
    st.markdown("<h1 style='text-align: center;'>🤖 CMRF RAG Assistant with TTS</h1>", unsafe_allow_html=True)
    
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
    
    # Model selection
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
    
    # TTS Settings
    st.sidebar.markdown("### 🔊 Text-to-Speech Settings")
    
    # Auto-play toggle
    st.session_state.auto_play_tts = st.sidebar.checkbox(
        "🎛️ Auto-play TTS", 
        value=st.session_state.auto_play_tts,
        help="Automatically play voice responses"
    )
    
    # TTS Speed control
    tts_speed = st.sidebar.slider(
        "🎵 Speech Speed", 
        min_value=0.5, 
        max_value=2.0, 
        value=1.0, 
        step=0.1,
        help="Adjust speech speed (1.0 = normal)"
    )
    
    # Voice language preference
    voice_lang_pref = st.sidebar.selectbox(
        "🌐 Voice Language Preference",
        options=["auto", "en", "hi", "mr"],
        format_func=lambda x: {
            "auto": "🧠 Auto-detect",
            "en": "🇺🇸 English", 
            "hi": "🇮🇳 Hindi",
            "mr": "🇮🇳 Marathi"
        }[x],
        help="Language for voice synthesis"
    )
    
    # Audio cache stats with error handling
    try:
        cache_stats = safe_get_cache_stats()
        if cache_stats["total"] > 0:
            st.sidebar.markdown("### 🧠 Audio Cache")
            st.sidebar.metric("Cached Responses", cache_stats["total"])
            st.sidebar.metric("Cache Hit Rate", f"{cache_stats['hit_rate']:.1%}")
            
            if st.sidebar.button("🗑️ Clear Audio Cache"):
                try:
                    from rag_chain2 import clear_audio_cache
                    clear_audio_cache()
                    st.sidebar.success("Audio cache cleared!")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Failed to clear cache: {e}")
    except Exception as e:
        st.sidebar.warning(f"Cache stats unavailable: {e}")
    
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
                    result = process_scheme_query_with_retry(
                        st.session_state.rag_chain, 
                        input_text
                    )
                    
                    # Handle both string and tuple responses
                    if isinstance(result, tuple):
                        assistant_reply = result[0] if result else "No response received"
                    else:
                        assistant_reply = result if result else "No response received"
                    
                    # Ensure we have a string
                    if not isinstance(assistant_reply, str):
                        assistant_reply = str(assistant_reply)
                
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
                is_cached = assistant_reply.startswith("[Cached]") if isinstance(assistant_reply, str) else False
                if is_cached:
                    st.info("🚀 This result was retrieved from cache (faster response)")
                    clean_reply = assistant_reply.replace("[Cached] ", "")
                else:
                    clean_reply = assistant_reply
                
                # Display text response
                st.markdown(
                    f"<div style='background-color:#E8F5E9; padding:15px; border-radius:8px; margin-bottom:15px; border-left: 4px solid #4CAF50;'>"
                    f"<b>🤖 Assistant:<br><br>{clean_reply}</div>", 
                    unsafe_allow_html=True
                )
                
                # Generate and display TTS audio
                with st.spinner("🔊 Generating voice response..."):
                    try:
                        # Check if generate_audio_response supports speed parameter
                        import inspect
                        audio_func_params = inspect.signature(generate_audio_response).parameters
                        
                        if 'speed' in audio_func_params:
                            result = generate_audio_response(
                                clean_reply, 
                                speed=tts_speed,
                                lang_preference=voice_lang_pref
                            )
                        else:
                            # Fallback without speed parameter
                            result = generate_audio_response(
                                clean_reply, 
                                lang_preference=voice_lang_pref
                            )
                        
                        # Handle different return formats
                        if isinstance(result, dict):
                            # If function returns a dictionary
                            audio_data = result.get('audio_data')
                            detected_lang = result.get('detected_lang', 'auto')
                            cache_hit = result.get('cache_hit', False)
                        elif isinstance(result, tuple) and len(result) >= 3:
                            # If function returns a tuple
                            audio_data, detected_lang, cache_hit = result
                        elif isinstance(result, tuple) and len(result) == 2:
                            # If function returns (audio_data, detected_lang)
                            audio_data, detected_lang = result
                            cache_hit = False
                        else:
                            # If function returns just audio_data
                            audio_data = result
                            detected_lang = 'auto'
                            cache_hit = False
                        
                        if audio_data:
                            # Show language detection info
                            lang_names = {"en": "English", "hi": "Hindi", "mr": "Marathi", "auto": "Mixed"}
                            lang_display = lang_names.get(detected_lang, detected_lang)
                            
                            cache_indicator = "🧠 (Cached)" if cache_hit else "🆕 (Generated)"
                            speed_info = f" | Speed: {tts_speed}x" if 'speed' in audio_func_params else ""
                            st.info(f"🔊 Voice: {lang_display}{speed_info} | {cache_indicator}")
                            
                            # Create and display audio player
                            audio_html = create_audio_player_html(
                                audio_data, 
                                auto_play=st.session_state.auto_play_tts
                            )
                            st.markdown(audio_html, unsafe_allow_html=True)
                        else:
                            st.warning("⚠️ Could not generate audio for this response")
                            
                    except Exception as audio_error:
                        st.warning(f"🔊 TTS Error: {audio_error}")
                        st.info("💡 Text response is still available above")
                
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
                
                # Add TTS playback for historical responses
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button(f"🔊 Play", key=f"tts_{i}", help="Generate voice for this response"):
                        # Safely handle the assistant response text
                        assistant_text = entry['assistant']
                        if isinstance(assistant_text, tuple):
                            assistant_text = assistant_text[0] if assistant_text else ""
                        elif not isinstance(assistant_text, str):
                            assistant_text = str(assistant_text)
                            
                        clean_text = assistant_text.replace("[Cached] ", "") if isinstance(assistant_text, str) else assistant_text
                        
                        try:
                            with st.spinner("Generating audio..."):
                                # Check if generate_audio_response supports speed parameter
                                import inspect
                                audio_func_params = inspect.signature(generate_audio_response).parameters
                                
                                if 'speed' in audio_func_params:
                                    result = generate_audio_response(
                                        clean_text,
                                        speed=tts_speed,
                                        lang_preference=voice_lang_pref
                                    )
                                else:
                                    # Fallback without speed parameter
                                    result = generate_audio_response(
                                        clean_text,
                                        lang_preference=voice_lang_pref
                                    )
                                
                                # Handle different return formats
                                if isinstance(result, dict):
                                    # If function returns a dictionary
                                    audio_data = result.get('audio_data')
                                elif isinstance(result, tuple) and len(result) >= 1:
                                    # If function returns a tuple
                                    audio_data = result[0]
                                else:
                                    # If function returns just audio_data
                                    audio_data = result
                                    
                            if audio_data:
                                audio_html = create_audio_player_html(audio_data, auto_play=True)
                                st.markdown(audio_html, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"TTS Error: {e}")
            
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
            **💡 Tips:**
            - Wait 2-3 seconds between queries
            - Use the 8B model for simple questions
            - 🔊 TTS responses are cached to save time and resources
            """)

    # Footer with tips
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.chat_history:
            st.markdown(f"📊 **Session:** {len(st.session_state.chat_history)} queries | Model: {selected_model}")
    with col2:
        st.markdown("💡 **Tip:** Use specific questions to avoid rate limits | 🔊 TTS available")

if __name__ == "__main__":
    main()
