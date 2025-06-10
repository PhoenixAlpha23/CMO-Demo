import streamlit as st
import pandas as pd # Required for chat history download
import io # Required for chat history download
import time # Required for chat history download/timestamp

def inject_chat_styles():
    """Injects CSS styles for the modern chat layout while maintaining original functionality."""
    st.markdown("""
    <style>
    /* Updated styles to match reference UI while keeping original classes */
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 15px;
        border: 1px solid #E5E5EA;
        border-radius: 10px;
        margin-bottom: 20px;
        background-color: #FAFAFA;
    }
    .chat-bubble-user {
        background-color: #007AFF;
        color: white;
        padding: 10px 15px;
        border-radius: 18px 18px 0 18px;
        margin: 8px 0;
        text-align: right;
        margin-left: 30%;
        max-width: 70%;
        word-wrap: break-word;
    }
    .chat-bubble-assistant {
        background-color: #E5E5EA;
        color: black;
        padding: 10px 15px;
        border-radius: 18px 18px 18px 0;
        margin: 8px 0;
        text-align: left;
        margin-right: 30%;
        max-width: 70%;
        word-wrap: break-word;
    }
    /* New styles for the reference UI elements */
    .suggestion-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 20px;
    }
    .suggestion-button {
        background-color: #F0F2F6;
        border: 1px solid #D3D3D3;
        border-radius: 20px;
        padding: 8px 16px;
        cursor: pointer;
        transition: all 0.2s;
    }
    .file-info {
        background-color: #F8F9FA;
        border: 1px solid #E5E5EA;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .analysis-steps {
        background-color: #F8F9FA;
        border: 1px solid #E5E5EA;
        border-radius: 8px;
        padding: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

def render_file_uploaders(st_obj):
    """Renders file uploaders for PDF and TXT files - UNCHANGED"""
    st_obj.markdown("<h4 style='text-align: center;'>📄 कृपया योजनेचे तपशील फाइल अपलोड करा</h4>", unsafe_allow_html=True)
    uploaded_pdf = st_obj.file_uploader("स्किम तपशील पीडीएफ अपलोड करा", type=["pdf"])
    uploaded_txt = st_obj.file_uploader("आरोग्य योजना बुकलेट अपलोड करा", type=["txt"])
    return uploaded_pdf, uploaded_txt  

def render_query_input(st_obj, whisper_client, transcribe_audio_func):
    """Renders text input at bottom when chat exists, otherwise at top"""
    # Only show input at top if no chat history exists
    if not st_obj.session_state.get('chat_history', []):
        st_obj.markdown("### Ask a question by typing or using audio input")
    
    # This container will hold our input and float to bottom
    input_container = st_obj.container()
    
    with input_container:
        col1, col2 = st_obj.columns([3, 1])
        
        with col1:
            default_value = st_obj.session_state.suggested_query if st_obj.session_state.suggested_query else ""
            user_input = st_obj.text_input(
                "Enter your question", 
                key="text_input", 
                placeholder="e.g. मुख्य योजना दाखवा / Show main schemes...",
                label_visibility="collapsed"
            )
            if st_obj.session_state.suggested_query:
                st_obj.session_state.suggested_query = ""
        
        with col2:
            audio_value = st.audio_input("🎤 Record your query", key="audio_input")

    # CSS to make input stick to bottom when chat exists
    if st_obj.session_state.get('chat_history', []):
        st_obj.markdown("""
        <style>
        [data-testid="stVerticalBlock"]:has(> div:last-child > .stContainer) {
            position: fixed;
            bottom: 50px;
            width: calc(100% - 2rem);
            background: white;
            padding: 1rem;
            z-index: 100;
            border-top: 1px solid #eee;
        }
        </style>
        """, unsafe_allow_html=True)

    user_text = None
    if audio_value is not None:
        try:
            with st_obj.spinner("🎧 Transcribing audio..."):
                succes, transcription = user_text = transcribe_audio_func(whisper_client, audio_value.getvalue())
                if not succes:
                    st_obj.error(transcription)
                    user_text=""
                else:
                    st_obj.success(f"🎧 Transcribed: {transcription}")
                    user_text= transcription
        except Exception as e:
            st_obj.error(f"Transcription Error: {str(e)}")
    
    if user_input:
        st_obj.session_state.last_user_input = user_input
    elif user_text:
        st_obj.session_state.last_user_input = user_text
        
    return user_input, user_text  # Same return as before
def render_answer_section(
    st_obj, 
    assistant_reply, 
    generate_audio_func, 
    create_audio_player_html_func,
    voice_lang_pref,
    lang_code_to_name_map,
    allowed_tts_langs_set,
    tts_available_flag
):
    """Displays the assistant's answer and TTS audio player in chatbot style."""
    
    is_cached = assistant_reply.startswith("[Cached]") if isinstance(assistant_reply, str) else False
    clean_reply = assistant_reply.replace("[Cached] ", "") if is_cached else assistant_reply

    # Chat UI Style
    st_obj.markdown('<div class="chat-container">', unsafe_allow_html=True)
    if 'last_user_input' in st_obj.session_state and st_obj.session_state.last_user_input:
        st_obj.markdown(
            f'<div class="chat-bubble-user">🤓{st_obj.session_state.last_user_input}</div>',
            unsafe_allow_html=True
        )
    st_obj.markdown(
        f'<div class="chat-bubble-assistant">🤖{clean_reply}</div>',
        unsafe_allow_html=True
    )
    st_obj.markdown('</div>', unsafe_allow_html=True)

    if is_cached:
        st_obj.info("🚀 This result was retrieved from cache (faster response)")

    # TTS Section
    if tts_available_flag:
        with st_obj.spinner("🔊 Generating voice response..."):
            try:
                audio_data, lang_used_for_tts, cache_hit = generate_audio_func(
                    text=clean_reply,
                    lang_preference=voice_lang_pref
                )

                if voice_lang_pref != 'auto' and voice_lang_pref != lang_used_for_tts:
                    st_obj.info(f"ℹ️ TTS was preferred in {lang_code_to_name_map.get(voice_lang_pref, voice_lang_pref)}, "
                                f"but generated in {lang_code_to_name_map.get(lang_used_for_tts, lang_used_for_tts)}.")
                elif lang_used_for_tts not in allowed_tts_langs_set and voice_lang_pref == 'auto':
                     st_obj.info(f"ℹ️ Content detected as '{lang_code_to_name_map.get(lang_used_for_tts, lang_used_for_tts)}'. "
                                 f"TTS may be generated in a default supported language.")

                if audio_data:
                    lang_display = lang_code_to_name_map.get(lang_used_for_tts, str(lang_used_for_tts).capitalize())
                    cache_indicator = "🧠 (Cached)" if cache_hit else "🆕 (Generated)"
                    st_obj.info(f"🔊 Voice: {lang_display} | {cache_indicator}")
                    
                    audio_html = create_audio_player_html_func(
                        audio_data,
                        auto_play=st_obj.session_state.auto_play_tts
                    )
                    st_obj.markdown(audio_html, unsafe_allow_html=True)
                elif not clean_reply.strip():
                    st_obj.info("ℹ️ No text to speak.")
                else:
                    st_obj.warning("⚠️ Could not generate audio for this response.")
            except Exception as audio_error:
                st_obj.warning(f"🔊 TTS Error: {audio_error}")
    elif clean_reply.strip(): 
        st_obj.info("ℹ️ TTS is not available. Text response is shown above.")


def render_chat_history(
    st_obj,
    pd_module,
    io_module,
    time_module,
    generate_audio_func,
    create_audio_player_html_func,
    voice_lang_pref,
    tts_available_flag,
    lang_code_to_name_map,
    allowed_tts_langs_set
):
    """Displays the chat history with options to download, clear, and reset RAG."""
    with st_obj.expander("📜 Chat History", expanded=len(st_obj.session_state.chat_history) > 0):
        if st_obj.session_state.chat_history:
            st_obj.markdown(f"**Total conversations: {len(st_obj.session_state.chat_history)}**")
            for i, entry in enumerate(st_obj.session_state.chat_history):
                st_obj.markdown(f"---")
                model_used = entry.get('model', 'Unknown')
                timestamp = entry.get('timestamp', 'Unknown time')
                st_obj.caption(f"#{len(st_obj.session_state.chat_history) - i} | Model: {model_used} | Time: {timestamp}")
                st_obj.markdown(f"<div style='background-color:#E3F2FD; padding:10px; border-radius:8px; margin-bottom:5px; border-left: 4px solid #2196F3;'><strong>🧑 Citizen:</strong><br>{entry['user']}</div>", unsafe_allow_html=True)
                st_obj.markdown(f"<div style='background-color:#E8F5E9; padding:10px; border-radius:8px; margin-bottom:15px; border-left: 4px solid #4CAF50;'><strong>🤖 Assistant:</strong><br>{entry['assistant']}</div>", unsafe_allow_html=True)

                if tts_available_flag:
                    if st_obj.button(f"🔊 Play History #{len(st_obj.session_state.chat_history) - i}", key=f"tts_hist_{i}"):
                        assistant_text = entry['assistant']
                        if isinstance(assistant_text, tuple):
                            assistant_text = assistant_text[0] if assistant_text else ""
                        elif not isinstance(assistant_text, str):
                            assistant_text = str(assistant_text)
                            
                        clean_text_hist = assistant_text.replace("[Cached] ", "") if isinstance(assistant_text, str) else assistant_text
                        
                        if clean_text_hist.strip():
                            with st_obj.spinner(f"🔊 Generating voice for history item #{len(st_obj.session_state.chat_history) - i}..."):
                                try:
                                    audio_data_hist, lang_used_hist, cache_hit_hist = generate_audio_func(
                                        text=clean_text_hist,
                                        lang_preference=voice_lang_pref
                                    )

                                    if audio_data_hist:
                                        audio_html_hist = create_audio_player_html_func(
                                            audio_data_hist,
                                            auto_play=True 
                                        )
                                        st_obj.markdown(audio_html_hist, unsafe_allow_html=True) 
                                    else:
                                        st_obj.warning("⚠️ Could not generate audio for this historical response.")
                                except Exception as audio_error_hist:
                                    st_obj.error(f"🔊 TTS Error for history: {audio_error_hist}")
                        else:
                            st_obj.info("ℹ️ No text to speak for this historical item.")

            # Download, Clear, Reset buttons
            col_hist1, col_hist2, col_hist3 = st_obj.columns(3)
            with col_hist1:
                df_hist = pd_module.DataFrame(st_obj.session_state.chat_history)
                if not df_hist.empty:
                    df_hist = df_hist[['user', 'assistant', 'model', 'timestamp']]
                    df_hist.columns = ['Query', 'Response', 'Model Used', 'Time']
                    df_hist.index = range(1, len(df_hist) + 1)
                
                buffer_hist = io_module.BytesIO()
                with pd_module.ExcelWriter(buffer_hist, engine='xlsxwriter') as writer_hist:
                    df_hist.to_excel(writer_hist, sheet_name='Chat History', index=not df_hist.empty)
                st_obj.download_button(label="📥 Download History", data=buffer_hist.getvalue(), file_name=f"cmrf_chat_history_{time_module.strftime('%Y%m%d_%H%M')}.xlsx", mime="application/vnd.ms-excel", use_container_width=True, key="download_hist_btn")
            with col_hist2:
                if st_obj.button("🗑️ Clear History", use_container_width=True, key="clear_hist_btn"):
                    st_obj.session_state.chat_history = []
                    st_obj.rerun()
            with col_hist3:
                if st_obj.button("🔄 Reset RAG Chain", use_container_width=True, key="reset_rag_btn", help="Rebuilds the RAG system with current documents and settings."):
                    st_obj.session_state.rag_chain = None 
                    st_obj.session_state.current_model_key = None 
                    st_obj.rerun()
        else:
            st_obj.info("No chat history yet.")
            
def render_footer(st_obj, selected_model):
    """Renders the footer section with session info and tips."""
    st_obj.markdown("---")
    col1, col2 = st_obj.columns(2)
    with col1:
        if st_obj.session_state.chat_history:
            st_obj.markdown(f"📊 **Session:** {len(st_obj.session_state.chat_history)} queries | Model: {selected_model}")
    with col2:
        st_obj.markdown("💡 **Tip:** Use specific questions to avoid rate limits | 🔊 TTS available")
