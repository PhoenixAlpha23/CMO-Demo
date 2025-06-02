import streamlit as st
import pandas as pd # Required for chat history download
import io # Required for chat history download
import time # Required for chat history download/timestamp

def render_file_uploaders(st_obj):
    """Renders file uploaders for PDF and TXT files."""
    st_obj.markdown("<h4 style='text-align: center;'>📄 कृपया योजनेचे तपशील फाइल अपलोड करा</h4>", unsafe_allow_html=True)
    uploaded_pdf = st_obj.file_uploader("स्किम तपशील पीडीएफ अपलोड करा", type=["pdf"])
    uploaded_txt = st_obj.file_uploader("आरोग्य योजना बुकलेट अपलोड करा", type=["txt"])
    return uploaded_pdf, uploaded_txt

def render_query_input(st_obj, whisper_client, transcribe_audio_func):
    """Renders text input and audio input for user queries."""
    st_obj.markdown("### Ask a question by typing or using audio input")
    col1, col2 = st_obj.columns([3, 1])
    
    with col1:
        default_value = st_obj.session_state.suggested_query if st_obj.session_state.suggested_query else ""
        user_input = st_obj.text_input(
            "Enter your question", 
            key="text_input", 
            placeholder="e.g. मुख्य योजना दाखवा / Show main schemes...",
            value=default_value
        )
        if st_obj.session_state.suggested_query:
            st_obj.session_state.suggested_query = ""
    
    with col2:
        audio_value = st.audio_input("🎤 Record your query")

    user_text = None
    if audio_value is not None:
        try:
            with st_obj.spinner("🎧 Transcribing audio..."):
                user_text = transcribe_audio_func(whisper_client, audio_value.getvalue())
            st_obj.success(f"🎧 Transcribed: {user_text}")
        except Exception as e:
            st_obj.error(f"Transcription Error: {str(e)}")
            
    return user_input, user_text

def render_answer_section(
    st_obj, 
    assistant_reply, 
    generate_audio_func, 
    create_audio_player_html_func,
    tts_speed, 
    voice_lang_pref,
    lang_code_to_name_map,
    allowed_tts_langs_set,
    tts_available_flag
):
    """Displays the assistant's answer and TTS audio player."""
    st_obj.markdown("### 📋 Answer:")
    
    is_cached = assistant_reply.startswith("[Cached]") if isinstance(assistant_reply, str) else False
    if is_cached:
        st_obj.info("🚀 This result was retrieved from cache (faster response)")
        clean_reply = assistant_reply.replace("[Cached] ", "")
    else:
        clean_reply = assistant_reply
    
    st_obj.markdown(
        f"<div style='background-color:#E8F5E9; padding:15px; border-radius:8px; margin-bottom:15px; border-left: 4px solid #4CAF50;'>"
        f"<b>🤖 Assistant:<br><br>{clean_reply}</div>", 
        unsafe_allow_html=True
    )
    
    if tts_available_flag:
        with st_obj.spinner("🔊 Generating voice response..."):
            try:
                audio_data, lang_used_for_tts, cache_hit = generate_audio_func(
                    text=clean_reply,
                    lang_preference=voice_lang_pref,
                    speed=tts_speed
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
                    speed_info = f" | Speed: {tts_speed}x"
                    st_obj.info(f"🔊 Voice: {lang_display}{speed_info} | {cache_indicator}")
                    
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
    elif clean_reply.strip(): # If TTS not available but there's text
        st_obj.info("ℹ️ TTS is not available. Text response is shown above.")


def render_chat_history(
    st_obj,
    pd_module,
    io_module,
    time_module,
    generate_audio_func,
    create_audio_player_html_func,
    tts_speed,
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
                                        lang_preference=voice_lang_pref,
                                        speed=tts_speed
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