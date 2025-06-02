import streamlit as st

def render_sidebar(
    st_obj, 
    model_options, 
    safe_get_cache_stats_func, 
    clear_audio_cache_func
):
    """Renders the sidebar elements and returns selected settings."""
    st_obj.sidebar.markdown("### ⚙️ Settings")
    
    # Model selection
    selected_model = st_obj.sidebar.selectbox(
        "Choose Model (for rate limit management):",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x]["name"],
        index=0  # Default to fastest model
    )
    st_obj.sidebar.markdown(f"**Selected:** {model_options[selected_model]['description']}")
    
    # Enhanced mode toggle
    enhanced_mode = st_obj.sidebar.checkbox(
        "Enhanced Mode", 
        value=True, 
        help="Better scheme coverage but uses more tokens"
    )
    
    # TTS Settings
    st_obj.sidebar.markdown("### 🔊 Text-to-Speech Settings")
    
    # Auto-play toggle
    st_obj.session_state.auto_play_tts = st_obj.sidebar.checkbox(
        "🎛️ Auto-play TTS", 
        value=st_obj.session_state.get("auto_play_tts", False),
        help="Automatically play voice responses"
    )
    
    # TTS Speed control
    tts_speed = st_obj.sidebar.slider(
        "🎵 Speech Speed", 
        min_value=0.5, 
        max_value=2.0, 
        value=1.0, 
        step=0.1,
        help="Adjust speech speed (1.0 = normal)"
    )
    
    # Voice language preference
    voice_lang_pref = st_obj.sidebar.selectbox(
        "🌐 Voice Language Preference",
        options=["auto", "en", "hi", "mr"],
        format_func=lambda x: {"auto": "🧠 Auto-detect", "en": "🇺🇸 English", "hi": "🇮🇳 Hindi", "mr": "🇮🇳 Marathi"}[x],
        help="Language for voice synthesis"
    )
    
    # Audio cache stats
    s_cache_stats = safe_get_cache_stats_func() # Call the passed function
    if s_cache_stats.get("total_audio_cached", 0) > 0:
        st_obj.sidebar.markdown("### 🧠 Audio Cache")
        st_obj.sidebar.metric("Cached Audio Files", s_cache_stats["total_audio_cached"])
        if st_obj.sidebar.button("🗑️ Clear Audio Cache"):
            clear_audio_cache_func() # Call the passed function
            st_obj.sidebar.success("Audio cache cleared!")
            st_obj.rerun()
            
    return selected_model, enhanced_mode, tts_speed, voice_lang_pref