import streamlit as st

def render_sidebar(
    st_obj, 
    model_options, 
    safe_get_cache_stats_func, 
    clear_audio_cache_func
):
    """Renders the sidebar elements with enhanced UI and animations."""
    
    # Streamlined CSS with essential animations only
    st_obj.markdown("""
    <style>
    /* Sidebar opening animation - Multiple selectors for compatibility */
    .css-1d391kg,
    .css-1cypcdb,
    .css-17eq0hr,
    [data-testid="stSidebar"],
    .stSidebar > div {
        background: linear-gradient(135deg, #ff7e5f 0%, #feb47b 100%) !important;
        animation: slideIn 0.6s ease-out;
    }
    
    /* Force orange gradient on all sidebar elements */
    .stSidebar > div {
        background: linear-gradient(135deg, #ff6b35 0%, #f7931e 50%, #ffb347 100%) !important;
        box-shadow: 4px 0 20px rgba(255, 107, 53, 0.15);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        animation: slideIn 0.6s ease-out;
        position: relative;
    }
    
    /* Settings sections */
    .settings-section {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 12px;
        padding: 15px;
        margin: 12px 0;
        animation: fadeInUp 0.5s ease forwards;
        opacity: 0;
        transition: all 0.3s ease;
    }
    
    .settings-section:nth-of-type(1) { animation-delay: 0.2s; }
    .settings-section:nth-of-type(2) { animation-delay: 0.4s; }
    .settings-section:nth-of-type(3) { animation-delay: 0.6s; }
    
    .settings-section:hover {
        transform: translateY(-2px);
        background: rgba(255,255,255,0.15);
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.1em;
        font-weight: 600;
        color: #fff;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .section-header::after {
        content: '';
        flex: 1;
        height: 1px;
        background: rgba(255,255,255,0.3);
    }
    
    /* Enhanced form elements */
    .stSelectbox > div > div,
    .stCheckbox {
        transition: all 0.2s ease;
    }
    
    .stSelectbox > div > div:hover {
        transform: scale(1.02);
    }
    
    /* Model description */
    .model-description {
        background: rgba(255,255,255,0.1);
        border-left: 3px solid #4CAF50;
        padding: 8px 12px;
        margin: 8px 0;
        border-radius: 0 6px 6px 0;
        color: #fff;
        font-size: 0.9em;
    }
    
    /* Cache metrics */
    .cache-metric {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        margin: 8px 0;
        transition: transform 0.2s ease;
    }
    
    .cache-metric:hover {
        transform: translateY(-2px);
    }
    
    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 8px 16px;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(255, 65, 108, 0.3);
    }
    
    /* Animations */
    @keyframes slideIn {
        from {
            transform: translateX(-100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Settings Section
    st_obj.sidebar.markdown("""
    <div class="settings-section">
        <div class="section-header">⚙️ Settings</div>
    """, unsafe_allow_html=True)
    
    # Model selection
    selected_model = st_obj.sidebar.selectbox(
        "Choose Model (for rate limit management):",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x]["name"],
        index=0
    )
    
    st_obj.sidebar.markdown(f"""
    <div class="model-description">
        <strong>Selected:</strong> {model_options[selected_model]['description']}
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced mode toggle
    enhanced_mode = st_obj.sidebar.checkbox(
        "🚀 Enhanced Mode", 
        value=True, 
        help="Better scheme coverage but uses more tokens"
    )
    
    st_obj.sidebar.markdown("</div>", unsafe_allow_html=True)
    
    # TTS Settings Section
    st_obj.sidebar.markdown("""
    <div class="settings-section">
        <div class="section-header">🔊 Text-to-Speech Settings</div>
    """, unsafe_allow_html=True)
    
    # Auto-play toggle
    st_obj.session_state.auto_play_tts = st_obj.sidebar.checkbox(
        "🎛️ Auto-play TTS", 
        value=st_obj.session_state.get("auto_play_tts", False),
        help="Automatically play voice responses"
    )
    
    # Voice language preference
    voice_lang_pref = st_obj.sidebar.selectbox(
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
    
    st_obj.sidebar.markdown("</div>", unsafe_allow_html=True)
    
    # Audio Cache Section
    s_cache_stats = safe_get_cache_stats_func()
    if s_cache_stats.get("total_audio_cached", 0) > 0:
        st_obj.sidebar.markdown("""
        <div class="settings-section">
            <div class="section-header">🧠 Audio Cache</div>
        """, unsafe_allow_html=True)
        
        st_obj.sidebar.markdown(f"""
        <div class="cache-metric">
            <div style="font-size: 1.5em; font-weight: bold;">
                {s_cache_stats["total_audio_cached"]}
            </div>
            <div style="font-size: 0.85em; opacity: 0.9;">
                Cached Audio Files
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st_obj.sidebar.button("🗑️ Clear Audio Cache"):
            clear_audio_cache_func()
            st_obj.sidebar.success("✨ Audio cache cleared!")
            st_obj.rerun()
        
        st_obj.sidebar.markdown("</div>", unsafe_allow_html=True)
            
    return selected_model, enhanced_mode, voice_lang_pref