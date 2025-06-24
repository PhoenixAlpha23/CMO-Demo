import streamlit as st
import time

def init_session_state():
    defaults = {
        "chat_history": [],
        "suggested_query": "",
        "last_query_time": 0,
        "rag_chain": None,
        "auto_play_tts": False,
        "current_model_key": "" # For tracking RAG chain rebuilds
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def check_rate_limit_delay(min_delay=2):
    """Check if we need to wait before making another query"""
    current_time = time.time()
    time_since_last = current_time - st.session_state.get("last_query_time", 0)
    
    if time_since_last < min_delay:
        wait_time = min_delay - time_since_last
        return wait_time
    return 0

def safe_get_cache_stats(get_audio_cache_stats_func):
    """Safely get cache stats with fallback values"""
    try:
        cache_stats = get_audio_cache_stats_func()
        if not isinstance(cache_stats, dict):
            # Fallback if the structure is not a dict
            return {"total_audio_cached": 0, "audio_cache_max_size": 0} 
        
        # Ensure expected keys exist, providing defaults if not
        return {
            "total_audio_cached": cache_stats.get("total_audio_cached", 0),
            "audio_cache_max_size": cache_stats.get("audio_cache_max_size", 0),
            # Add hit_rate if it's reliably calculated and returned by get_audio_cache_stats_func
        }
    except Exception as e:
        st.warning(f"Cache stats unavailable: {e}")
        return {"total_audio_cached": 0, "audio_cache_max_size": 0}

# Language name mapping for display
LANG_CODE_TO_NAME = {'en': 'English', 'hi': 'Hindi', 'mr': 'Marathi', 'auto': 'Auto-Detected'}
ALLOWED_TTS_LANGS = {'en', 'hi', 'mr'}