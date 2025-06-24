import streamlit as st
import pandas as pd
import io
import time
import requests
import base64
import json
from typing import Optional, Tuple

# UI module imports (keep your existing UI components)
from ui.main_panel import render_file_uploaders, render_query_input, render_answer_section, render_chat_history, render_footer
from ui.components import create_audio_player_html

# Utility imports
from utils.config import load_env_vars
from utils.helpers import init_session_state, LANG_CODE_TO_NAME, ALLOWED_TTS_LANGS

load_env_vars()

# API Configuration
API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")

class APIClient:
    """Client for communicating with FastAPI backend"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session_id = st.session_state.get('api_session_id', 'streamlit_session')
    
    def health_check(self) -> bool:
        """Check if API is available"""
        try:
            response = self.session.get(f"{self.base_url}/health/", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def upload_files(self, pdf_file=None, txt_file=None) -> Tuple[bool, dict]:
        """Upload files to initialize RAG system"""
        try:
            files = {}
            if pdf_file:
                files['pdf_file'] = (pdf_file.name, pdf_file.getvalue(), 'application/pdf')
            if txt_file:
                files['txt_file'] = (txt_file.name, txt_file.getvalue(), 'text/plain')
            
            response = self.session.post(
                f"{self.base_url}/upload/", 
                files=files,
                timeout=120  # Longer timeout for file processing
            )
            
            if response.status_code == 200:
                return True, response.json()
            else:
                error_data = response.json() if response.headers.get('content-type') == 'application/json' else {"error": response.text}
                return False, error_data
                
        except requests.exceptions.Timeout:
            return False, {"error": "Request timeout - file processing took too long"}
        except Exception as e:
            return False, {"error": str(e)}
    
    def query(self, input_text: str, model: str = "llama-3.3-70b-versatile", 
              enhanced_mode: bool = True, voice_lang_pref: str = "auto") -> Tuple[bool, dict]:
        """Send query to RAG system"""
        try:
            payload = {
                "input_text": input_text,
                "model": model,
                "enhanced_mode": enhanced_mode,
                "voice_lang_pref": voice_lang_pref,
                "session_id": self.session_id
            }
            
            response = self.session.post(
                f"{self.base_url}/query/",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                return True, response.json()
            elif response.status_code == 429:
                # Rate limited
                return False, response.json()
            else:
                error_data = response.json() if response.headers.get('content-type') == 'application/json' else {"error": response.text}
                return False, error_data
                
        except requests.exceptions.Timeout:
            return False, {"error": "Query timeout - please try a simpler question"}
        except Exception as e:
            return False, {"error": str(e)}
    
    def get_chat_history(self) -> Tuple[bool, list]:
        """Get chat history for current session"""
        try:
            params = {"session_id": self.session_id}
            response = self.session.get(f"{self.base_url}/chat-history/", params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return True, data.get("chat_history", [])
            else:
                return False, []
        except Exception as e:
            st.error(f"Failed to fetch chat history: {e}")
            return False, []
    
    def transcribe_audio(self, audio_bytes: bytes) -> Tuple[bool, str]:
        """Transcribe audio using API"""
        try:
            files = {"audio_file": ("audio.wav", audio_bytes, "audio/wav")}
            response = self.session.post(
                f"{self.base_url}/transcribe/",
                files=files,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return True, data.get("transcription", "")
            else:
                error_data = response.json() if response.headers.get('content-type') == 'application/json' else {"error": response.text}
                return False, error_data.get("error", "Transcription failed")
                
        except Exception as e:
            return False, str(e)
    
    def generate_tts(self, text: str, lang_preference: str = "auto") -> Tuple[bool, dict]:
        """Generate TTS audio"""
        try:
            data = {
                "text": text,
                "lang_preference": lang_preference
            }
            response = self.session.post(
                f"{self.base_url}/tts/",
                data=data,
                timeout=30
            )
            
            if response.status_code == 200:
                return True, response.json()
            else:
                error_data = response.json() if response.headers.get('content-type') == 'application/json' else {"error": response.text}
                return False, error_data
                
        except Exception as e:
            return False, {"error": str(e)}

# Initialize API client
@st.cache_resource
def get_api_client():
    return APIClient(API_BASE_URL)

# Custom transcription function that uses API
def api_transcribe_audio(client_unused, audio_bytes):
    """Wrapper for API transcription to match existing interface"""
    api_client = get_api_client()
    success, result = api_client.transcribe_audio(audio_bytes)
    return success, result

# Custom TTS function that uses API
def api_generate_audio_response(text: str, lang_preference: str = "auto"):
    """Wrapper for API TTS to match existing interface"""
    api_client = get_api_client()
    success, result = api_client.generate_tts(text, lang_preference)
    
    if success and result.get("audio_base64"):
        # Convert base64 back to bytes
        audio_bytes = base64.b64decode(result["audio_base64"])
        return audio_bytes, result.get("lang_used", "en"), result.get("cache_hit", False)
    else:
        return None, "en", False

def init_streamlit_session_state():
    """Initialize session state for Streamlit"""
    if 'api_session_id' not in st.session_state:
        import uuid
        st.session_state.api_session_id = str(uuid.uuid4())
    if 'rag_initialized' not in st.session_state:
        st.session_state.rag_initialized = False
    if 'current_files_hash' not in st.session_state:
        st.session_state.current_files_hash = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'api_available' not in st.session_state:
        st.session_state.api_available = None
    if 'suggested_query' not in st.session_state:
        st.session_state.suggested_query = ""
    if 'auto_play_tts' not in st.session_state:
        st.session_state.auto_play_tts = False

def check_api_availability():
    """Check if API is available and cache result"""
    if st.session_state.api_available is None:
        api_client = get_api_client()
        st.session_state.api_available = api_client.health_check()
    return st.session_state.api_available

def get_files_hash(pdf_file, txt_file):
    """Generate hash for uploaded files to detect changes"""
    import hashlib
    hasher = hashlib.md5()
    if pdf_file:
        hasher.update(pdf_file.name.encode())
        hasher.update(str(pdf_file.size).encode())
    if txt_file:
        hasher.update(txt_file.name.encode())
        hasher.update(str(txt_file.size).encode())
    return hasher.hexdigest()

def main():
    st.set_page_config(page_title="RAG Agent", layout="wide")
    
    # Header
    col1, col2, col3 = st.columns([2, 2, 1])
    with col2:
        st.image("assets/cmrf logo.jpg", width=250)
    st.markdown("<h1 style='text-align: center;'>🤖 CMRF AI AGENT </h1>", unsafe_allow_html=True)
    
    # Initialize session state
    init_streamlit_session_state()
    
    # Check API availability
    if not check_api_availability():
        st.error("❌ FastAPI backend is not available. Please ensure the API server is running at " + API_BASE_URL)
        st.info("To start the API server, run: `uvicorn main:app --reload`")
        st.stop()
    else:
        st.success("✅ Connected to FastAPI backend")
    
    api_client = get_api_client()

    # Upload files
    uploaded_pdf, uploaded_txt = render_file_uploaders(st)

    if not (uploaded_pdf or uploaded_txt):
        st.warning("कृपया पुढे जानेसाठी किमान एक फाइल (PDF किंवा TXT) अपलोड करा.")
        st.stop()

    # Settings (hardcoded as in original)
    selected_model = "llama-3.3-70b-versatile"
    enhanced_mode = True
    voice_lang_pref = "auto"
    
    # Check if files changed and reinitialize if needed
    current_files_hash = get_files_hash(uploaded_pdf, uploaded_txt)
    files_changed = st.session_state.current_files_hash != current_files_hash
    
    if not st.session_state.rag_initialized or files_changed:
        with st.spinner("🔧 Initializing AI system via API... Please wait."):
            try:
                success, result = api_client.upload_files(uploaded_pdf, uploaded_txt)
                
                if success:
                    st.session_state.rag_initialized = True
                    st.session_state.current_files_hash = current_files_hash
                    
                    msg = st.empty()
                    msg.success("✅ RAG Agent ready!")
                    time.sleep(2)
                    msg.empty()
                    
                    # Display additional info
                    if result.get("redis_available"):
                        st.info("📊 Using Redis for persistent storage")
                else:
                    st.error(f"Failed to initialize RAG system: {result.get('error', 'Unknown error')}")
                    st.stop()
                    
            except Exception as e:
                st.error(f"Failed to communicate with API: {e}")
                st.stop()

    # Load chat history from API
    if st.session_state.rag_initialized:
        success, api_history = api_client.get_chat_history()
        if success and api_history:
            st.session_state.chat_history = api_history

    # Input Section (modified to use API transcription)
    user_input, user_text, get_answer_clicked = render_query_input(
        st, 
        None,  # No need for whisper client 
        api_transcribe_audio  # Use API transcription
    )

    # Query processing
    if get_answer_clicked or user_text:
        input_text = user_text if user_text else user_input.strip()
        if input_text:
            try:
                with st.spinner("🔍 Processing query via API..."):
                    success, result = api_client.query(
                        input_text=input_text,
                        model=selected_model,
                        enhanced_mode=enhanced_mode,
                        voice_lang_pref=voice_lang_pref
                    )
                    
                    if success:
                        assistant_reply = result.get("reply", "No response received")
                        
                        # Update local chat history
                        chat_entry = {
                            "user": input_text,
                            "assistant": assistant_reply,
                            "model": selected_model,
                            "timestamp": time.strftime("%H:%M:%S")
                        }
                        st.session_state.chat_history.insert(0, chat_entry)
                        
                        # Render answer with TTS
                        render_answer_section(
                            st,
                            assistant_reply,
                            api_generate_audio_response,  # Use API TTS
                            create_audio_player_html,
                            voice_lang_pref,
                            LANG_CODE_TO_NAME,
                            ALLOWED_TTS_LANGS,
                            True  # TTS_AVAILABLE = True (API handles availability)
                        )
                    else:
                        error_msg = result.get("error", "Unknown error")
                        if "rate limited" in error_msg.lower() or result.get("message", "").startswith("Rate limited"):
                            st.warning(f"⏳ {result.get('message', error_msg)}")
                        else:
                            st.error(f"Error: {error_msg}")
                            
            except Exception as e:
                st.error(f"Error communicating with API: {e}")
        else:
            st.warning("Please enter a question or record audio.")

    # Chat History (using API TTS)
    render_chat_history(
        st,
        pd,
        io,
        time,
        api_generate_audio_response,  # Use API TTS
        create_audio_player_html,
        voice_lang_pref,
        True,  # TTS_AVAILABLE
        LANG_CODE_TO_NAME,
        ALLOWED_TTS_LANGS
    )

    # Footer
    render_footer(st, selected_model)
    
    # Debug info in sidebar
    with st.sidebar:
        st.subheader("🔧 Debug Info")
        st.write(f"API URL: {API_BASE_URL}")
        st.write(f"Session ID: {st.session_state.api_session_id}")
        st.write(f"RAG Initialized: {st.session_state.rag_initialized}")
        st.write(f"Chat History Length: {len(st.session_state.chat_history)}")
        
        if st.button("🔄 Reset Session"):
            # Clear session state
            for key in ['rag_initialized', 'current_files_hash', 'chat_history']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        if st.button("🧹 Clear API Chat History"):
            try:
                # Call API to clear session
                response = requests.delete(f"{API_BASE_URL}/sessions/{st.session_state.api_session_id}")
                if response.status_code == 200:
                    st.session_state.chat_history = []
                    st.success("Chat history cleared!")
                else:
                    st.error("Failed to clear chat history")
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()