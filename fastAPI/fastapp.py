from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import time
import functools
import io
import base64
import json
import hashlib
import redis
import pickle
from contextlib import asynccontextmanager
import os
from groq import Groq
import asyncio
import ffmpeg
from core.transcription import transcribe_audio_whisper, transcribe_audio_robust
import logging
# Core services
from core.rag_services import build_rag_chain_with_model_choice, process_scheme_query_with_retry
from core.tts_services import generate_audio_response, TTS_AVAILABLE
from core.transcription import transcribe_audio, transcribe_audio_google
from utils.config import load_env_vars, GROQ_API_KEY
from utils.helpers import check_rate_limit_delay, LANG_CODE_TO_NAME, ALLOWED_TTS_LANGS

load_env_vars()

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost") 
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

class RedisManager:
    def __init__(self):
        self.redis_client = None
        self._connect()
    
    def _connect(self):
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=False,  # Keep as bytes for pickle
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            # Test connection
            self.redis_client.ping()
            print("Redis connected successfully")
        except Exception as e:
            print(f"Redis connection failed: {e}")
            self.redis_client = None

    def is_available(self) -> bool:
        return self.redis_client is not None
    
    def reconnect(self):
        """Reconnect to Redis if connection is lost."""
        self._connect()
    
    def set_rag_chain(self, key: str, rag_chain, expire_hours: int = 24):
        """Store RAG chain with expiration"""
        if not self.is_available() or self.redis_client is None:
            return False
        try:
            serialized = pickle.dumps(rag_chain)
            self.redis_client.setex(f"rag_chain:{key}", expire_hours * 3600, serialized)
            return True
        except Exception as e:
            print(f"Failed to store RAG chain: {e}")
            return False
    
    def get_rag_chain(self, key: str):
        """Retrieve RAG chain"""
        if not self.is_available() or self.redis_client is None:
            return None
        try:
            data = self.redis_client.get(f"rag_chain:{key}")
            if hasattr(data, '__await__') or not isinstance(data, (bytes, bytearray)):
                return None
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            print(f"Failed to retrieve RAG chain: {e}")
            return None
    
    def set_chat_history(self, session_id: str, history: List[dict], expire_hours: int = 48):
        """Store chat history"""
        if not self.is_available() or self.redis_client is None:
            return False
        try:
            serialized = json.dumps(history)
            self.redis_client.setex(f"chat:{session_id}", expire_hours * 3600, serialized)
            return True
        except Exception as e:
            print(f"Failed to store chat history: {e}")
            return False
    
    def get_chat_history(self, session_id: str) -> List[dict]:
        """Retrieve chat history"""
        if not self.is_available() or self.redis_client is None:
            return []
        try:
            data = self.redis_client.get(f"chat:{session_id}")
            if hasattr(data, '__await__') or not isinstance(data, (bytes, bytearray)):
                return []
            if data:
                try:
                    return json.loads(data.decode('utf-8'))
                except Exception:
                    return []
            return []
        except Exception as e:
            print(f"Failed to retrieve chat history: {e}")
            return []
    
    def add_chat_message(self, session_id: str, message: dict):
        """Add single message to chat history"""
        history = self.get_chat_history(session_id)
        history.insert(0, message)
        # Keep only last 50 messages
        history = history[:50]
        self.set_chat_history(session_id, history)
    
    def set_rate_limit(self, key: str, expire_seconds: int = 60):
        """Set rate limit marker"""
        if not self.is_available() or self.redis_client is None:
            return False
        try:
            self.redis_client.setex(f"rate_limit:{key}", expire_seconds, "1")
            return True
        except Exception:
            return False
    
    def check_rate_limit(self, key: str) -> bool:
        """Check if rate limited"""
        if not self.is_available() or self.redis_client is None:
            return False
        try:
            result = self.redis_client.exists(f"rate_limit:{key}")
            if hasattr(result, '__await__') or not isinstance(result, int):
                return False
            return result > 0
        except Exception:
            return False

# Initialize Redis manager
redis_manager = RedisManager()

# Enhanced STATE with Redis fallback
class LangChainStateManager:
    def __init__(self, redis_manager):
        self.redis = redis_manager
        self._rag_cache = {}
        self._memory_fallback = {}

    def store_rag_chain_config(self, model_key: str, 
                              pdf_bytes: Optional[bytes] = None,
                              txt_bytes: Optional[bytes] = None,
                              model_choice: str = "llama-3.3-70b-versatile",
                              enhanced_mode: bool = True,
                              pdf_name: str = "None",
                              txt_name: str = "None",
                              rag_chain=None):
        config = {
            "model_choice": model_choice,
            "enhanced_mode": enhanced_mode,
            "pdf_name": pdf_name,
            "txt_name": txt_name,
            "timestamp": time.time(),
            "pdf_content": base64.b64encode(pdf_bytes).decode() if pdf_bytes else None,
            "txt_content": base64.b64encode(txt_bytes).decode() if txt_bytes else None,
            "pdf_size": len(pdf_bytes) if pdf_bytes else 0,
            "txt_size": len(txt_bytes) if txt_bytes else 0,
        }
        stored = False
        if self.redis.is_available():
            try:
                config_json = json.dumps(config)
                self.redis.redis_client.setex(
                    f"rag_config:{model_key}", 
                    24 * 3600,
                    config_json
                )
                stored = True
                logging.info(f"RAG config stored in Redis for key: {model_key}")
            except Exception as e:
                logging.error(f"Failed to store config in Redis: {e}")
        if not stored:
            self._memory_fallback[f"rag_config:{model_key}"] = config
            logging.info(f"RAG config stored in memory for key: {model_key}")
        if rag_chain:
            self._rag_cache[model_key] = {
                "chain": rag_chain,
                "created_at": time.time()
            }
            logging.info(f"RAG chain cached in memory for key: {model_key}")
        return True

    def get_rag_chain(self, model_key: str, groq_api_key: str):
        if model_key in self._rag_cache:
            cached = self._rag_cache[model_key]
            if time.time() - cached["created_at"] < 3600:
                return cached["chain"]
            else:
                del self._rag_cache[model_key]
        config = self._get_rag_config(model_key)
        if not config:
            return None
        rag_chain = self._rebuild_rag_chain(config, groq_api_key)
        if rag_chain:
            self._rag_cache[model_key] = {
                "chain": rag_chain,
                "created_at": time.time()
            }
        return rag_chain

    def _get_rag_config(self, model_key: str) -> Optional[dict]:
        if self.redis.is_available():
            try:
                data = self.redis.redis_client.get(f"rag_config:{model_key}")
                if data:
                    return json.loads(data.decode('utf-8'))
            except Exception as e:
                logging.error(f"Failed to get config from Redis: {e}")
        return self._memory_fallback.get(f"rag_config:{model_key}")

    def _rebuild_rag_chain(self, config: dict, groq_api_key: str):
        try:
            from core.rag_services import build_rag_chain_with_model_choice
            pdf_bytes = base64.b64decode(config["pdf_content"]) if config.get("pdf_content") else None
            txt_bytes = base64.b64decode(config["txt_content"]) if config.get("txt_content") else None
            pdf_io = io.BytesIO(pdf_bytes) if pdf_bytes else None
            txt_io = io.BytesIO(txt_bytes) if txt_bytes else None
            rag_chain = build_rag_chain_with_model_choice(
                pdf_io,
                txt_io,
                groq_api_key,
                model_choice=config["model_choice"],
                enhanced_mode=config["enhanced_mode"]
            )
            return rag_chain
        except Exception as e:
            logging.error(f"Failed to rebuild RAG chain: {e}")
            return None

    # Chat history methods (unchanged)
    def get_chat_history(self, session_id: str = "default"):
        if self.redis.is_available():
            return self.redis.get_chat_history(session_id)
        return []

    def add_chat_message(self, message: dict, session_id: str = "default"):
        if self.redis.is_available():
            self.redis.add_chat_message(session_id, message)

# Initialize state manager
state_manager = LangChainStateManager(redis_manager)

# Dependency functions
def get_groq_client() -> Groq:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Missing GROQ_API_KEY")
    return Groq(api_key=GROQ_API_KEY)

def get_session_id(session_id: Optional[str] = None) -> str:
    """Get or generate session ID"""
    return session_id or "default"

# Utility functions
def generate_model_key(model: str, enhanced: bool, pdf_name: str, txt_name: str) -> str:
    """Generate unique key for model configuration"""
    key_string = f"{model}_{enhanced}_{pdf_name}_{txt_name}"
    return hashlib.md5(key_string.encode()).hexdigest()

def improved_rate_limit_check(session_id: str) -> Optional[float]:
    """Enhanced rate limiting with Redis"""
    if redis_manager.is_available():
        if redis_manager.check_rate_limit(session_id):
            return 5.0  # Wait 5 seconds
        redis_manager.set_rate_limit(session_id, 5)  # 5 second window
        return None
    else:
        # Fallback to original logic
        return check_rate_limit_delay()

def convert_webm_to_wav(webm_bytes):
    input_buffer = io.BytesIO(webm_bytes)
    out, _ = (
        ffmpeg
        .input('pipe:0')
        .output('pipe:1', format='wav', acodec='pcm_s16le', ac=1, ar='16000')
        .run(input=input_buffer.read(), capture_stdout=True, capture_stderr=True)
    )
    return out

# Enhanced models
class QueryRequest(BaseModel):
    input_text: str
    model: str = "llama-3.3-70b-versatile"
    enhanced_mode: bool = True
    voice_lang_pref: str = "auto"
    session_id: Optional[str] = None
    model_key: Optional[str] = None  # <-- Add this line

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up FastAPI application...")
    yield
    # Shutdown
    if redis_manager.is_available() and redis_manager.redis_client is not None:
        redis_manager.redis_client.close()
    print("FastAPI application shutting down...")

app = FastAPI(title="CMRF AI Agent", lifespan=lifespan)

# Enable CORS for all origins (for development). For production, specify allowed origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["http://localhost:3000"] for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    try:
        redis_status = redis_manager.is_available()
        return {"message": "CMRF AI Agent FastAPI backend is running.", "docs": "/docs", "health": "/health/", "redis_available": redis_status}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Startup error: {str(e)}"})

@app.post("/upload/")
async def upload_files(
    pdf_file: Optional[UploadFile] = File(None), 
    txt_file: Optional[UploadFile] = File(None),
    session_id: str = Depends(get_session_id),
    groq_client: Groq = Depends(get_groq_client)
):
    if not (pdf_file or txt_file):
        return JSONResponse(status_code=400, content={"error": "Please upload at least one file (PDF or TXT)."})

    pdf_name = pdf_file.filename if pdf_file else "None"
    txt_name = txt_file.filename if txt_file else "None"
    model_key = generate_model_key("llama-3.3-70b-versatile", True, pdf_name, txt_name)

    existing_chain = state_manager.get_rag_chain(model_key, GROQ_API_KEY)
    if existing_chain is not None:
        return {
            "message": "RAG system already initialized.", 
            "model_key": model_key,
            "source": "cache"
        }

    try:
        pdf_bytes = await pdf_file.read() if pdf_file else None
        txt_bytes = await txt_file.read() if txt_file else None
        from core.rag_services import build_rag_chain_with_model_choice
        rag_chain = build_rag_chain_with_model_choice(
            io.BytesIO(pdf_bytes) if pdf_bytes else None,
            io.BytesIO(txt_bytes) if txt_bytes else None,
            GROQ_API_KEY,
            model_choice="llama-3.3-70b-versatile",
            enhanced_mode=True
        )
        state_manager.store_rag_chain_config(
            model_key=model_key,
            pdf_bytes=pdf_bytes,
            txt_bytes=txt_bytes,
            pdf_name=pdf_name,
            txt_name=txt_name,
            rag_chain=rag_chain
        )
        return {
            "message": "RAG system initialized successfully.", 
            "model_key": model_key,
            "redis_available": redis_manager.is_available(),
            "storage_method": "configuration_based",
            "source": "fresh_build"
        }
    except Exception as e:
        import traceback
        logging.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": f"Failed to build RAG system: {str(e)}"})

@app.post("/query/")
async def get_answer(req: QueryRequest):
    input_text = req.input_text.strip()
    if not input_text:
        return JSONResponse(status_code=400, content={"error": "Empty query input."})

    session_id = req.session_id or "default"
    wait_time = improved_rate_limit_check(session_id)
    if wait_time:
        return JSONResponse(status_code=429, content={"message": f"Rate limited. Wait {wait_time:.1f} seconds."})

    model_key = req.model_key
    if not model_key:
        return JSONResponse(status_code=400, content={"error": "model_key is required. Please upload files first."})

    rag_chain = state_manager.get_rag_chain(model_key, GROQ_API_KEY)
    if not rag_chain:
        return JSONResponse(status_code=400, content={"error": "No RAG system found. Please upload files first."})

    try:
        from core.rag_services import process_scheme_query_with_retry
        result = process_scheme_query_with_retry(rag_chain, input_text)
        assistant_reply = result[0] if isinstance(result, tuple) else result or "No response received"
        message = {
            "user": input_text,
            "assistant": assistant_reply,
            "model": req.model,
            "timestamp": time.strftime("%H:%M:%S")
        }
        state_manager.add_chat_message(message, session_id)
        return {
            "reply": assistant_reply,
            "session_id": session_id,
            "model_key": model_key
        }
    except Exception as e:
        logging.error(f"Query processing error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/chat-history/")
async def get_chat_history(session_id: str = Depends(get_session_id)):
    try:
        history = state_manager.get_chat_history(session_id)
        return {
            "chat_history": history,
            "session_id": session_id,
            "redis_available": redis_manager.is_available()
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"{str(e)}"})

@app.post("/tts/")
async def get_audio(text: str = Form(...), lang_preference: str = Form("auto")):
    """Generate TTS audio from text"""
    try:
        # Check rate limit
        delay_needed = improved_rate_limit_check("tts")
        if delay_needed:
            return JSONResponse(
                status_code=429,
                content={"error": f"Rate limited. Please wait {delay_needed:.1f} seconds."}
            )
        
        # Generate audio
        audio_bytes, lang_used, cache_hit = generate_audio_response(text, lang_preference)
        
        if audio_bytes:
            # Convert to base64 for JSON response
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            return {
                "audio_base64": audio_base64,
                "lang_used": lang_used,
                "cache_hit": cache_hit,
                "text_length": len(text)
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
            
    except Exception as e:
        print(f"TTS Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe/")
async def transcribe_audio_endpoint(audio_file: UploadFile = File(...)):
    """Transcribe uploaded audio file"""
    try:
        # Check rate limit
        delay_needed = improved_rate_limit_check("transcribe")
        if delay_needed:
            return JSONResponse(
                status_code=429,
                content={"error": f"Rate limited. Please wait {delay_needed:.1f} seconds."}
            )
        
        # Read audio file
        audio_bytes = await audio_file.read()
        
        # Convert webm to wav if needed
        if audio_file.filename and audio_file.filename.endswith('.webm'):
            try:
                audio_bytes = convert_webm_to_wav(audio_bytes)
            except Exception as e:
                print(f"Audio conversion error: {e}")
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Failed to convert audio format: {str(e)}"}
                )
        
        # Validate audio file size
        if len(audio_bytes) < 100:  # Too small to be valid audio
            return JSONResponse(
                status_code=400,
                content={"error": "Audio file is too small or corrupted"}
            )
        
        # Transcribe using Whisper
        success, transcription = transcribe_audio_robust(audio_bytes)
        
        if success and transcription:
            # Ensure transcription is a string and strip whitespace
            transcription_text = str(transcription).strip()
            return {"transcription": transcription_text}
        else:
            error_msg = str(transcription) if not success else "Failed to transcribe audio"
            return JSONResponse(status_code=400, content={"error": error_msg})
            
    except Exception as e:
        print(f"Transcription Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health/")
async def health_check():
    try:
        redis_status = redis_manager.is_available()
        groq_status = True
        try:
            _ = get_groq_client()
        except Exception:
            groq_status = False
        return {
            "status": "ok",
            "redis_available": redis_status,
            "groq_available": groq_status,
            "timestamp": time.time()
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Health check failed: {str(e)}"})

@app.get("/sessions/")
async def list_sessions():
    if not redis_manager.is_available() or redis_manager.redis_client is None:
        return {"error": "Redis not available"}
    try:
        keys = redis_manager.redis_client.keys("chat:*")
        if hasattr(keys, '__await__'):
            keys = []
        if not hasattr(keys, '__iter__') or not hasattr(keys, '__getitem__'):
            keys = []
        sessions = []
        for key in keys:
            if hasattr(key, 'decode'):
                key = key.decode()
            sessions.append(key.replace("chat:", ""))
        return {"sessions": sessions}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    try:
        if redis_manager.is_available() and redis_manager.redis_client is not None:
            redis_manager.redis_client.delete(f"chat:{session_id}")
            return {"message": f"Session {session_id} cleared"}
        else:
            if session_id == "default":
                state_manager._memory_state["chat_history"] = []
            return {"message": f"Session {session_id} cleared (memory only)"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
