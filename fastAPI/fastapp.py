from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
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

# Core services
from core.rag_services import build_rag_chain_with_model_choice, process_scheme_query_with_retry
from core.tts_services import generate_audio_response, TTS_AVAILABLE
from core.transcription import transcribe_audio
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
    
    def set_rag_chain(self, key: str, rag_chain, expire_hours: int = 24):
        """Store RAG chain with expiration"""
        if not self.is_available():
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
        if not self.is_available():
            return None
        try:
            data = self.redis_client.get(f"rag_chain:{key}")
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            print(f"Failed to retrieve RAG chain: {e}")
            return None
    
    def set_chat_history(self, session_id: str, history: List[dict], expire_hours: int = 48):
        """Store chat history"""
        if not self.is_available():
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
        if not self.is_available():
            return []
        try:
            data = self.redis_client.get(f"chat:{session_id}")
            if data:
                return json.loads(data.decode('utf-8'))
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
        if not self.is_available():
            return False
        try:
            self.redis_client.setex(f"rate_limit:{key}", expire_seconds, "1")
            return True
        except Exception:
            return False
    
    def check_rate_limit(self, key: str) -> bool:
        """Check if rate limited"""
        if not self.is_available():
            return False
        try:
            return self.redis_client.exists(f"rate_limit:{key}") > 0
        except Exception:
            return False

# Initialize Redis manager
redis_manager = RedisManager()

# Enhanced STATE with Redis fallback
class StateManager:
    def __init__(self, redis_manager: RedisManager):
        self.redis = redis_manager
        # In-memory fallback
        self._memory_state = {
            "rag_chain": None,
            "current_model_key": "",
            "chat_history": [],
            "last_query_time": 0
        }
    
    def get_rag_chain(self, model_key: str):
        if self.redis.is_available():
            return self.redis.get_rag_chain(model_key)
        return self._memory_state.get("rag_chain") if self._memory_state.get("current_model_key") == model_key else None
    
    def set_rag_chain(self, model_key: str, rag_chain):
        if self.redis.is_available():
            self.redis.set_rag_chain(model_key, rag_chain)
        else:
            self._memory_state["rag_chain"] = rag_chain
            self._memory_state["current_model_key"] = model_key
    
    def get_chat_history(self, session_id: str = "default") -> List[dict]:
        if self.redis.is_available():
            return self.redis.get_chat_history(session_id)
        return self._memory_state.get("chat_history", [])
    
    def add_chat_message(self, message: dict, session_id: str = "default"):
        if self.redis.is_available():
            self.redis.add_chat_message(session_id, message)
        else:
            self._memory_state["chat_history"].insert(0, message)
            # Keep only last 50 messages in memory
            self._memory_state["chat_history"] = self._memory_state["chat_history"][:50]

# Initialize state manager
state_manager = StateManager(redis_manager)

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

# Enhanced models
class QueryRequest(BaseModel):
    input_text: str
    model: str = "llama-3.3-70b-versatile"
    enhanced_mode: bool = True
    voice_lang_pref: str = "auto"
    session_id: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up FastAPI application...")
    yield
    # Shutdown
    if redis_manager.is_available():
        redis_manager.redis_client.close()
    print("FastAPI application shutting down...")

app = FastAPI(title="CMRF AI Agent", lifespan=lifespan)

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
    
    # Generate model key
    model_key = generate_model_key("llama-3.3-70b-versatile", True, pdf_name, txt_name)

    # Check if RAG chain already exists
    existing_chain = state_manager.get_rag_chain(model_key)
    if existing_chain is not None:
        return {"message": "RAG system already initialized.", "model_key": model_key}

    try:
        # Read files
        pdf_bytes = await pdf_file.read() if pdf_file else None
        txt_bytes = await txt_file.read() if txt_file else None

        # Build RAG chain
        rag_chain = build_rag_chain_with_model_choice(
            io.BytesIO(pdf_bytes) if pdf_bytes else None,
            io.BytesIO(txt_bytes) if txt_bytes else None,
            GROQ_API_KEY,
            model_choice="llama-3.3-70b-versatile",
            enhanced_mode=True
        )
        
        # Store in Redis/memory
        state_manager.set_rag_chain(model_key, rag_chain)
        
        return {
            "message": "RAG system initialized successfully.", 
            "model_key": model_key,
            "redis_available": redis_manager.is_available()
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to build RAG system: {str(e)}"})

@app.post("/query/")
async def get_answer(req: QueryRequest):
    input_text = req.input_text.strip()
    if not input_text:
        return JSONResponse(status_code=400, content={"error": "Empty query input."})

    session_id = req.session_id or "default"
    
    # Enhanced rate limiting
    wait_time = improved_rate_limit_check(session_id)
    if wait_time:
        return JSONResponse(status_code=429, content={"message": f"Rate limited. Wait {wait_time:.1f} seconds."})

    # Get appropriate RAG chain (simplified - you might want to pass model_key)
    # For now, try to get any available chain
    rag_chain = None
    if redis_manager.is_available():
        # Try to get the most recent chain (this is simplified)
        try:
            keys = redis_manager.redis_client.keys("rag_chain:*")
            if keys:
                rag_chain = redis_manager.get_rag_chain(keys[0].decode().replace("rag_chain:", ""))
        except Exception:
            pass
    
    if not rag_chain:
        # Fallback to memory state
        rag_chain = state_manager._memory_state.get("rag_chain")
    
    if not rag_chain:
        return JSONResponse(status_code=400, content={"error": "No RAG system initialized. Please upload files first."})

    try:
        result = process_scheme_query_with_retry(rag_chain, input_text)
        assistant_reply = result[0] if isinstance(result, tuple) else result or "No response received"
        
        # Store chat message
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
            "redis_available": redis_manager.is_available()
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"{str(e)}"})

@app.get("/chat-history/")
async def get_chat_history(session_id: str = Depends(get_session_id)):
    history = state_manager.get_chat_history(session_id)
    return {
        "chat_history": history,
        "session_id": session_id,
        "redis_available": redis_manager.is_available()
    }

@app.post("/tts/")
async def get_audio(text: str = Form(...), lang_preference: str = Form("auto")):
    if not TTS_AVAILABLE:
        return JSONResponse(status_code=501, content={"error": "TTS not available."})

    try:
        audio_data, lang_used, cache_hit = generate_audio_response(
            text=text,
            lang_preference=lang_preference
        )
        return JSONResponse(content={
            "lang_used": lang_used,
            "cache_hit": cache_hit,
            "audio_base64": base64.b64encode(audio_data).decode('utf-8') if audio_data else None
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"TTS generation failed: {str(e)}"})

@app.get("/health/")
async def health_check():
    return {
        "status": "ok",
        "redis_available": redis_manager.is_available(),
        "timestamp": time.time()
    }

@app.post("/transcribe/")
async def transcribe_audio_endpoint(
    audio_file: UploadFile = File(...),
    groq_client: Groq = Depends(get_groq_client)
):
    try:
        audio_bytes = await audio_file.read()
        success, result = transcribe_audio(groq_client, audio_bytes)
        if success:
            return {"transcription": result}
        else:
            return JSONResponse(status_code=400, content={"error": result})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Transcription failed: {str(e)}"})

# Additional utility endpoints
@app.get("/sessions/")
async def list_sessions():
    """List all active sessions (Redis only)"""
    if not redis_manager.is_available():
        return {"error": "Redis not available"}
    
    try:
        keys = redis_manager.redis_client.keys("chat:*")
        sessions = [key.decode().replace("chat:", "") for key in keys]
        return {"sessions": sessions}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a specific session"""
    if redis_manager.is_available():
        try:
            redis_manager.redis_client.delete(f"chat:{session_id}")
            return {"message": f"Session {session_id} cleared"}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})
    else:
        if session_id == "default":
            state_manager._memory_state["chat_history"] = []
        return {"message": f"Session {session_id} cleared (memory only)"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)