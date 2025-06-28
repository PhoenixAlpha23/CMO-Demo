from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import time
import functools
import io
import base64
import ffmpeg

from groq import Groq

# Core services
from core.rag_services import build_rag_chain_with_model_choice, process_scheme_query_with_retry
from core.tts_services import generate_audio_response, TTS_AVAILABLE
#from core.cache_manager import _audio_cache
from core.transcription import transcribe_audio
from utils.config import load_env_vars, GROQ_API_KEY
from utils.helpers import check_rate_limit_delay, LANG_CODE_TO_NAME, ALLOWED_TTS_LANGS

load_env_vars()

app = FastAPI(title="CMRF AI Agent")

# Global store (can be replaced with Redis or DB for production)
STATE = {
    "rag_chain": None,
    "current_model_key": "",
    "chat_history": [],
    "last_query_time": 0
}

class QueryRequest(BaseModel):
    input_text: str
    model: str = "llama-3.3-70b-versatile"
    enhanced_mode: bool = True
    voice_lang_pref: str = "auto"

@app.post("/upload/")
async def upload_files(pdf_file: Optional[UploadFile] = File(None), txt_file: Optional[UploadFile] = File(None)):
    if not GROQ_API_KEY:
        return JSONResponse(status_code=500, content={"error": "Missing GROQ_API_KEY."})
    
    if not (pdf_file or txt_file):
        return JSONResponse(status_code=400, content={"error": "Please upload at least one file (PDF or TXT)."})

    pdf_name = pdf_file.filename if pdf_file else "None"
    txt_name = txt_file.filename if txt_file else "None"
    current_model_key = f"llama-3.3-70b-versatile_True_{pdf_name}_{txt_name}"

    # Build RAG chain if needed
    if STATE["rag_chain"] is None or STATE["current_model_key"] != current_model_key:
        try:
            if pdf_file:
                pdf_bytes = await pdf_file.read()
            else:
                pdf_bytes = None
            if txt_file:
                txt_bytes = await txt_file.read()
            else:
                txt_bytes = None

            rag_chain = build_rag_chain_with_model_choice(
                io.BytesIO(pdf_bytes) if pdf_bytes else None,
                io.BytesIO(txt_bytes) if txt_bytes else None,
                GROQ_API_KEY,
                model_choice="llama-3.3-70b-versatile",
                enhanced_mode=True
            )
            STATE["rag_chain"] = rag_chain
            STATE["current_model_key"] = current_model_key
            return {"message": "RAG system initialized successfully."}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Failed to build RAG system: {str(e)}"})
    
    return {"message": "RAG system already initialized."}

@app.post("/query/")
async def get_answer(req: QueryRequest):
    input_text = req.input_text.strip()
    if not input_text:
        return JSONResponse(status_code=400, content={"error": "Empty query input."})

    # Rate limit
    wait_time = check_rate_limit_delay()
    if wait_time > 0:
        return JSONResponse(status_code=429, content={"message": f"Rate limited. Wait {wait_time:.1f} seconds."})

    try:
        STATE["last_query_time"] = time.time()
        result = process_scheme_query_with_retry(STATE["rag_chain"], input_text)
        assistant_reply = result[0] if isinstance(result, tuple) else result or "No response received"
        STATE["chat_history"].insert(0, {
            "user": input_text,
            "assistant": assistant_reply,
            "model": req.model,
            "timestamp": time.strftime("%H:%M:%S")
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"{str(e)}"})

    return {"reply": assistant_reply}

@app.get("/chat-history/")
async def get_chat_history():
    return {"chat_history": STATE["chat_history"]}

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
    return {"status": "ok"}

def convert_to_wav(audio_bytes):
    input_buffer = io.BytesIO(audio_bytes)
    out, _ = (
        ffmpeg
        .input('pipe:0')
        .output('pipe:1', format='wav', acodec='pcm_s16le', ac=1, ar='16000')
        .run(input=input_buffer.read(), capture_stdout=True, capture_stderr=True)
    )
    return out

@app.post("/transcribe")
async def transcribe_endpoint(audio: UploadFile = File(...)):
    try:
        audio_bytes = await audio.read()
        # Always convert to mono 16kHz WAV
        wav_bytes = convert_to_wav(audio_bytes)
        from core.transcription import transcribe_audio_whisper
        success, transcription = transcribe_audio_whisper(wav_bytes, model_name="large")
        if not success:
            return {"error": transcription}
        return {"transcription": transcription}
    except Exception as e:
        return {"error": str(e)}
