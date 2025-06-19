from flask import Flask, render_template, request, redirect, url_for, send_file, flash # type: ignore
from werkzeug.utils import secure_filename # type: ignore
import os
import time
import functools

# Core services
from core.rag_services import build_rag_chain_with_model_choice, process_scheme_query_with_retry
from core.tts_services import generate_audio_response, TTS_AVAILABLE
from core.cache_manager import _audio_cache, get_audio_hash, cache_audio, get_cached_audio
from core.transcription import transcribe_audio
from utils.config import load_env_vars, GROQ_API_KEY
from utils.helpers import check_rate_limit_delay, LANG_CODE_TO_NAME, ALLOWED_TTS_LANGS

load_env_vars()

#app = Flask(__name__)
app = Flask(__name__, template_folder="templates")
app.secret_key = "your-secret-key"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# In-memory session
session_state = {
    "chat_history": [],
    "rag_chain": None,
    "last_query_time": 0,
    "current_model_key": ""
}

@app.route('/', methods=["GET", "POST"])
def index():
    message = ""
    answer = ""
    audio_url = None

    if request.method == "POST":
        pdf_file = request.files.get("pdf_file")
        txt_file = request.files.get("txt_file")
        user_query = request.form.get("user_query")

        pdf_path = os.path.join(UPLOAD_FOLDER, secure_filename(pdf_file.filename)) if pdf_file else None
        txt_path = os.path.join(UPLOAD_FOLDER, secure_filename(txt_file.filename)) if txt_file else None

        if pdf_file:
            pdf_file.save(pdf_path)
        if txt_file:
            txt_file.save(txt_path)

        # Model and config setup
        selected_model = "llama-3.3-70b-versatile"
        enhanced_mode = True
        voice_lang_pref = "auto"

        # Build RAG chain if not cached
        current_model_key = f"{selected_model}_{enhanced_mode}_{pdf_file.filename if pdf_file else 'None'}_{txt_file.filename if txt_file else 'None'}"
        if session_state["rag_chain"] is None or session_state["current_model_key"] != current_model_key:
            try:
                session_state["rag_chain"] = build_rag_chain_with_model_choice(
                    pdf_path if pdf_file else None,
                    txt_path if txt_file else None,
                    GROQ_API_KEY,
                    model_choice=selected_model,
                    enhanced_mode=enhanced_mode
                )
                session_state["current_model_key"] = current_model_key
                flash("✅ RAG system ready!", "success")
            except Exception as e:
                flash(f"Error building RAG chain: {e}", "danger")
                return redirect(url_for('index'))

        # Query processing
        if user_query:
            wait_time = check_rate_limit_delay()
            if wait_time > 0:
                flash(f"Please wait {wait_time:.1f}s to avoid rate limits.", "warning")
                time.sleep(wait_time)

            try:
                result = process_scheme_query_with_retry(session_state["rag_chain"], user_query)
                assistant_reply = result[0] if isinstance(result, tuple) else result or "No response"

                session_state["chat_history"].insert(0, {
                    "user": user_query,
                    "assistant": assistant_reply,
                    "timestamp": time.strftime("%H:%M:%S")
                })

                answer = assistant_reply

                # Optional: Generate TTS
                if TTS_AVAILABLE:
                    audio_data, _, _ = generate_audio_response(
                        text=answer,
                        lang_preference=voice_lang_pref,
                        audio_cache=_audio_cache,
                        get_audio_hash_func=get_audio_hash,
                        cache_audio_func=cache_audio,
                        get_cached_audio_func=get_cached_audio
                    )
                    if audio_data:
                        audio_path = os.path.join(UPLOAD_FOLDER, "response.mp3")
                        with open(audio_path, "wb") as f:
                            f.write(audio_data)
                        audio_url = url_for("static", filename="response.mp3")
            except Exception as e:
                flash(f"Query error: {e}", "danger")

    return render_template("index.html", chat_history=session_state["chat_history"], answer=answer, audio_url=audio_url)

if __name__ == '__main__':
    app.run(debug=True)
