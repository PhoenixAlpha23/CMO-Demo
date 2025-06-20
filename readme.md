This is the Demo application for our CMRF AI project. You can view the application [here](https://raghu-cmrf.streamlit.app/).
The knowledge base is used to provide information about various government schemes, in english and marathi.

```
web/
├── app.py                   # Main Streamlit application (renamed from rag_app2.py)
├── core/
│   ├── __init __.py
│   ├── rag_services.py      # RAG chain building, query processing, scheme extraction
│   ├── tts_services.py      # TTS generation, language detection, audio utilities
│   ├── cache_manager.py     # Manages query and audio caches
│   └── transcription.py     # Audio transcription logic with restrction for 3 languages only
├── ui/
│   ├── __init __.py
│   ├── sidebar.py           # Functions to build the sidebar
│   ├── main_panel.py        # Functions for the main application layout (input, output, history)
│   └── components.py        # Reusable UI components (e.g., audio player HTML)
├── utils/
│   ├── __init __.py
│   ├── config.py            # Configuration loading (e.g., API keys)
│   └── helpers.py           # General helper functions, constants
├── .env                     # Environment variables
├── requirements.txt
├── main.py                  # streamlit application
└──fastapp.py                # fastapi set up
```

## Features

### Speech-to-Text Functionality
*   Accurately understands citizen queries during calls using Whisper.
*   Searches the knowledge base based on the query and suggests relevant solutions.
*   Recommends necessary next steps to resolve the citizen's issue.

### Language Support
*   Supports Marathi, Hindi, and English.
*   Translates other languages into Marathi, Hindi, or English in real-time.
*   Handles regional and rural pronunciation variations.

### Query Tracking
*   Tracks query history and previous interactions using a citizen’s query ID.
*   Enables personalized, context-aware assistance during recurring calls.

### RAG Implementation
*   Uses a Retrieval-Augmented Generation (RAG) approach for improved response generation.
*   Leverages TF-IDF for efficient document retrieval.
*   Utilizes Groq's LLM for generating contextually relevant and informative responses.

### Voice Chat
*   Allows users to ask questions using voice input.
*   Transcribes audio queries using Whisper.

### Download Chat History
*   Enables users to download their chat history in Excel format.