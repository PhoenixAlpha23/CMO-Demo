# CMRF AI Agent with Speech-to-Text

This project includes a FastAPI backend with speech-to-text capabilities and a React frontend with microphone input and wave animation.

## Features

- **Speech-to-Text**: Real-time audio transcription using OpenAI Whisper
- **Wave Animation**: Visual feedback during voice recording
- **RAG System**: Document-based question answering
- **TTS Support**: Text-to-speech capabilities
- **Modern UI**: React frontend with Tailwind CSS

## Setup Instructions

### Backend Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   REDIS_HOST=localhost
   REDIS_PORT=6379
   REDIS_DB=0
   REDIS_PASSWORD=
   ```

3. **Start the FastAPI backend:**
   ```bash
   uvicorn fastapp:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Setup

1. **Navigate to the React app:**
   ```bash
   cd my-react-app
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the React development server:**
   ```bash
   npm start
   ```

## Usage

### Speech-to-Text Feature

1. **Click the microphone button** in the QueryInput panel
2. **Allow microphone access** when prompted by your browser
3. **Speak your question** - you'll see wave animation during recording
4. **Click the microphone again** to stop recording
5. **Wait for transcription** - the text will appear in the input field
6. **Send the question** using the "Send Question" button

### Wave Animation

The wave animation provides visual feedback during recording:
- **Static bars**: Microphone is ready
- **Animated bars**: Recording in progress with real-time audio level visualization
- **Processing spinner**: Transcription is being processed

### API Endpoints

- `POST /transcribe/` - Transcribe audio file
- `POST /query/` - Send text query to RAG system
- `POST /upload/` - Upload documents for RAG
- `POST /tts/` - Generate text-to-speech audio
- `GET /health/` - Health check
- `GET /chat-history/` - Get chat history

## Technical Details

### Backend Architecture

- **FastAPI**: Modern Python web framework
- **OpenAI Whisper**: Speech-to-text transcription
- **Groq**: LLM for question answering
- **Redis**: Caching and session management (optional)
- **FAISS**: Vector similarity search

### Frontend Architecture

- **React**: UI framework
- **Tailwind CSS**: Styling
- **Web Audio API**: Real-time audio analysis
- **MediaRecorder API**: Audio recording
- **Axios**: HTTP client

### Audio Processing

1. **Recording**: Uses MediaRecorder API to capture audio
2. **Analysis**: Web Audio API provides real-time frequency data
3. **Visualization**: Wave animation based on audio levels
4. **Transmission**: Audio sent as WebM blob to backend
5. **Transcription**: OpenAI Whisper processes the audio
6. **Response**: Transcribed text returned to frontend

## Troubleshooting

### Microphone Issues

- **Permission denied**: Make sure to allow microphone access in your browser
- **No audio detected**: Check your microphone settings and try refreshing the page
- **Transcription fails**: Ensure the backend is running and accessible

### Backend Issues

- **Whisper model download**: First transcription may take time to download the model
- **Memory usage**: Whisper models can be large, ensure sufficient RAM
- **Audio format**: The system supports WebM and WAV formats

### Frontend Issues

- **Audio context**: Some browsers require HTTPS for audio recording
- **CORS**: Ensure the backend CORS settings allow your frontend domain
- **Browser compatibility**: Tested on Chrome, Firefox, and Safari

## Development

### Adding New Features

1. **Backend**: Add new endpoints in `fastapp.py`
2. **Frontend**: Create new components in `my-react-app/src/components/`
3. **API Client**: Update `ApiClient.js` for new endpoints
4. **Styling**: Use Tailwind CSS classes for consistent design

### Testing

- **Backend**: Use the `/health/` endpoint to verify API availability
- **Frontend**: Check browser console for any JavaScript errors
- **Audio**: Test with different audio input devices and volumes

## License

This project is part of the CMRF AI Agent system. 