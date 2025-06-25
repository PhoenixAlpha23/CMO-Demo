import React, { useState, useRef } from 'react';
import { Send, Mic, MicOff, Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const QueryInput = ({
  onSubmit,
  onTranscribeAudio,
  isLoading,
  placeholderText = '',
  isRagBuilding,
  assistantReply, // <-- add this prop
}) => {
  const [inputText, setInputText] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [ws, setWs] = useState(null); // WebSocket instance
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  // WebSocket URL (update to your backend endpoint)
  const WS_URL = 'ws://localhost:8000/ws/transcribe';

  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputText.trim() && !isLoading) {
      onSubmit(inputText.trim());
      setInputText('');
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      // Open WebSocket connection
      const socket = new window.WebSocket(WS_URL);
      setWs(socket);
      setIsTranscribing(true);

      socket.onopen = () => {
        mediaRecorder.start(250); // send data every 250ms
        setIsRecording(true);
      };

      socket.onmessage = (event) => {
        // Expecting JSON: { partial: string, final: string }
        try {
          const data = JSON.parse(event.data);
          if (data.partial) {
            setInputText(data.partial);
          }
          if (data.final) {
            setInputText(data.final);
            setIsTranscribing(false);
            setIsRecording(false);
            mediaRecorder.stop();
            stream.getTracks().forEach(track => track.stop());
            socket.close();
          }
        } catch (err) {
          // Ignore parse errors
        }
      };

      socket.onerror = () => {
        alert('WebSocket error during transcription');
        setIsTranscribing(false);
        setIsRecording(false);
        mediaRecorder.stop();
        stream.getTracks().forEach(track => track.stop());
        socket.close();
      };

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0 && socket.readyState === 1) {
          socket.send(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        if (socket.readyState === 1) {
          socket.send(JSON.stringify({ event: 'end' }));
        }
      };
    } catch (error) {
      alert('Microphone access denied or not available');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (ws && ws.readyState === 1) {
        ws.send(JSON.stringify({ event: 'end' }));
        ws.close();
      }
    }
  };

  // Allow mic button to always stop recording/transcription
  const handleMicClick = () => {
    if (isRecording || isTranscribing) {
      stopRecording();
      setIsTranscribing(false);
      setIsRecording(false);
    } else {
      startRecording();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !isLoading && !isTranscribing) {
      e.preventDefault();
      if (inputText.trim()) {
        onSubmit(inputText.trim());
        setInputText('');
      }
    }
  };

  return (
    <div className="flex flex-col items-center space-y-4">
      {/* Show loader when building RAG system */}
      {isRagBuilding && (
        <div className="flex items-center space-x-2 text-blue-600 text-lg font-semibold mb-2">
          <Loader2 className="w-6 h-6 animate-spin" />
          <span>Building RAG system, please wait...</span>
        </div>
      )}

      {/* Speech bubble for transcription or status */}
      <div className="mb-2">
        <div className="rounded-xl shadow-lg px-6 py-3 bg-white text-gray-800 text-xl text-center min-w-[250px] min-h-[48px] flex items-center justify-center">
          {isTranscribing
            ? inputText || 'Transcribing...'
            : isRecording
            ? inputText || 'Listening...'
            : inputText || 'Type or use the mic to ask'}
        </div>
      </div>

      {/* Mic button - always enabled, changes color based on state */}
      <button
        type="button"
        onClick={handleMicClick}
        className={`w-16 h-16 flex items-center justify-center rounded-full shadow-lg transition-all border-4 focus:outline-none
          ${isTranscribing ? 'bg-blue-500 border-blue-700 animate-pulse' : isRecording ? 'bg-red-500 border-red-700 animate-pulse' : 'bg-blue-100 border-blue-300'}
          hover:scale-105 active:scale-95
        `}
        aria-label={isRecording || isTranscribing ? 'Stop recording' : 'Start recording'}
      >
        {isTranscribing ? (
          <Loader2 className="w-8 h-8 text-white animate-spin" />
        ) : isRecording ? (
          <MicOff className="w-8 h-8 text-white" />
        ) : (
          <Mic className="w-8 h-8 text-blue-600" />
        )}
      </button>

      {/* Text input and send button below mic */}
      <form onSubmit={handleSubmit} className="w-full max-w-xl flex flex-col items-center space-y-2 mt-4">
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholderText || 'Enter your question here...'}
          className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-base"
          rows="2"
          disabled={isLoading || isTranscribing}
        />
        <button
          type="submit"
          disabled={!inputText.trim() || isLoading || isTranscribing}
          className={`px-6 py-3 rounded-lg font-medium transition-all flex items-center space-x-2 w-full justify-center
            ${!inputText.trim() || isLoading || isTranscribing
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-blue-600 text-white hover:bg-blue-700 shadow-lg hover:shadow-xl'}
          `}
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Processing...</span>
            </>
          ) : (
            <>
              <Send className="w-4 h-4" />
              <span>Send Question</span>
            </>
          )}
        </button>
      </form>

      {/* Only render assistant reply as markdown if present */}
      {assistantReply && (
        <div className="w-full max-w-xl mt-4 prose lg:prose-xl">
          <ReactMarkdown>{assistantReply}</ReactMarkdown>
        </div>
      )}
    </div>
  );
};

export default QueryInput;