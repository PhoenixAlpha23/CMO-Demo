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
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const transcriptionTimeout = useRef(null);
  const silenceTimeoutRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const sourceRef = useRef(null);
  const wsRef = useRef(null); // <-- Add this line

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
      setIsTranscribing(true);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      // --- Silence detection setup ---
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      analyserRef.current = audioContextRef.current.createAnalyser();
      sourceRef.current = audioContextRef.current.createMediaStreamSource(stream);
      sourceRef.current.connect(analyserRef.current);

      const detectSilence = () => {
        const bufferLength = analyserRef.current.fftSize;
        const dataArray = new Uint8Array(bufferLength);
        analyserRef.current.getByteTimeDomainData(dataArray);
        // Calculate average volume
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {
          sum += Math.abs(dataArray[i] - 128);
        }
        const avg = sum / bufferLength;
        // If avg is below threshold, consider it silence
        if (avg < 5) {
          if (!silenceTimeoutRef.current) {
            silenceTimeoutRef.current = setTimeout(() => {
              stopRecording();
            }, 2000);
          }
        } else {
          if (silenceTimeoutRef.current) {
            clearTimeout(silenceTimeoutRef.current);
            silenceTimeoutRef.current = null;
          }
        }
        if (isRecording) {
          requestAnimationFrame(detectSilence);
        }
      };

      requestAnimationFrame(detectSilence);
      // --- End silence detection setup ---

      // Open WebSocket connection
      const socket = new window.WebSocket(WS_URL);
      wsRef.current = socket;

      socket.onopen = () => {
        setIsRecording(true);
        mediaRecorder.start(250); // send data every 250ms
      };

      socket.onmessage = (event) => {
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
            // Clean up audio context
            if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
              audioContextRef.current.close();
            }
          }
        } catch (err) {}
      };

      socket.onerror = (event) => {
        console.error('WebSocket error:', event);
        alert('WebSocket error during transcription');
        setIsTranscribing(false);
        setIsRecording(false);
        mediaRecorder.stop();
        stream.getTracks().forEach(track => track.stop());
        socket.close();
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
          audioContextRef.current.close();
        }
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
        // Clean up audio context
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
          audioContextRef.current.close();
        }
        if (silenceTimeoutRef.current) clearTimeout(silenceTimeoutRef.current);
      };

      transcriptionTimeout.current = setTimeout(() => {
        if (mediaRecorder.state !== 'inactive') {
          mediaRecorder.stop();
        }
        setIsTranscribing(false);
        setIsRecording(false);
        socket.close();
        stream.getTracks().forEach(track => track.stop());
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
          audioContextRef.current.close();
        }
      }, 20000); // 20 seconds max
    } catch (error) {
      alert('Microphone access denied or not available');
      setIsTranscribing(false);
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    if (transcriptionTimeout.current) clearTimeout(transcriptionTimeout.current);
    if (silenceTimeoutRef.current) clearTimeout(silenceTimeoutRef.current);

    // Stop mediaRecorder if active
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }

    // Stop WebSocket if open
    if (wsRef.current && wsRef.current.readyState === 1) {
      wsRef.current.send(JSON.stringify({ event: 'end' }));
      wsRef.current.close();
    }

    // Stop audio context
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
    }

    // Stop all audio tracks if any
    if (mediaRecorderRef.current && mediaRecorderRef.current.stream) {
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }

    setIsRecording(false);
    setIsTranscribing(false);
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
            ? (inputText ? inputText : 'Transcribing...')
            : isRecording
            ? (inputText ? inputText : 'Listening...')
            : (inputText || 'Type or use the mic to ask')}
        </div>
      </div>

      {/* Mic button - always enabled, changes color based on state */}
      <button
        type="button"
        onClick={handleMicClick}
        disabled={isTranscribing}
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