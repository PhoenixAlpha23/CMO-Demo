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
  const [recordingTime, setRecordingTime] = useState(0);
  const recordingIntervalRef = useRef(null);
  const [errorMessage, setErrorMessage] = useState('');

  // Utility: get supported mimeType
  function getSupportedMimeType() {
    const types = ['audio/webm', 'audio/wav', 'audio/mp4'];
    for (const type of types) {
      if (MediaRecorder.isTypeSupported(type)) return type;
    }
    return '';
  }

  const handleSubmit = (e) => {
    e.preventDefault();
    if ((inputText || '').trim() && !isLoading) {
      onSubmit((inputText || '').trim());
      setInputText('');
    }
  };

  const startRecording = async () => {
    setErrorMessage('');
    try {
      setIsTranscribing(true);
      setRecordingTime(0);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = getSupportedMimeType();
      if (!mimeType) {
        alert('No supported audio recording format found in this browser.');
        setIsTranscribing(false);
        return;
      }
      const mediaRecorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      console.log('[Mic] Recording started with mimeType:', mimeType);

      // Timer
      recordingIntervalRef.current = setInterval(() => {
        setRecordingTime((t) => t + 1);
      }, 1000);

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
          console.log('[Mic] Data available, size:', event.data.size);
        }
      };

      mediaRecorder.onstop = async () => {
        clearInterval(recordingIntervalRef.current);
        setRecordingTime(0);
        // Clean up audio context
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
          audioContextRef.current.close();
        }
        if (silenceTimeoutRef.current) clearTimeout(silenceTimeoutRef.current);
        // Send audio to backend
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
        console.log('[Mic] Recording stopped. Blob size:', audioBlob.size, 'type:', audioBlob.type);
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.' + mimeType.split('/')[1]);
        try {
          const response = await fetch('/transcribe', {
            method: 'POST',
            body: formData,
          });
          if (response.ok) {
            const data = await response.json();
            if (data.transcription) {
              setInputText(data.transcription);
              setErrorMessage('');
            } else if (data.error) {
              setErrorMessage('Transcription error: ' + data.error);
              setInputText('');
              console.error('Backend error:', data.error);
            } else {
              setErrorMessage('Unknown backend response.');
              setInputText('');
              console.error('Unknown backend response:', data);
            }
            setIsTranscribing(false);
            setIsRecording(false);
          } else {
            const errorText = await response.text();
            setErrorMessage('Transcription failed: ' + errorText);
            setIsTranscribing(false);
            setIsRecording(false);
            console.error('Transcription failed. Response:', errorText);
          }
        } catch (err) {
          setErrorMessage('Transcription failed: ' + err.message);
          setIsTranscribing(false);
          setIsRecording(false);
          console.error('Error sending audio to backend:', err);
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
      // No silence detection: user must click mic button to stop
    } catch (error) {
      alert('Microphone access denied or not available');
      setIsTranscribing(false);
      setIsRecording(false);
      console.error('[Mic] Error accessing microphone:', error);
    }
  };

  const stopRecording = () => {
    if (transcriptionTimeout.current) clearTimeout(transcriptionTimeout.current);
    if (silenceTimeoutRef.current) clearTimeout(silenceTimeoutRef.current);
    clearInterval(recordingIntervalRef.current);
    setRecordingTime(0);
    // Stop mediaRecorder if active
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      console.log('[Mic] Stopping recording...');
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
    setErrorMessage('');
  };

  // Allow mic button to always stop recording/transcription
  const handleMicClick = () => {
    if (isRecording) {
      stopRecording();
    } else if (!isTranscribing) {
      startRecording();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !isLoading && !isTranscribing) {
      e.preventDefault();
      if ((inputText || '').trim()) {
        onSubmit((inputText || '').trim());
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
        className={`w-16 h-16 flex items-center justify-center rounded-full shadow-lg transition-all border-4 focus:outline-none
          ${isTranscribing ? 'bg-blue-500 border-blue-700 animate-pulse' : isRecording ? 'bg-red-500 border-red-700 animate-pulse' : 'bg-blue-100 border-blue-300'}
          hover:scale-105 active:scale-95
        `}
        aria-label={isRecording || isTranscribing ? 'Stop recording' : 'Start recording'}
      >
        {isTranscribing ? (
          <>
            <Loader2 className="w-8 h-8 text-white animate-spin" />
            <span className="ml-2 text-white font-bold">Stop</span>
          </>
        ) : isRecording ? (
          <>
            <MicOff className="w-8 h-8 text-white" />
            <span className="ml-2 text-white font-bold">Stop</span>
          </>
        ) : (
          <>
            <Mic className="w-8 h-8 text-blue-600" />
            <span className="ml-2 text-blue-600 font-bold">Start</span>
          </>
        )}
      </button>

      {/* Recording timer */}
      {isRecording && (
        <div className="text-red-600 font-bold text-lg mt-2">Recording: {recordingTime}s</div>
      )}

      {/* Show loader/message when backend is processing */}
      {isTranscribing && !isRecording && (
        <div className="text-blue-600 font-bold text-lg mt-2 flex items-center">
          <Loader2 className="w-5 h-5 animate-spin mr-2" />
          Processing audio...
        </div>
      )}

      {/* Show error message if transcription fails */}
      {errorMessage && (
        <div className="text-red-600 font-bold text-lg mt-2">{errorMessage}</div>
      )}

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
          disabled={!((inputText || '').trim()) || isLoading || isTranscribing}
          className={`px-6 py-3 rounded-lg font-medium transition-all flex items-center space-x-2 w-full justify-center
            ${!((inputText || '').trim()) || isLoading || isTranscribing
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