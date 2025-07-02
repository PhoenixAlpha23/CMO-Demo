import React, { useRef } from 'react';

const MicrophoneButton = ({ isRecording, setIsRecording, onTranscription, disabled, language = 'en-US' }) => {
  const recognitionRef = useRef(null);

  const handleMicClick = () => {
    if (isRecording) {
      setIsRecording(false);
      if (recognitionRef.current) recognitionRef.current.stop();
      return;
    }
    setIsRecording(true);

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert('SpeechRecognition not supported in this browser.');
      setIsRecording(false);
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = language; // Use selected language prop
    recognition.interimResults = true;
    recognition.continuous = true;
    recognitionRef.current = recognition;

    recognition.onresult = (event) => {
      let transcript = '';
      for (let i = 0; i < event.results.length; ++i) {
        transcript += event.results[i][0].transcript;
      }
      onTranscription(transcript);
      console.log('Transcript:', transcript);
    };
    recognition.onend = () => setIsRecording(false);
    recognition.onerror = () => setIsRecording(false);

    recognition.start();
  };

  React.useEffect(() => {
    return () => {
      if (recognitionRef.current) recognitionRef.current.stop();
    };
  }, []);

  return (
    <button
      type="button"
      onClick={handleMicClick}
      disabled={disabled}
      className={`p-6 rounded-full flex items-center justify-center transition-all duration-200
        ${isRecording ? 'bg-red-500 animate-pulse shadow-2xl scale-125' : 'bg-blue-500'}
        text-white`}
      style={{ fontSize: '2.5rem', width: '80px', height: '80px' }}
      aria-label={isRecording ? 'Stop Recording' : 'Start Recording'}
    >
      <span role="img" aria-label="microphone">
        🎤
      </span>
    </button>
  );
};

export default MicrophoneButton;

