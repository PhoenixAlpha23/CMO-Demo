import React, { useRef, useState } from 'react';

const languageOptions = [
  { label: 'English', code: 'en-US' },
  { label: 'Hindi', code: 'hi-IN' },
  { label: 'Marathi', code: 'mr-IN' },
];

const MicrophoneButton = ({ isRecording, setIsRecording, onTranscription, disabled }) => {
  const [selectedLanguage, setSelectedLanguage] = useState('en-US');
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
    recognition.lang = selectedLanguage;
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
    <div>
      <select
        value={selectedLanguage}
        onChange={e => setSelectedLanguage(e.target.value)}
        disabled={isRecording}
        style={{
          marginBottom: '1rem',
          padding: '0.5rem',
          fontSize: '1rem',
          background: 'transparent',
          border: 'none',
          color: '#222',
        }}
        aria-label="Select language"
      >
        {languageOptions.map(opt => (
          <option key={opt.code} value={opt.code}>{opt.label}</option>
        ))}
      </select>
      <button
        type="button"
        onClick={handleMicClick}
        disabled={disabled}
        className={`p-6 rounded-full flex items-center justify-center transition-all duration-200
          ${isRecording ? 'bg-red-500 animate-pulse scale-125' : 'bg-blue-500'}
          text-white`}
        style={{ fontSize: '2.5rem', width: '80px', height: '80px' }}
        aria-label={isRecording ? 'Stop Recording' : 'Start Recording'}
      >
        <span role="img" aria-label="microphone">
          🎤
        </span>
      </button>
    </div>
  );
};

export default MicrophoneButton;

