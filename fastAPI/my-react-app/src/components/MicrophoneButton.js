import React, { useState, useRef, useEffect } from 'react';
import { Mic, MicOff, Loader2, AlertTriangle } from 'lucide-react';

const MicrophoneButton = ({ onTranscription, disabled = false }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [warning, setWarning] = useState('');
  const [animationScale, setAnimationScale] = useState(1);
  const mediaRecorderRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const animationFrameRef = useRef(null);
  const fallbackAnimationRef = useRef(null);
  const scaleAnimationRef = useRef(null);

  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (fallbackAnimationRef.current) {
        clearInterval(fallbackAnimationRef.current);
      }
      if (scaleAnimationRef.current) {
        clearInterval(scaleAnimationRef.current);
      }
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
        mediaRecorderRef.current.stop();
      }
    };
  }, []);

  // Fallback animation when audio level detection fails
  useEffect(() => {
    if (isRecording && !audioLevel) {
      fallbackAnimationRef.current = setInterval(() => {
        setAudioLevel(0.3 + Math.sin(Date.now() * 0.003) * 0.3);
      }, 50);
    } else {
      if (fallbackAnimationRef.current) {
        clearInterval(fallbackAnimationRef.current);
        fallbackAnimationRef.current = null;
      }
    }
  }, [isRecording, audioLevel]);

  // Simple scale animation for fallback - only when speaking
  useEffect(() => {
    if (isRecording && audioLevel > 0.15) {
      let scale = 1;
      let growing = true;
      scaleAnimationRef.current = setInterval(() => {
        if (growing) {
          scale += 0.02;
          if (scale >= 1.1) growing = false;
        } else {
          scale -= 0.02;
          if (scale <= 1) growing = true;
        }
        setAnimationScale(scale);
      }, 50);
    } else {
      setAnimationScale(1);
      if (scaleAnimationRef.current) {
        clearInterval(scaleAnimationRef.current);
        scaleAnimationRef.current = null;
      }
    }
  }, [isRecording, audioLevel]);

  const startRecording = async () => {
    setWarning('');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // Set up audio analysis for wave animation
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 256;
      analyserRef.current.smoothingTimeConstant = 0.8;
      source.connect(analyserRef.current);

      // Start wave animation with improved audio level detection
      const updateAudioLevel = () => {
        if (!analyserRef.current) return;
        
        try {
          const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
          analyserRef.current.getByteFrequencyData(dataArray);
          
          // Calculate average volume level with better sensitivity
          const average = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length;
          const normalizedLevel = Math.max(0.1, average / 255); // Minimum level for visibility
          setAudioLevel(normalizedLevel);
        } catch (error) {
          console.warn('Audio analysis error:', error);
          // Fallback to a breathing effect if audio analysis fails
          setAudioLevel(0.3 + Math.sin(Date.now() * 0.003) * 0.2);
        }
        
        animationFrameRef.current = requestAnimationFrame(updateAudioLevel);
      };
      updateAudioLevel();

      // Set up media recorder
      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      const chunks = [];
      mediaRecorderRef.current.ondataavailable = (event) => {
        chunks.push(event.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        setIsProcessing(true);
        try {
          const audioBlob = new Blob(chunks, { type: 'audio/webm' });
          
          // Import ApiClient dynamically to avoid circular dependencies
          const { default: ApiClient } = await import('../services/ApiClient');
          const apiClient = new ApiClient();
          
          const result = await apiClient.transcribeAudio(audioBlob);
          if (result.transcription) {
            onTranscription(result.transcription);
          } else {
            throw new Error('No transcription received');
          }
        } catch (error) {
          console.error('Transcription error:', error);
          
          // Provide more specific error messages
          let errorMessage = 'Failed to transcribe audio. Please try again.';
          
          if (error.message.includes('No speech detected')) {
            errorMessage = 'No speech detected. Please speak more clearly.';
          } else if (error.message.includes('language')) {
            errorMessage = 'Language not supported. Please speak in English, Hindi, or Marathi.';
          } else if (error.message.includes('Rate limited')) {
            errorMessage = 'Too many requests. Please wait a moment and try again.';
          } else if (error.message.includes('Failed to convert audio')) {
            errorMessage = 'Audio format not supported. Please try again.';
          }
          
          setWarning(errorMessage);
        } finally {
          setIsProcessing(false);
        }
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (error) {
      console.error('Error accessing microphone:', error);
      setWarning('Please allow microphone access to use voice input.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
    
    setIsRecording(false);
    setAudioLevel(0);
  };

  const handleClick = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  // Calculate glow properties based on audio level
  const glowSize = 80 + (audioLevel * 100); // Much larger size for better visibility
  const animationSpeed = 1 + (audioLevel * 1.5); // Chrome-style speed adjustment
  const isSpeaking = audioLevel > 0.15; // Threshold to determine if user is speaking

  return (
    <div className="flex flex-col items-center space-x-2">
      <div className="relative flex items-center justify-center">
        {/* Simple visible animation ring - guaranteed to work */}
        {isRecording && (
          <div
            className="absolute rounded-full pointer-events-none"
            style={{
              width: `${100 + (audioLevel * 80)}px`,
              height: `${100 + (audioLevel * 80)}px`,
              left: `calc(50% - ${(100 + (audioLevel * 80)) / 2}px)`,
              top: `calc(50% - ${(100 + (audioLevel * 80)) / 2}px)`,
              border: `4px solid rgba(59, 130, 246, ${0.8 + audioLevel * 0.2})`,
              borderRadius: '50%',
              zIndex: 20,
              transform: `scale(${isSpeaking ? animationScale : 1})`,
              transition: 'all 0.1s ease-out',
              boxShadow: `0 0 30px rgba(59, 130, 246, ${0.5 + audioLevel * 0.5})`
            }}
          />
        )}
        
        {/* Google Chrome-style ripple animations - only when speaking */}
        {isRecording && isSpeaking && (
          <>
            {/* Ripple 1 - Largest */}
            <div
              className="absolute rounded-full pointer-events-none"
              style={{
                width: `${glowSize}px`,
                height: `${glowSize}px`,
                left: `calc(50% - ${glowSize / 2}px)`,
                top: `calc(50% - ${glowSize / 2}px)`,
                background: `radial-gradient(circle, rgba(59, 130, 246, 1) 0%, rgba(59, 130, 246, 0.8) 30%, rgba(59, 130, 246, 0.5) 50%, rgba(59, 130, 246, 0.3) 70%, transparent 85%)`,
                boxShadow: '0 0 40px rgba(59, 130, 246, 0.6)',
                zIndex: 20,
                animation: `chromeRipple1 ${2 / animationSpeed}s ease-out infinite`
              }}
            />
            {/* Ripple 2 - Medium */}
            <div
              className="absolute rounded-full pointer-events-none"
              style={{
                width: `${glowSize * 0.8}px`,
                height: `${glowSize * 0.8}px`,
                left: `calc(50% - ${glowSize * 0.4}px)`,
                top: `calc(50% - ${glowSize * 0.4}px)`,
                background: `radial-gradient(circle, rgba(59, 130, 246, 0.9) 0%, rgba(59, 130, 246, 0.7) 40%, rgba(59, 130, 246, 0.4) 60%, rgba(59, 130, 246, 0.2) 80%, transparent 90%)`,
                boxShadow: '0 0 30px rgba(59, 130, 246, 0.5)',
                zIndex: 20,
                animation: `chromeRipple2 ${2 / animationSpeed}s ease-out infinite`,
                animationDelay: `${0.5 / animationSpeed}s`
              }}
            />
            {/* Ripple 3 - Smallest */}
            <div
              className="absolute rounded-full pointer-events-none"
              style={{
                width: `${glowSize * 0.6}px`,
                height: `${glowSize * 0.6}px`,
                left: `calc(50% - ${glowSize * 0.3}px)`,
                top: `calc(50% - ${glowSize * 0.3}px)`,
                background: `radial-gradient(circle, rgba(59, 130, 246, 0.8) 0%, rgba(59, 130, 246, 0.6) 50%, rgba(59, 130, 246, 0.3) 70%, rgba(59, 130, 246, 0.1) 85%, transparent 95%)`,
                boxShadow: '0 0 25px rgba(59, 130, 246, 0.4)',
                zIndex: 20,
                animation: `chromeRipple3 ${2 / animationSpeed}s ease-out infinite`,
                animationDelay: `${1 / animationSpeed}s`
              }}
            />
            {/* Chrome-style pulse core */}
            <div
              className="absolute rounded-full pointer-events-none"
              style={{
                width: `${glowSize * 0.4}px`,
                height: `${glowSize * 0.4}px`,
                left: `calc(50% - ${glowSize * 0.2}px)`,
                top: `calc(50% - ${glowSize * 0.2}px)`,
                background: `radial-gradient(circle, rgba(59, 130, 246, 1) 0%, rgba(59, 130, 246, 0.9) 30%, rgba(59, 130, 246, 0.7) 60%, rgba(59, 130, 246, 0.4) 80%, transparent 95%)`,
                boxShadow: '0 0 35px rgba(59, 130, 246, 0.8)',
                zIndex: 20,
                animation: `chromePulse ${1.5 / animationSpeed}s ease-in-out infinite`
              }}
            />
            {/* Recording indicator - Chrome red */}
            <div
              className="absolute rounded-full pointer-events-none"
              style={{
                width: `${glowSize * 0.25}px`,
                height: `${glowSize * 0.25}px`,
                left: `calc(50% - ${glowSize * 0.125}px)`,
                top: `calc(50% - ${glowSize * 0.125}px)`,
                background: `radial-gradient(circle, rgba(239, 68, 68, 1) 0%, rgba(239, 68, 68, 0.9) 40%, rgba(239, 68, 68, 0.6) 70%, transparent 90%)`,
                boxShadow: '0 0 20px rgba(239, 68, 68, 0.7)',
                zIndex: 20,
                animation: `recordingPulse ${0.8 / animationSpeed}s cubic-bezier(0.4, 0, 0.6, 1) infinite`
              }}
            />
          </>
        )}
        
        {/* Static recording indicator when not speaking */}
        {isRecording && !isSpeaking && (
          <div
            className="absolute rounded-full pointer-events-none"
            style={{
              width: `${glowSize * 0.25}px`,
              height: `${glowSize * 0.25}px`,
              left: `calc(50% - ${glowSize * 0.125}px)`,
              top: `calc(50% - ${glowSize * 0.125}px)`,
              background: `radial-gradient(circle, rgba(239, 68, 68, 1) 0%, rgba(239, 68, 68, 0.7) 50%, rgba(239, 68, 68, 0.3) 80%, transparent 95%)`,
              boxShadow: '0 0 15px rgba(239, 68, 68, 0.6)',
              zIndex: 20
            }}
          />
        )}
        <button
          onClick={handleClick}
          disabled={disabled || isProcessing}
          className={`relative p-5 rounded-full transition-all duration-200 shadow-lg focus:outline-none focus:ring-4 focus:ring-blue-300
            ${isRecording
              ? 'bg-red-500 hover:bg-red-600 text-white shadow-xl'
              : 'bg-blue-500 hover:bg-blue-600 text-white shadow-md hover:shadow-lg'}
            ${disabled || isProcessing ? 'opacity-50 cursor-not-allowed' : ''}`}
          style={{ 
            minWidth: '64px', 
            minHeight: '64px',
            zIndex: 30
          }}
          title={isRecording ? 'Stop recording' : 'Start voice input'}
        >
          {isProcessing ? (
            <Loader2 className="w-10 h-10 animate-spin" />
          ) : isRecording ? (
            <MicOff className="w-10 h-10" />
          ) : (
            <Mic className="w-10 h-10" />
          )}
        </button>
      </div>
      
      {warning && (
        <div className="mt-2 text-xs text-red-600 font-semibold text-center max-w-xs flex items-center justify-center">
          <AlertTriangle className="w-4 h-4 text-red-500" />
          <span className="ml-1">{warning}</span>
        </div>
      )}
    </div>
  );
};

export default MicrophoneButton; 