import React, { useState, useEffect, useRef } from 'react';
import { Volume2, VolumeX, Copy, CheckCircle, Loader2, Pause, RotateCcw } from 'lucide-react';

// Simple bold markdown parser for **text**
function parseMarkdownBold(text) {
  return text.replace(/\*\*(.*?)\*\*/g, '<strong class="font-bold">$1</strong>');
}

// Simple language detector for Marathi, Hindi, English
function detectLang(text) {
  if (/[\u0900-\u097F]/.test(text)) {
    // Devanagari: could be Marathi or Hindi, default to Marathi if "च्या", else Hindi
    if (text.includes('च्या') || text.includes('आहे')) return 'mr';
    return 'hi';
  }
  if (/[A-Za-z]/.test(text)) return 'en';
  return 'en'; // fallback
}

const AnswerSection = ({ answer, question, onGenerateTTS }) => {
  const [isPlayingTTS, setIsPlayingTTS] = useState(false);
  const [isGeneratingTTS, setIsGeneratingTTS] = useState(false);
  const [isCopied, setIsCopied] = useState(false);
  const [currentAudio, setCurrentAudio] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const autoPlayBlocked = useRef(false);
  const lastPlayedAnswerRef = useRef(null); // <-- Add this line

  const handleCopyAnswer = async () => {
    try {
      await navigator.clipboard.writeText(answer);
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    } catch (error) {
      console.error('Failed to copy text:', error);
    }
  };

  const handleTTSToggle = async () => {
    // If audio is paused and exists, resume from where it was paused
    if (currentAudio && !isPlayingTTS && !isGeneratingTTS) {
      await currentAudio.play();
      setIsPlayingTTS(true);
      return;
    }

    // If already playing, treat as stop (pause, but don't reset)
    if (isPlayingTTS && currentAudio) {
      currentAudio.pause();
      setIsPlayingTTS(false);
      autoPlayBlocked.current = true; // Block auto-play for this answer
      return;
    }

    // If generating, don't start again
    if (isGeneratingTTS) return;

    // Always stop and clean up any existing audio before starting new
    if (currentAudio) {
      currentAudio.pause();
      setIsPlayingTTS(false);
      setCurrentAudio(null);
    }

    setIsGeneratingTTS(true);
    try {
      const lang = detectLang(answer);
      const ttsResult = await onGenerateTTS(answer, lang);
      if (ttsResult && ttsResult.audio_base64) {
        const audioBlob = new Blob(
          [Uint8Array.from(atob(ttsResult.audio_base64), c => c.charCodeAt(0))],
          { type: 'audio/wav' }
        );
        const url = URL.createObjectURL(audioBlob);
        const audio = new Audio(url);

        audio.onplay = () => setIsPlayingTTS(true);
        audio.onended = () => {
          setIsPlayingTTS(false);
        };
        audio.onerror = () => {
          setIsPlayingTTS(false);
          URL.revokeObjectURL(url);
        };

        setCurrentAudio(audio);
        setAudioUrl(url);
        await audio.play();
      }
    } catch (error) {
      console.error('TTS generation failed:', error);
    } finally {
      setIsGeneratingTTS(false);
    }
  };

  // Pause audio
  const handlePauseAudio = () => {
    if (currentAudio) {
      currentAudio.pause();
      setIsPlayingTTS(false);
      // Do NOT reset currentTime here
    }
  };

  // Replay audio from start
  const handleReplayAudio = () => {
    if (audioUrl) {
      if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
      }
      const audio = new Audio(audioUrl);
      audio.onplay = () => setIsPlayingTTS(true);
      audio.onended = () => setIsPlayingTTS(false);
      audio.onerror = () => setIsPlayingTTS(false);
      setCurrentAudio(audio);
      audio.play();
    }
  };

  // Cleanup old audio when answer changes
  useEffect(() => {
    if (currentAudio) {
      currentAudio.pause();
      setIsPlayingTTS(false);
      setCurrentAudio(null);
    }
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }
    // Reset auto-play block for new answer
    autoPlayBlocked.current = false;
    // Reset last played answer
    lastPlayedAnswerRef.current = null;
    // eslint-disable-next-line
  }, [answer]);

  // Auto-play TTS when a new answer is generated, unless blocked
  useEffect(() => {
    if (
      answer &&
      !autoPlayBlocked.current &&
      lastPlayedAnswerRef.current !== answer
    ) {
      lastPlayedAnswerRef.current = answer;
      handleTTSToggle();
    }
    // eslint-disable-next-line
  }, [answer]);

  if (!answer) return null;

  return (
    <div className="bg-white/60 backdrop-blur-sm rounded-xl shadow-lg p-6">
      {/* Show the user's question */}
      {question && (
        <div className="mb-2">
          <span className="block text-gray-800 text-base font-bold whitespace-pre-wrap">
            You Asked: {question}
          </span>
        </div>
      )}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-800 flex items-center space-x-2">
          <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
            <span className="text-white text-sm font-bold">AI</span>
          </div>
          <span>Answer</span>
        </h3>
        <div className="flex items-center space-x-2">
          <button
            onClick={handleTTSToggle}
            disabled={isGeneratingTTS}
            className="p-2 rounded-lg bg-green-100 text-green-600 hover:bg-green-200 transition-colors disabled:opacity-50"
            title={isPlayingTTS ? 'Stop Audio' : 'Play Audio'}
          >
            {isGeneratingTTS ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : isPlayingTTS ? (
              <VolumeX className="w-4 h-4" />
            ) : (
              <Volume2 className="w-4 h-4" />
            )}
          </button>
          {/* Pause Button */}
          <button
            onClick={handlePauseAudio}
            disabled={!isPlayingTTS}
            className="p-2 rounded-lg bg-yellow-100 text-yellow-600 hover:bg-yellow-200 transition-colors disabled:opacity-50"
            title="Pause Audio"
          >
            <Pause className="w-4 h-4" />
          </button>
          {/* Replay Button */}
          <button
            onClick={handleReplayAudio}
            disabled={!audioUrl}
            className="p-2 rounded-lg bg-purple-100 text-purple-600 hover:bg-purple-200 transition-colors disabled:opacity-50"
            title="Replay Audio"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
          {/* ...existing copy button... */}
          <button
            onClick={handleCopyAnswer}
            className="p-2 rounded-lg bg-blue-100 text-blue-600 hover:bg-blue-200 transition-colors"
            title="Copy Answer"
          >
            {isCopied ? (
              <CheckCircle className="w-4 h-4" />
            ) : (
              <Copy className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>
      
      <div className="prose prose-blue max-w-none">
        <div className="text-gray-700 leading-relaxed whitespace-pre-wrap"
          dangerouslySetInnerHTML={{ __html: parseMarkdownBold(answer) }}
        />
      </div>
    </div>
  );
};

export default AnswerSection;
