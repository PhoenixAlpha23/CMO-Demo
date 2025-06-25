import React, { useState } from 'react';
import { Volume2, VolumeX, Copy, CheckCircle, Loader2 } from 'lucide-react';

const AnswerSection = ({ answer, onGenerateTTS }) => {
  const [isPlayingTTS, setIsPlayingTTS] = useState(false);
  const [isGeneratingTTS, setIsGeneratingTTS] = useState(false);
  const [isCopied, setIsCopied] = useState(false);
  const [currentAudio, setCurrentAudio] = useState(null);

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
    if (isPlayingTTS && currentAudio) {
      currentAudio.pause();
      setIsPlayingTTS(false);
      return;
    }

    setIsGeneratingTTS(true);
    try {
      const ttsResult = await onGenerateTTS(answer);
      if (ttsResult && ttsResult.audio_base64) {
        const audioBlob = new Blob(
          [Uint8Array.from(atob(ttsResult.audio_base64), c => c.charCodeAt(0))],
          { type: 'audio/wav' }
        );
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        
        audio.onplay = () => setIsPlayingTTS(true);
        audio.onended = () => {
          setIsPlayingTTS(false);
          URL.revokeObjectURL(audioUrl);
        };
        audio.onerror = () => {
          setIsPlayingTTS(false);
          URL.revokeObjectURL(audioUrl);
        };
        
        setCurrentAudio(audio);
        await audio.play();
      }
    } catch (error) {
      console.error('TTS generation failed:', error);
    } finally {
      setIsGeneratingTTS(false);
    }
  };

  if (!answer) return null;

  return (
    <div className="bg-white/60 backdrop-blur-sm rounded-xl shadow-lg p-6">
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
        <div className="text-gray-700 leading-relaxed whitespace-pre-wrap">
          {answer}
        </div>
      </div>
    </div>
  );
};

export default AnswerSection;
