import React, { useState, useRef } from 'react';
import { Volume2, VolumeX, Pause, RotateCcw, Copy, CheckCircle, Loader2 } from 'lucide-react';

function parseMarkdownBold(text) {
  if (!text) return '';
  return text.replace(/\*\*(.*?)\*\*/g, '<strong class="font-bold">$1</strong>');
}

// Simple language detector (copy from AnswerSection)
function detectLang(text) {
  if (/[\u0900-\u097F]/.test(text)) {
    if (text.includes('च्या') || text.includes('आहे')) return 'mr';
    return 'hi';
  }
  if (/[A-Za-z]/.test(text)) return 'en';
  return 'en';
}

const ChatHistory = ({ chatHistory, onGenerateTTS }) => {
  // Transform backend messages to frontend format
  const flatHistory = [];
  chatHistory.forEach(msg => {
    if (msg.user) {
      flatHistory.push({ role: 'user', content: msg.user });
    }
    if (msg.assistant) {
      flatHistory.push({ role: 'assistant', content: msg.assistant });
    }
  });

  const [playingIndex, setPlayingIndex] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [currentAudio, setCurrentAudio] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [pausedAt, setPausedAt] = useState(0);
  const [copiedIndex, setCopiedIndex] = useState(null);

  // Play or resume audio
  const handlePlay = async (answer, idx) => {
    // If already playing this, stop
    if (isPlaying && playingIndex === idx && currentAudio) {
      currentAudio.pause();
      setPausedAt(currentAudio.currentTime);
      setIsPlaying(false);
      setPlayingIndex(null);
      return;
    }

    // If paused and same index, resume
    if (!isPlaying && playingIndex === idx && currentAudio) {
      currentAudio.currentTime = pausedAt;
      await currentAudio.play();
      setIsPlaying(true);
      return;
    }

    // Stop any currently playing audio
    if (currentAudio) {
      currentAudio.pause();
      setIsPlaying(false);
      setCurrentAudio(null);
      setPlayingIndex(null);
      setPausedAt(0);
    }
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }
    setIsGenerating(true);
    try {
      const lang = detectLang(answer);
      const ttsResult = await onGenerateTTS(answer, lang);
      if (ttsResult && ttsResult.audio_base64) {
        const audioBlob = new Blob(
          [Uint8Array.from(atob(ttsResult.audio_base64), c => c.charCodeAt(0))],
          { type: 'audio/wav' }
        );
        const url = URL.createObjectURL(audioBlob);
        setAudioUrl(url);
        const audio = new Audio(url);
        audio.onplay = () => {
          setIsPlaying(true);
          setPlayingIndex(idx);
        };
        audio.onpause = () => {
          setPausedAt(audio.currentTime);
        };
        audio.onended = () => {
          setIsPlaying(false);
          setPlayingIndex(null);
          setPausedAt(0);
        };
        audio.onerror = () => {
          setIsPlaying(false);
          setPlayingIndex(null);
          setPausedAt(0);
        };
        setCurrentAudio(audio);
        setPausedAt(0);
        await audio.play();
      }
    } catch (e) {
      setIsPlaying(false);
      setPlayingIndex(null);
      setPausedAt(0);
    } finally {
      setIsGenerating(false);
    }
  };

  // Pause audio (do not reset)
  const handlePause = () => {
    if (currentAudio) {
      currentAudio.pause();
      setIsPlaying(false);
      // pausedAt is set in onpause
    }
  };

  // Replay audio from start
  const handleReplay = () => {
    if (audioUrl) {
      if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
      }
      const audio = new Audio(audioUrl);
      audio.onplay = () => {
        setIsPlaying(true);
      };
      audio.onended = () => {
        setIsPlaying(false);
        setPlayingIndex(null);
        setPausedAt(0);
      };
      audio.onerror = () => {
        setIsPlaying(false);
        setPlayingIndex(null);
        setPausedAt(0);
      };
      setCurrentAudio(audio);
      setPausedAt(0);
      audio.play();
    }
  };

  const handleCopyMessage = async (text, index) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedIndex(index);
      setTimeout(() => setCopiedIndex(null), 2000);
    } catch (error) {
      console.error('Failed to copy text:', error);
    }
  };

  return (
    <div>
      {flatHistory.map((msg, idx) => (
        <div key={idx} className="mb-4 p-4 rounded bg-gray-50">
          <div className="mb-2 font-bold">{msg.role === 'user' ? 'You' : 'AI'}</div>
          {msg.content ? (
            <div
              className="mb-2 text-gray-800"
              dangerouslySetInnerHTML={{ __html: parseMarkdownBold(msg.content) }}
            />
          ) : (
            <div className="mb-2 text-gray-400 italic">No content</div>
          )}
          {msg.role === 'assistant' && (
            <div className="flex space-x-2">
              {/* Play/Pause Button */}
              <button
                onClick={() => handlePlay(msg.content, idx)}
                disabled={isGenerating && playingIndex === idx}
                className="p-2 rounded bg-green-100 text-green-600 hover:bg-green-200 disabled:opacity-50"
                title={
                  isPlaying && playingIndex === idx
                    ? 'Stop Audio'
                    : pausedAt > 0 && playingIndex === idx
                    ? 'Resume Audio'
                    : 'Play Audio'
                }
              >
                {isGenerating && playingIndex === idx ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : isPlaying && playingIndex === idx ? (
                  <VolumeX className="w-4 h-4" />
                ) : pausedAt > 0 && playingIndex === idx ? (
                  <Pause className="w-4 h-4" />
                ) : (
                  <Volume2 className="w-4 h-4" />
                )}
              </button>
              {/* Pause Button */}
              <button
                onClick={handlePause}
                disabled={!isPlaying || playingIndex !== idx}
                className="p-2 rounded bg-yellow-100 text-yellow-600 hover:bg-yellow-200 disabled:opacity-50"
                title="Pause Audio"
              >
                <Pause className="w-4 h-4" />
              </button>
              {/* Replay Button */}
              <button
                onClick={handleReplay}
                disabled={!audioUrl || playingIndex !== idx}
                className="p-2 rounded bg-purple-100 text-purple-600 hover:bg-purple-200 disabled:opacity-50"
                title="Replay Audio"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
              {/* Copy Button */}
              <button
                onClick={() => handleCopyMessage(msg.content, idx)}
                className="p-2 rounded bg-blue-100 text-blue-600 hover:bg-blue-200 transition-colors"
                title="Copy Answer"
              >
                {copiedIndex === idx ? (
                  <CheckCircle className="w-4 h-4" />
                ) : (
                  <Copy className="w-4 h-4" />
                )}
              </button>
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

export default ChatHistory;