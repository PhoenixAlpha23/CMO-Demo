import React, { useState, useEffect, useRef } from 'react';
import { Volume2, VolumeX, Copy, CheckCircle, Loader2, Pause, RotateCcw, Play } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './AnswerSection.css';

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
  const [pausedAt, setPausedAt] = useState(0);
  const autoPlayBlocked = useRef(false);
  const lastPlayedAnswerRef = useRef(null);

  // Global audio playing flag
  const isAnyAudioPlaying = () => window.isAnyAudioPlaying;
  const setAnyAudioPlaying = (val) => { window.isAnyAudioPlaying = val; };

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
    // Always stop all other audio before playing
    window.dispatchEvent(new Event('stopAllAudioPlayback'));
    if (window.isAnyAudioPlaying) return; // Don't play if another audio is still flagged as playing
    if (currentAudio && !isPlayingTTS && !isGeneratingTTS && pausedAt > 0) {
      currentAudio.currentTime = pausedAt;
      await currentAudio.play();
      setIsPlayingTTS(true);
      setPausedAt(0);
      autoPlayBlocked.current = false;
      // window.isAnyAudioPlaying will be set in onplay
      return;
    }
    if (isPlayingTTS && currentAudio) {
      currentAudio.pause();
      setPausedAt(currentAudio.currentTime);
      setIsPlayingTTS(false);
      autoPlayBlocked.current = true;
      window.isAnyAudioPlaying = false;
      return;
    }
    if (currentAudio) {
      currentAudio.pause();
      setIsPlayingTTS(false);
      setCurrentAudio(null);
      setPausedAt(0);
      window.isAnyAudioPlaying = false;
    }
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
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
        audio.onplay = () => { setIsPlayingTTS(true); window.isAnyAudioPlaying = true; };
        audio.onended = () => {
          setIsPlayingTTS(false);
          setPausedAt(0);
          window.isAnyAudioPlaying = false;
        };
        audio.onpause = () => { window.isAnyAudioPlaying = false; };
        audio.onerror = () => {
          setIsPlayingTTS(false);
          setPausedAt(0);
          window.isAnyAudioPlaying = false;
          URL.revokeObjectURL(url);
        };
        setCurrentAudio(audio);
        setAudioUrl(url);
        setPausedAt(0);
        await audio.play();
        // window.isAnyAudioPlaying will be set in onplay
        autoPlayBlocked.current = false;
      }
    } catch (error) {
      console.error('TTS generation failed:', error);
      window.isAnyAudioPlaying = false;
    } finally {
      setIsGeneratingTTS(false);
    }
  };

  // Pause audio
  const handlePauseAudio = () => {
    if (currentAudio) {
      currentAudio.pause();
      setPausedAt(currentAudio.currentTime);
      setIsPlayingTTS(false);
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
      setPausedAt(0);
    }
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }
    autoPlayBlocked.current = false;
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
      window.dispatchEvent(new Event('stopAllAudioPlayback'));
      setTimeout(() => {
        if (!window.isAnyAudioPlaying) {
          lastPlayedAnswerRef.current = answer;
          handleTTSToggle();
        }
      }, 50); // Wait for event handler to run
    }
    // eslint-disable-next-line
  }, [answer]);

  // Stop all audio playback if event received
  useEffect(() => {
    const stopAll = () => {
      if (currentAudio) {
        currentAudio.pause();
        setIsPlayingTTS(false);
        setPausedAt(currentAudio.currentTime);
        window.isAnyAudioPlaying = false;
      }
    };
    window.addEventListener('stopAllAudioPlayback', stopAll);
    return () => {
      window.removeEventListener('stopAllAudioPlayback', stopAll);
      if (currentAudio) {
        currentAudio.pause();
        window.isAnyAudioPlaying = false;
      }
    };
  }, [currentAudio]);

  if (!answer) return null;

  return (
    <div className="flex flex-col items-end space-y-2 w-full">
      {/* User Question Bubble */}
      {question && (
        <div className="flex w-full justify-end">
          <div className="bg-blue-50 text-blue-900 rounded-2xl px-5 py-3 shadow max-w-2xl text-right animate-fade-in">
            <span className="block font-semibold text-blue-700 mb-1">You</span>
            <span className="whitespace-pre-wrap break-words">{question}</span>
          </div>
        </div>
      )}
      {/* AI Answer Bubble */}
      <div className="flex w-full justify-start">
        <div className="bg-green-50 text-green-900 rounded-2xl px-5 py-3 shadow max-w-2xl animate-fade-in relative">
          <span className="block font-semibold text-green-700 mb-1">AI Assistant</span>
          <div className="prose prose-blue max-w-none">
            <div className="text-gray-700 leading-relaxed">
              <div className="markdown-content" lang={detectLang(answer)}>
                <ReactMarkdown 
                  remarkPlugins={[remarkGfm]}
                  components={{
                    h1: ({node, ...props}) => <h1 className="text-2xl font-bold text-gray-800 mb-4 mt-8 first:mt-0" {...props} />,
                    h2: ({node, ...props}) => <h2 className="text-xl font-bold text-gray-800 mb-3 mt-6 first:mt-0" {...props} />,
                    h3: ({node, ...props}) => <h3 className="text-lg font-bold text-gray-800 mb-2 mt-5 first:mt-0" {...props} />,
                    h4: ({node, ...props}) => <h4 className="text-base font-bold text-gray-800 mb-2 mt-4 first:mt-0" {...props} />,
                    p: ({node, ...props}) => <p className="mb-4 leading-relaxed" {...props} />,
                    ul: ({node, ...props}) => <ul className="list-disc list-inside mb-6 space-y-2" {...props} />,
                    ol: ({node, ...props}) => <ol className="list-decimal list-inside mb-6 space-y-2" {...props} />,
                    li: ({node, ...props}) => <li className="mb-2 leading-relaxed" {...props} />,
                    strong: ({node, ...props}) => <strong className="font-bold text-gray-900" {...props} />,
                    em: ({node, ...props}) => <em className="italic" {...props} />,
                    blockquote: ({node, ...props}) => <blockquote className="border-l-4 border-blue-500 pl-4 italic text-gray-600 my-6" {...props} />,
                  }}
                >
                  {answer}
                </ReactMarkdown>
              </div>
            </div>
          </div>
          {/* Controls */}
          <div className="flex items-center space-x-2 mt-2">
            <button
              onClick={handleTTSToggle}
              disabled={isGeneratingTTS}
              className="p-2 rounded-lg bg-green-100 text-green-600 hover:bg-green-200 transition-colors disabled:opacity-50"
              title={isPlayingTTS ? 'Pause Audio' : 'Play Audio'}
            >
              {isGeneratingTTS ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : isPlayingTTS ? (
                <Pause className="w-4 h-4" />
              ) : (
                <Play className="w-4 h-4" />
              )}
            </button>
            <button
              onClick={handleReplayAudio}
              disabled={!audioUrl}
              className="p-2 rounded-lg bg-purple-100 text-purple-600 hover:bg-purple-200 transition-colors disabled:opacity-50"
              title="Replay Audio"
            >
              <RotateCcw className="w-4 h-4" />
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
      </div>
    </div>
  );
};

export default AnswerSection;
