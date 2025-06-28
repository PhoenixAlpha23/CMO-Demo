import React, { useState, useEffect, useRef } from 'react';
import { Volume2, VolumeX, Copy, CheckCircle, Loader2, Pause, RotateCcw, Play, Download } from 'lucide-react';
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

const langCodeToNameMap = {
  en: "English",
  hi: "Hindi",
  mr: "Marathi",
};

const AnswerSection = ({ answer, question, onGenerateTTS, audioUrl, autoPlay }) => {
  const [isPlayingTTS, setIsPlayingTTS] = useState(false);
  const [isGeneratingTTS, setIsGeneratingTTS] = useState(false);
  const [isCopied, setIsCopied] = useState(false);
  const [audioUrlState, setAudioUrl] = useState(null);
  const [audioProgress, setAudioProgress] = useState(0);
  const [audioDuration, setAudioDuration] = useState(0);
  const audioRef = useRef(null);

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

  // When audioUrl changes, create a new Audio object
  useEffect(() => {
    if (!audioUrlState) return;
    // Pause any global audio before starting new one
    if (window.currentlyPlayingAudio && typeof window.currentlyPlayingAudio.pause === 'function') {
      window.currentlyPlayingAudio.pause();
    }
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    const audio = new Audio(audioUrlState);
    audioRef.current = audio;
    window.currentlyPlayingAudio = audio;

    audio.addEventListener('timeupdate', () => {
      setAudioProgress(audio.currentTime);
      setAudioDuration(audio.duration || 0);
    });
    audio.addEventListener('ended', () => setIsPlayingTTS(false));
    audio.addEventListener('pause', () => setIsPlayingTTS(false));
    audio.addEventListener('play', () => setIsPlayingTTS(true));

    // Try to autoplay as soon as the audio is ready
    const tryPlay = () => {
      audio.load();
      const playPromise = audio.play();
      if (playPromise !== undefined) {
        playPromise.catch(() => {
          // Show a UI hint to the user and retry on interaction
          if (!window.__audio_autoplay_hint_shown) {
            alert('🔊 Please click anywhere on the page to enable audio playback (browser autoplay policy).');
            window.__audio_autoplay_hint_shown = true;
          }
          const onUserInteract = () => {
            audio.play();
            window.removeEventListener('click', onUserInteract);
            window.removeEventListener('keydown', onUserInteract);
          };
          window.addEventListener('click', onUserInteract);
          window.addEventListener('keydown', onUserInteract);
        });
      }
    };
    if (audio.readyState >= 1) {
      tryPlay();
    } else {
      audio.addEventListener('loadedmetadata', tryPlay, { once: true });
    }

    return () => {
      audio.pause();
      audioRef.current = null;
      if (window.currentlyPlayingAudio === audio) {
        window.currentlyPlayingAudio = null;
      }
    };
  }, [audioUrlState]);

  // Play/Pause handler
  const handlePlayPause = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) {
      audio.play();
    } else {
      audio.pause();
    }
  };

  // Seek handler
  const handleSeek = (e) => {
    const audio = audioRef.current;
    if (!audio || !audioDuration) return;
    const rect = e.target.getBoundingClientRect();
    const percent = (e.clientX - rect.left) / rect.width;
    const seekTime = percent * audioDuration;
    audio.currentTime = seekTime;
    setAudioProgress(seekTime);
    if (isPlayingTTS) audio.play();
  };

  // Generate TTS and set audioUrl
  const handleGenerateTTS = async () => {
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
        setAudioUrl(url);
        setTimeout(() => {
          if (audioRef.current) audioRef.current.play();
        }, 100);
      }
    } catch (error) {
      console.error('TTS generation failed:', error);
    } finally {
      setIsGeneratingTTS(false);
    }
  };

  // Cleanup on answer change
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    if (audioUrlState) {
      URL.revokeObjectURL(audioUrlState);
      setAudioUrl(null);
    }
    setAudioProgress(0);
    setAudioDuration(0);
    setIsPlayingTTS(false);
  }, [answer]);

  // Auto-generate and play TTS when answer changes and autoPlay is true
  useEffect(() => {
    if (autoPlay && answer && !audioUrlState && !isGeneratingTTS) {
      handleGenerateTTS();
    }
    // eslint-disable-next-line
  }, [answer, autoPlay]);

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
          <div className="flex flex-col space-y-1 mt-2">
            <div className="flex items-center space-x-2">
              <button
                onClick={audioUrlState ? handlePlayPause : handleGenerateTTS}
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
                onClick={() => {
                  const audio = audioRef.current;
                  if (audio) {
                    audio.currentTime = 0;
                    audio.play();
                  }
                }}
                disabled={!audioUrlState}
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
            {/* Audio Player Bar */}
            {audioUrlState && (
              <>
                <div className="flex items-center space-x-2 w-full">
                  <span className="text-xs text-gray-500 w-10 text-right">
                    {new Date(audioProgress * 1000).toISOString().substr(14, 5)}
                  </span>
                  <div
                    className="flex-1 h-2 bg-gray-200 rounded cursor-pointer relative"
                    onClick={handleSeek}
                  >
                    <div
                      className="h-2 bg-green-400 rounded"
                      style={{ width: `${(audioProgress / (audioDuration || 1)) * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-xs text-gray-500 w-10 text-left">
                    {audioDuration ? new Date(audioDuration * 1000).toISOString().substr(14, 5) : '00:00'}
                  </span>
                </div>
                {/* Download Audio Button */}
                <div className="flex justify-end mt-1">
                  <a
                    href={audioUrlState}
                    download="answer-audio.mp3"
                    className="inline-flex items-center px-3 py-1 bg-blue-100 text-blue-700 rounded hover:bg-blue-200 text-xs font-medium transition-colors"
                    title="Download Audio as MP3"
                  >
                    <Download className="w-4 h-4" />
                  </a>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnswerSection;
