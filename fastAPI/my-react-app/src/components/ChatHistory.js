import React, { useState, useEffect, useRef } from 'react';
import { MessageCircle, User, Bot, Volume2, VolumeX, Pause, RotateCcw, Copy, CheckCircle, Loader2, Clock, Download, Play } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import * as XLSX from 'xlsx';
import './AnswerSection.css'; // Import animation

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

const ChatHistory = ({ chatHistory, onGenerateTTS }) => {
  const [playingIndex, setPlayingIndex] = useState(null);
  const [generatingTTSIndex, setGeneratingTTSIndex] = useState(null);
  const [copiedIndex, setCopiedIndex] = useState(null);
  const [currentAudio, setCurrentAudio] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [pausedAt, setPausedAt] = useState(0);
  const [audioProgress, setAudioProgress] = useState({});
  const [audioDuration, setAudioDuration] = useState({});
  const audioRefs = useRef({});
  const [audioUrls, setAudioUrls] = useState({});
  const playedOnce = useRef({}); // Add this near your other refs
  // Add autoPlayBlocked ref for each message
  const autoPlayBlocked = useRef({});

  // Global audio playing flag
  const isAnyAudioPlaying = () => window.isAnyAudioPlaying;
  const setAnyAudioPlaying = (val) => { window.isAnyAudioPlaying = val; };

  // Clean [Cached] text from answer
  const cleanAnswer = (text) => {
    if (typeof text === 'string') {
      return text.replace(/^\[Cached\]\s*/, '');
    }
    return text;
  };

  const handleCopyMessage = async (text, index) => {
    try {
      await navigator.clipboard.writeText(cleanAnswer(text));
      setCopiedIndex(index);
      setTimeout(() => setCopiedIndex(null), 2000);
    } catch (error) {
      console.error('Failed to copy text:', error);
    }
  };

  // Download chat history as Excel
  const handleDownloadExcel = () => {
    if (!chatHistory || chatHistory.length === 0) return;
    // Prepare data for Excel
    const data = chatHistory.map((chat, idx) => ({
      'S.No.': chatHistory.length - idx,
      'User': chat.user,
      'AI Assistant': chat.assistant,
      'Time': chat.timestamp
    }));
    const ws = XLSX.utils.json_to_sheet(data);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, 'ChatHistory');
    XLSX.writeFile(wb, 'chat_history.xlsx');
  };

  // When audioUrl changes for an index, create a new Audio object
  useEffect(() => {
    Object.entries(audioUrls).forEach(([idx, url]) => {
      if (!url) return;
      if (audioRefs.current[idx]) {
        audioRefs.current[idx].pause();
        audioRefs.current[idx] = null;
      }
      const audio = new Audio(url);
      audioRefs.current[idx] = audio;
      audio.addEventListener('timeupdate', () => {
        setAudioProgress(prev => ({ ...prev, [idx]: audio.currentTime }));
        setAudioDuration(prev => ({ ...prev, [idx]: audio.duration || 0 }));
      });
      audio.addEventListener('ended', () => setPlayingIndex(null));
      audio.addEventListener('pause', () => setPlayingIndex(null));
      audio.addEventListener('play', () => setPlayingIndex(idx));
      // Autoplay logic removed: audio will only play on user action
    });
    return () => {
      Object.values(audioRefs.current).forEach(audio => {
        if (audio) audio.pause();
      });
    };
  }, [audioUrls]);

  // Play/Pause handler
  const handlePlayPause = (idx) => {
    const audio = audioRefs.current[idx];
    if (!audio) return;
    if (audio.paused) {
      audio.play();
    } else {
      audio.pause();
    }
  };

  // Seek handler
  const handleSeek = (e, idx) => {
    const audio = audioRefs.current[idx];
    if (!audio || !audioDuration[idx]) return;
    const rect = e.target.getBoundingClientRect();
    const percent = (e.clientX - rect.left) / rect.width;
    const seekTime = percent * audioDuration[idx];
    audio.currentTime = seekTime;
    setAudioProgress(prev => ({ ...prev, [idx]: seekTime }));
    if (playingIndex === idx) audio.play();
  };

  // Generate TTS and set audioUrl for an index
  const handleGenerateTTS = async (text, idx) => {
    try {
      const cleanedText = cleanAnswer(text);
      const ttsResult = await onGenerateTTS(cleanedText);
      if (ttsResult && ttsResult.audio_base64) {
        const audioBlob = new Blob(
          [Uint8Array.from(atob(ttsResult.audio_base64), c => c.charCodeAt(0))],
          { type: 'audio/wav' }
        );
        const url = URL.createObjectURL(audioBlob);
        setAudioUrls(prev => ({ ...prev, [idx]: url }));
        setTimeout(() => {
          if (audioRefs.current[idx]) audioRefs.current[idx].play();
        }, 100);
      }
    } catch (error) {
      console.error('TTS generation failed:', error);
    }
  };

  // Stop all audio playback if event received
  useEffect(() => {
    const stopAll = () => {
      if (currentAudio) {
        currentAudio.pause();
        setPlayingIndex(null);
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

  if (!chatHistory || chatHistory.length === 0) {
    return (
      <div className="text-center py-12">
        <MessageCircle className="w-16 h-16 text-gray-300 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-500 mb-2">No Chat History</h3>
        <p className="text-gray-400">Your conversation history will appear here</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="mb-6 flex items-center justify-between">
        <div className="text-center flex-1">
          <h3 className="text-xl font-semibold text-gray-800 mb-2">Chat History</h3>
          <p className="text-gray-600">Previous conversations and responses</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleDownloadExcel}
            className="flex items-center gap-1 px-2 py-1 bg-green-600 text-white rounded shadow hover:bg-green-700 transition-colors text-xs"
            title="Download chat history as Excel"
            disabled={!chatHistory || chatHistory.length === 0}
          >
            <Download className="w-3 h-3 mr-1" />
            <span>Download</span>
          </button>
        </div>
      </div>

      {chatHistory.map((chat, index) => (
        <div key={index} className="space-y-2">
          {/* User Message Bubble */}
          <div className="flex w-full justify-end">
            <div className="bg-blue-50 text-blue-900 rounded-2xl px-5 py-3 shadow max-w-2xl text-right animate-fade-in">
              <span className="block font-semibold text-blue-700 mb-1">You</span>
              <span className="whitespace-pre-wrap break-words">{chat.user}</span>
              <div className="flex items-center justify-end space-x-2 mt-2">
                <span className="text-xs text-blue-600 flex items-center space-x-1">
                  <Clock className="w-3 h-3" />
                  <span>{chat.timestamp}</span>
                </span>
                <button
                  onClick={() => handleCopyMessage(chat.user, `user-${index}`)}
                  className="p-1 rounded hover:bg-blue-200 transition-colors"
                  title="Copy Message"
                >
                  {copiedIndex === `user-${index}` ? (
                    <CheckCircle className="w-3 h-3 text-green-600" />
                  ) : (
                    <Copy className="w-3 h-3 text-blue-600" />
                  )}
                </button>
              </div>
            </div>
          </div>
          {/* AI Assistant Message Bubble */}
          <div className="flex w-full justify-start">
            <div className="bg-green-50 text-green-900 rounded-2xl px-5 py-3 shadow max-w-2xl animate-fade-in relative">
              <span className="block font-semibold text-green-700 mb-1">AI Assistant</span>
              <div
                className="text-gray-700 leading-relaxed"
              >
                <div className="markdown-content" lang={detectLang(cleanAnswer(chat.assistant))}>
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
                    {cleanAnswer(chat.assistant)}
                  </ReactMarkdown>
                </div>
              </div>
              {/* Controls */}
              <div className="flex flex-col space-y-1 mt-2">
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => audioUrls[`assistant-${index}`] ? handlePlayPause(`assistant-${index}`) : handleGenerateTTS(cleanAnswer(chat.assistant), `assistant-${index}`)}
                    className="p-1 rounded hover:bg-green-200 transition-colors disabled:opacity-50"
                    title={playingIndex === `assistant-${index}` ? 'Pause Audio' : 'Play Audio'}
                  >
                    {playingIndex === `assistant-${index}` ? (
                      <Pause className="w-3 h-3 text-green-600" />
                    ) : (
                      <Play className="w-3 h-3 text-green-600" />
                    )}
                  </button>
                  <button
                    onClick={() => {
                      const audio = audioRefs.current[`assistant-${index}`];
                      if (audio) {
                        audio.currentTime = 0;
                        audio.play();
                      }
                    }}
                    disabled={!audioUrls[`assistant-${index}`]}
                    className="p-1 rounded hover:bg-purple-200 transition-colors disabled:opacity-50"
                    title="Replay Audio"
                  >
                    <RotateCcw className="w-3 h-3 text-purple-600" />
                  </button>
                  <button
                    onClick={() => handleCopyMessage(chat.assistant, `assistant-${index}`)}
                    className="p-1 rounded hover:bg-green-200 transition-colors"
                    title="Copy Message"
                  >
                    {copiedIndex === `assistant-${index}` ? (
                      <CheckCircle className="w-3 h-3 text-green-600" />
                    ) : (
                      <Copy className="w-3 h-3 text-green-600" />
                    )}
                  </button>
                </div>
                {/* Audio Player Bar */}
                {audioUrls[`assistant-${index}`] && (
                  <>
                    <div className="flex items-center space-x-2 w-full mt-1">
                      <span className="text-xs text-gray-500 w-10 text-right">
                        {new Date((audioProgress[`assistant-${index}`] || 0) * 1000).toISOString().substr(14, 5)}
                      </span>
                      <div
                        className="flex-1 h-2 bg-gray-200 rounded cursor-pointer relative"
                        onClick={e => handleSeek(e, `assistant-${index}`)}
                      >
                        <div
                          className="h-2 bg-green-400 rounded"
                          style={{ width: `${((audioProgress[`assistant-${index}`] || 0) / ((audioDuration[`assistant-${index}`] || 1))) * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-xs text-gray-500 w-10 text-left">
                        {audioDuration[`assistant-${index}`] ? new Date(audioDuration[`assistant-${index}`] * 1000).toISOString().substr(14, 5) : '00:00'}
                      </span>
                    </div>
                    {/* Download Audio Button */}
                    <div className="flex justify-end mt-1">
                      <a
                        href={audioUrls[`assistant-${index}`]}
                        download={`chat-audio-${index}.mp3`}
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
      ))}
    </div>
  );
};

export default ChatHistory;