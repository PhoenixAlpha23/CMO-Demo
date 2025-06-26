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
  // Add autoPlayBlocked ref for each message
  const autoPlayBlocked = useRef({});

  // Global audio playing flag
  const isAnyAudioPlaying = () => window.isAnyAudioPlaying;
  const setAnyAudioPlaying = (val) => { window.isAnyAudioPlaying = val; };

  const handleCopyMessage = async (text, index) => {
    try {
      await navigator.clipboard.writeText(text);
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

  // Play or resume audio
  const handleTTSPlay = async (text, index) => {
    window.dispatchEvent(new Event('stopAllAudioPlayback'));
    if (window.isAnyAudioPlaying) return; // Don't play if another audio is still flagged as playing
    if (!generatingTTSIndex && playingIndex === index && currentAudio && currentAudio.paused && pausedAt > 0) {
      if (autoPlayBlocked.current[index]) return;
      currentAudio.currentTime = pausedAt;
      await currentAudio.play();
      setPausedAt(0);
      // window.isAnyAudioPlaying will be set in onplay
      return;
    }
    if (playingIndex === index && currentAudio && !currentAudio.paused) {
      currentAudio.pause();
      setPausedAt(currentAudio.currentTime);
      setPlayingIndex(null);
      autoPlayBlocked.current[index] = true;
      window.isAnyAudioPlaying = false;
      return;
    }
    if (currentAudio) {
      currentAudio.pause();
      setPlayingIndex(null);
      setCurrentAudio(null);
      setPausedAt(0);
      window.isAnyAudioPlaying = false;
    }
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }
    setGeneratingTTSIndex(index);
    try {
      const ttsResult = await onGenerateTTS(text);
      if (ttsResult && ttsResult.audio_base64) {
        const audioBlob = new Blob(
          [Uint8Array.from(atob(ttsResult.audio_base64), c => c.charCodeAt(0))],
          { type: 'audio/wav' }
        );
        const url = URL.createObjectURL(audioBlob);
        setAudioUrl(url);
        const audio = new Audio(url);
        audio.onplay = () => { setPlayingIndex(index); window.isAnyAudioPlaying = true; };
        audio.onpause = () => { setPausedAt(audio.currentTime); window.isAnyAudioPlaying = false; };
        audio.onended = () => {
          setPlayingIndex(null);
          setPausedAt(0);
          window.isAnyAudioPlaying = false;
          URL.revokeObjectURL(url);
        };
        audio.onerror = () => {
          setPlayingIndex(null);
          setPausedAt(0);
          window.isAnyAudioPlaying = false;
          URL.revokeObjectURL(url);
        };
        setCurrentAudio(audio);
        setPausedAt(0);
        await audio.play();
        // window.isAnyAudioPlaying will be set in onplay
        autoPlayBlocked.current[index] = false;
      }
    } catch (error) {
      console.error('TTS generation failed:', error);
      window.isAnyAudioPlaying = false;
    } finally {
      setGeneratingTTSIndex(null);
    }
  };

  // Pause audio (do not reset)
  const handleTTSPause = () => {
    if (currentAudio) {
      currentAudio.pause();
      setPausedAt(currentAudio.currentTime);
      setPlayingIndex(null);
    }
  };

  // Replay audio from start
  const handleTTSReplay = () => {
    if (audioUrl) {
      if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
      }
      const audio = new Audio(audioUrl);
      audio.onplay = () => setPlayingIndex(playingIndex);
      audio.onpause = () => setPausedAt(audio.currentTime);
      audio.onended = () => {
        setPlayingIndex(null);
        setPausedAt(0);
      };
      audio.onerror = () => {
        setPlayingIndex(null);
        setPausedAt(0);
      };
      setCurrentAudio(audio);
      setPausedAt(0);
      audio.play();
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
                <div className="markdown-content" lang={detectLang(chat.assistant)}>
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
                    {chat.assistant}
                  </ReactMarkdown>
                </div>
              </div>
              <div className="flex items-center space-x-2 mt-2">
                <button
                  onClick={() => handleTTSPlay(chat.assistant, `assistant-${index}`)}
                  disabled={generatingTTSIndex === `assistant-${index}`}
                  className="p-1 rounded hover:bg-green-200 transition-colors disabled:opacity-50"
                  title={
                    generatingTTSIndex === `assistant-${index}`
                      ? 'Loading Audio'
                      : playingIndex === `assistant-${index}` && pausedAt === 0
                      ? 'Pause Audio'
                      : pausedAt > 0 && playingIndex === `assistant-${index}`
                      ? 'Resume Audio'
                      : 'Play Audio'
                  }
                >
                  {generatingTTSIndex === `assistant-${index}` ? (
                    <Loader2 className="w-3 h-3 animate-spin text-green-600" />
                  ) : playingIndex === `assistant-${index}` && pausedAt === 0 ? (
                    <Pause className="w-3 h-3 text-green-600" />
                  ) : (
                    <Play className="w-3 h-3 text-green-600" />
                  )}
                </button>
                <button
                  onClick={handleTTSReplay}
                  disabled={!audioUrl}
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
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default ChatHistory;