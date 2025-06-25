import React, { useState } from 'react';
import { MessageCircle, User, Bot, Volume2, Copy, CheckCircle, Loader2, Clock, Download } from 'lucide-react';
import * as XLSX from 'xlsx';

const ChatHistory = ({ chatHistory, onGenerateTTS }) => {
  const [playingIndex, setPlayingIndex] = useState(null);
  const [generatingTTSIndex, setGeneratingTTSIndex] = useState(null);
  const [copiedIndex, setCopiedIndex] = useState(null);

  const handleCopyMessage = async (text, index) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedIndex(index);
      setTimeout(() => setCopiedIndex(null), 2000);
    } catch (error) {
      console.error('Failed to copy text:', error);
    }
  };

  const handleTTSPlay = async (text, index) => {
    if (playingIndex === index) {
      setPlayingIndex(null);
      return;
    }

    setGeneratingTTSIndex(index);
    try {
      const ttsResult = await onGenerateTTS(text);
      if (ttsResult && ttsResult.audio_base64) {
        const audioBlob = new Blob(
          [Uint8Array.from(atob(ttsResult.audio_base64), c => c.charCodeAt(0))],
          { type: 'audio/wav' }
        );
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        
        audio.onplay = () => setPlayingIndex(index);
        audio.onended = () => {
          setPlayingIndex(null);
          URL.revokeObjectURL(audioUrl);
        };
        audio.onerror = () => {
          setPlayingIndex(null);
          URL.revokeObjectURL(audioUrl);
        };
        
        await audio.play();
      }
    } catch (error) {
      console.error('TTS generation failed:', error);
    } finally {
      setGeneratingTTSIndex(null);
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

      <div className="space-y-6">
        {chatHistory.map((chat, index) => (
          <div key={index} className="space-y-4">
            {/* User Message */}
            <div className="flex items-start space-x-3">
              <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center flex-shrink-0">
                <User className="w-4 h-4 text-white" />
              </div>
              <div className="flex-1 bg-blue-50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-blue-800">You</span>
                  <div className="flex items-center space-x-2">
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
                <p className="text-blue-900 leading-relaxed">{chat.user}</p>
              </div>
            </div>

            {/* Assistant Message */}
            <div className="flex items-start space-x-3">
              <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center flex-shrink-0">
                <Bot className="w-4 h-4 text-white" />
              </div>
              <div className="flex-1 bg-green-50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-green-800">AI Assistant</span>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => handleTTSPlay(chat.assistant, `assistant-${index}`)}
                      disabled={generatingTTSIndex === `assistant-${index}`}
                      className="p-1 rounded hover:bg-green-200 transition-colors disabled:opacity-50"
                      title={playingIndex === `assistant-${index}` ? 'Stop Audio' : 'Play Audio'}
                    >
                      {generatingTTSIndex === `assistant-${index}` ? (
                        <Loader2 className="w-3 h-3 animate-spin text-green-600" />
                      ) : (
                        <Volume2 className="w-3 h-3 text-green-600" />
                      )}
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
                <div className="text-green-900 leading-relaxed whitespace-pre-wrap">
                  {chat.assistant}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ChatHistory;