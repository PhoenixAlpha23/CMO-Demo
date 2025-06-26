import React, { useState, useEffect } from 'react';
import { Bot, Upload, MessageCircle, Settings, History, Mic, Volume2, X, FileText, File, Loader2 } from 'lucide-react';
import ApiClient from './services/ApiClient';
import FileUploader from './components/FileUploader';
import QueryInput from './components/QueryInput';
import ChatHistory from './components/ChatHistory';
import AnswerSection from './components/AnswerSection';
import StatusBar from './components/StatusBar';

function App() {
  const [apiClient] = useState(new ApiClient());
  const [isApiAvailable, setIsApiAvailable] = useState(null);
  const [ragInitialized, setRagInitialized] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [currentAnswer, setCurrentAnswer] = useState('');
  const [currentQuestion, setCurrentQuestion] = useState(''); // <-- Add state for current question
  const [uploadedFiles, setUploadedFiles] = useState({ pdf: null, txt: null });
  const [activeTab, setActiveTab] = useState('upload');
  const [langWarning, setLangWarning] = useState('');
  const [isRagBuilding, setIsRagBuilding] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null); // Add this state at the top with other states

  useEffect(() => {
    checkApiAvailability();
  }, []);

  useEffect(() => {
    if (activeTab !== 'chat') {
      window.dispatchEvent(new Event('stopAllAudioPlayback'));
      window.isAnyAudioPlaying = false;
    }
  }, [activeTab]);

  const checkApiAvailability = async () => {
    try {
      const available = await apiClient.healthCheck();
      setIsApiAvailable(available);
      if (available) {
        loadChatHistory();
      }
    } catch (error) {
      setIsApiAvailable(false);
    }
  };

  const loadChatHistory = async () => {
    try {
      const history = await apiClient.getChatHistory();
      setChatHistory(history);
    } catch (error) {
      console.error('Failed to load chat history:', error);
    }
  };

  const handleFileUpload = async (pdfFile, txtFile) => {
    setIsRagBuilding(true);
    try {
      const result = await apiClient.uploadFiles(pdfFile, txtFile);
      setUploadedFiles({ pdf: pdfFile, txt: txtFile });
      setRagInitialized(true);
      setActiveTab('chat');
      // After upload, call your backend to build RAG system
      // await buildRagSystem(files); // your function
      return { success: true, message: result.message };
    } catch (error) {
      return { success: false, message: error.message };
    } finally {
      setIsRagBuilding(false);
    }
  };

  // Utility: detect language (very basic, for demo)
  function detectLanguage(text) {
    // Unicode ranges: Marathi (Devanagari), Hindi (Devanagari), English (A-Z, a-z)
    const marathiHindiRegex = /[\u0900-\u097F]/;
    const englishRegex = /[A-Za-z]/;
    if (marathiHindiRegex.test(text)) {
      // Could be Marathi or Hindi, allow
      return 'mr_hi';
    } else if (englishRegex.test(text)) {
      return 'en';
    } else if (text.trim() !== '') {
      return 'other';
    }
    return 'empty';
  }

  const handleQuery = async (inputText) => {
    if (!inputText.trim()) return;
    // Language check
    const lang = detectLanguage(inputText);
    if (lang === 'other') {
      setLangWarning('⚠️ Please use only Marathi, Hindi, or English for your questions.');
      return;
    } else {
      setLangWarning('');
    }
    setIsLoading(true);
    try {
      // Start both requests in parallel
      const textPromise = apiClient.query(inputText);
      const ttsPromise = apiClient.generateTTS(inputText); // or use textPromise's reply if needed
      const [result, ttsResult] = await Promise.all([textPromise, ttsPromise]);
      setCurrentAnswer(result.reply);
      setCurrentQuestion(inputText);

      // Prepare audio
      if (ttsResult && ttsResult.audio_base64) {
        const audioBlob = new Blob(
          [Uint8Array.from(atob(ttsResult.audio_base64), c => c.charCodeAt(0))],
          { type: 'audio/wav' }
        );
        const url = URL.createObjectURL(audioBlob);
        setAudioUrl(url);
      } else {
        setAudioUrl(null);
      }

      const newEntry = {
        user: inputText,
        assistant: result.reply,
        model: 'llama-3.3-70b-versatile',
        timestamp: new Date().toLocaleTimeString()
      };
      setChatHistory(prev => [newEntry, ...prev]);
    } catch (error) {
      setCurrentAnswer(`Error: ${error.message}`);
      setAudioUrl(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleTranscribeAudio = async (audioBlob) => {
    try {
      const result = await apiClient.transcribeAudio(audioBlob);
      return result.transcription;
    } catch (error) {
      throw new Error(`Transcription failed: ${error.message}`);
    }
  };

  const handleGenerateTTS = async (text, langPreference = 'auto') => {
    try {
      const result = await apiClient.generateTTS(text, langPreference);
      return result;
    } catch (error) {
      console.error('TTS generation failed:', error);
      return null;
    }
  };

  if (isApiAvailable === null) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4 text-blue-600" />
          <p className="text-gray-600">Checking API availability...</p>
        </div>
      </div>
    );
  }

  if (!isApiAvailable) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-red-50 to-pink-100 flex items-center justify-center">
        <div className="bg-white rounded-lg shadow-xl p-8 max-w-md text-center">
          <X className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-gray-800 mb-2">API Not Available</h2>
          <p className="text-gray-600 mb-4">
            FastAPI backend is not running. Please start the server at http://localhost:8000
          </p>
          <button
            onClick={checkApiAvailability}
            className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors"
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md border-b border-white/20 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative">
          <div className="absolute right-8 top-8">
            <StatusBar isApiAvailable={isApiAvailable} ragInitialized={ragInitialized} />
          </div>
          
          {/* Header Content */}
          <div className="flex items-center justify-between h-32">
            {/* Header Buttons - Left Side */}
            {(ragInitialized && (activeTab === 'chat' || activeTab === 'history')) && (
              <div className="flex items-center space-x-4">
                <button
                  onClick={() => setActiveTab('chat')}
                  disabled={!ragInitialized}
                  className={`p-3 rounded-full transition-all ${
                    activeTab === 'chat' && ragInitialized
                      ? 'bg-blue-600 text-white shadow-lg'
                      : ragInitialized
                      ? 'text-gray-600 hover:text-blue-600 hover:bg-white/50'
                      : 'text-gray-400 cursor-not-allowed'
                  }`}
                  title="Chat"
                >
                  <MessageCircle className="w-6 h-6" />
                </button>
                <button
                  onClick={() => setActiveTab('history')}
                  disabled={!ragInitialized}
                  className={`p-3 rounded-full transition-all ${
                    activeTab === 'history' && ragInitialized
                      ? 'bg-blue-600 text-white shadow-lg'
                      : ragInitialized
                      ? 'text-gray-600 hover:text-blue-600 hover:bg-white/50'
                      : 'text-gray-400 cursor-not-allowed'
                  }`}
                  title="History"
                >
                  <History className="w-6 h-6" />
                </button>
              </div>
            )}
            
            {/* Logo and Title - Center */}
            <div className="flex items-center space-x-4 absolute left-1/2 transform -translate-x-1/2">
              <img
                src="/cmrf-logo-removebg-preview.png"
                alt="CMRF Logo"
                className="w-24 h-24 rounded-full object-cover"
              />
              <h1 className="text-5xl font-extrabold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                CMRF AI Agent
              </h1>
            </div>
            
            {/* Empty div for balance */}
            <div className="w-32"></div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Tab Content */}
        <div className="space-y-8">
          {activeTab === 'upload' && (
            <div className="bg-white/60 backdrop-blur-sm rounded-xl shadow-lg p-8">
              <div className="text-center mb-8">
                <h2 className="text-3xl font-bold text-gray-800 mb-2">📄कृपया योजनेचे तपशील फाइल अपलोड करा</h2>
                <p className="text-gray-600">Upload PDF or TXT files to initialize the AI system</p>
              </div>
              <FileUploader
                onUpload={handleFileUpload}
                isLoading={isLoading}
                uploadedFiles={uploadedFiles}
              />
            </div>
          )}

          {activeTab === 'chat' && ragInitialized && (
            <div className="space-y-6">
              {/* Query Input */}
              <div className="bg-white/60 backdrop-blur-sm rounded-xl shadow-lg p-6">
                <div className="relative">
                  <QueryInput
                    onSubmit={handleQuery}
                    onTranscribeAudio={handleTranscribeAudio}
                    isLoading={isLoading}
                    enableEnterSubmit={true}
                    isRagBuilding={isRagBuilding}
                  />
                  {langWarning && (
                    <span className="absolute right-0 top-0 mt-2 mr-2 text-xs text-red-600 font-medium">
                      {langWarning}
                    </span>
                  )}
                </div>
              </div>

              {/* Current Answer */}
              {currentAnswer && (
                <AnswerSection
                  answer={currentAnswer}
                  question={currentQuestion}
                  onGenerateTTS={handleGenerateTTS}
                  audioUrl={audioUrl}
                />
              )}
            </div>
          )}

          {activeTab === 'history' && ragInitialized && (
            <div className="bg-white/60 backdrop-blur-sm rounded-xl shadow-lg p-6">
              <ChatHistory
                chatHistory={chatHistory}
                onGenerateTTS={handleGenerateTTS}
              />
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white/80 backdrop-blur-md border-t border-white/20 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center">
            <p className="text-gray-600">
              Powered by <span className="font-semibold text-blue-600">CMRF AI Agent</span>
            </p>
            <p className="text-sm text-gray-500 mt-2">
               Features: AI RAG System & 📝 TTS 🗣️ STT
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}


export default App;