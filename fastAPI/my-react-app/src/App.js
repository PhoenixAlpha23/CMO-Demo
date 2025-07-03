import React, { useState, useEffect, useCallback } from 'react';
import { X, Loader2 } from 'lucide-react';
import ApiClient from './services/ApiClient';
import FileUploader from './components/FileUploader';
import QueryInput from './components/QueryInput';
import ChatHistory from './components/ChatHistory';
import AnswerSection from './components/AnswerSection';
import StatusBar from './components/StatusBar';
import SidePanel from './components/SidePanel';

function App() {
  const [apiClient] = useState(new ApiClient());
  const [isApiAvailable, setIsApiAvailable] = useState(null);
  const [ragInitialized, setRagInitialized] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [currentAnswer, setCurrentAnswer] = useState('');
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [uploadedFiles, setUploadedFiles] = useState({ pdf: null, txt: null });
  const [activeTab, setActiveTab] = useState('upload');
  const [langWarning, setLangWarning] = useState('');
  const [isRagBuilding, setIsRagBuilding] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [modelKey, setModelKey] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const loadChatHistory = useCallback(async () => {
    try {
      const history = await apiClient.getChatHistory();
      setChatHistory(history);
    } catch (error) {
      console.error('Failed to load chat history:', error);
    }
  }, [apiClient]);

  const checkApiAvailability = useCallback(async () => {
    try {
      const available = await apiClient.healthCheck();
      setIsApiAvailable(available);
      if (available) {
        loadChatHistory();
      }
    } catch (error) {
      setIsApiAvailable(false);
    }
  }, [apiClient, loadChatHistory]);

  useEffect(() => {
    checkApiAvailability();
  }, [checkApiAvailability]);

  useEffect(() => {
    if (activeTab !== 'chat') {
      window.dispatchEvent(new Event('stopAllAudioPlayback'));
      window.isAnyAudioPlaying = false;
    }
  }, [activeTab]);

  const handleFileUpload = async (pdfFile, txtFile) => {
    setIsRagBuilding(true);
    try {
      const result = await apiClient.uploadFiles(pdfFile, txtFile);
      setUploadedFiles({ pdf: pdfFile, txt: txtFile });
      setRagInitialized(true);
      setActiveTab('chat');
      setModelKey(result.model_key);
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
    setCurrentQuestion(inputText); // Always set the question first!
    if (lang === 'other') {
      setLangWarning('⚠️ Please use only Marathi, Hindi, or English for your questions.');
      setCurrentAnswer('');
      return;
    } else {
      setLangWarning('');
    }
    setIsLoading(true);
    try {
      const textPromise = apiClient.query(inputText, modelKey);
      const ttsPromise = apiClient.generateTTS(inputText);
      const [result, ttsResult] = await Promise.all([textPromise, ttsPromise]);
      setCurrentAnswer(result.reply);

      // Prepare audio
      if (ttsResult && ttsResult.audio_base64) {
        if (audioUrl) {
          URL.revokeObjectURL(audioUrl);
        }
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
      setCurrentQuestion(inputText);
    } finally {
      setIsLoading(false);
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
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 flex flex-row">
      {/* Sidebar (always rendered, width changes) */}
      <div
        className={`transition-all duration-300 ease-in-out h-screen ${
          sidebarOpen ? 'w-[64px] min-w-[64px]' : 'w-0 min-w-0'
        }`}
        style={{ overflow: 'hidden', position: 'sticky', top: 0, left: 0, zIndex: 30 }}
      >
        <SidePanel
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          ragInitialized={ragInitialized}
          sidebarOpen={sidebarOpen}
          setSidebarOpen={setSidebarOpen}
        />
      </div>

      {/* Hamburger button, only when sidebar is closed */}
      {!sidebarOpen && (
        <button
          className="fixed top-4 left-4 z-50 bg-white border border-gray-300 rounded-full shadow p-2 hover:bg-blue-100 transition-all"
          onClick={() => setSidebarOpen(true)}
          title="Open sidebar"
          style={{
            width: 40,
            height: 40,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="5" y1="7" x2="19" y2="7" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            <line x1="5" y1="12" x2="19" y2="12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            <line x1="5" y1="17" x2="19" y2="17" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
        </button>
      )}

      {/* Main Content */}
      <div className="flex-1 flex flex-col transition-all duration-300 ease-in-out">
        {/* Header */}
        <header className="bg-white/80 backdrop-blur-md border-b border-white/20 relative">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative">
            <div className="absolute right-8 top-8">
              <StatusBar isApiAvailable={isApiAvailable} ragInitialized={ragInitialized} />
            </div>
            {/* Header Content */}
            <div className="flex items-center justify-between h-32">
              {/* Logo and Title - Center */}
              <div className="flex items-center space-x-4 absolute left-1/2 transform -translate-x-1/2">
                <img
                  src="/cmrf-logo-removebg-preview.png"
                  alt="CMRF Logo"
                  className="w-32 h-32 rounded-full object-cover"
                />
                <h1 className="text-6xl font-extrabold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  CMRF AI Agent
                </h1>
              </div>
              {/* Empty div for balance */}
              <div className="w-32"></div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 flex flex-col w-full">
          {/* Tab Content */}
          <div className="space-y-8 flex-1 flex flex-col justify-center">
            {activeTab === 'upload' && (
              <div className="flex items-center justify-center flex-1">
                <div className="bg-white/60 backdrop-blur-sm rounded-2xl shadow-xl p-10 w-full max-w-3xl">
                  <div className="text-center mb-8">
                    <h2 className="text-4xl font-extrabold text-gray-800 mb-4 flex items-center justify-center gap-3">
                      <span role="img" aria-label="file">📄</span>
                      कृपया योजनेचे तपशील फाइल अपलोड करा
                    </h2>
                    <p className="text-lg text-gray-700">
                      Upload PDF or TXT files to initialize the AI system
                    </p>
                  </div>
                  <FileUploader
                    onUpload={handleFileUpload}
                    isLoading={isLoading}
                    uploadedFiles={uploadedFiles}
                  />
                </div>
              </div>
            )}

            {activeTab === 'chat' && ragInitialized && (
              <div className="space-y-6">
                {/* Query Input */}
                <div className="bg-white/60 backdrop-blur-sm rounded-xl shadow-lg p-6">
                  <div className="relative">
                    <QueryInput
                      onSubmit={handleQuery}
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
                    autoPlay={true}
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
    </div>
  );
}

export default App;