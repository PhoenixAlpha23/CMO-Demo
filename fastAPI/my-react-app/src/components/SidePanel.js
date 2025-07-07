import React, { useEffect, useState } from 'react';
import { Home, MessageCircle, History, ChevronLeft } from 'lucide-react';

const SidePanel = ({ activeTab, setActiveTab, ragInitialized, sidebarOpen, setSidebarOpen }) => {
  const [show, setShow] = useState(false);

  useEffect(() => {
    // Trigger animation after mount
    const timer = setTimeout(() => setShow(true), 100);
    return () => clearTimeout(timer);
  }, []);

  return (
    <nav
      className={`relative h-screen flex flex-col bg-white/80 border-r border-gray-200 shadow-lg py-4 w-[64px] min-w-[64px] transition-all duration-700
        ${show ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}
      `}
    >
      {/* Close button, only when sidebar is open */}
      {sidebarOpen && (
        <button
          className="absolute top-4 right-2 z-10 bg-white border border-gray-300 rounded-full shadow p-1 hover:bg-blue-100 transition-all"
          onClick={() => setSidebarOpen(false)}
          title="Close sidebar"
          tabIndex={0}
        >
          <ChevronLeft className="w-5 h-5" />
        </button>
      )}

      <div className="flex flex-col items-center mt-16 space-y-2 px-0.5">
        <button
          className={`flex flex-col items-center w-full mb-2 p-4 rounded-xl transition-all
            ${activeTab === 'upload'
              ? 'bg-blue-600 text-white shadow mx-2'
              : 'text-gray-600 hover:bg-blue-100'}
          `}
          onClick={() => setActiveTab('upload')}
          title="Home"
        >
          <Home className="w-7 h-7" />
        </button>
        <button
          className={`flex flex-col items-center w-full mb-2 p-4 rounded-xl transition-all
            ${activeTab === 'chat' && ragInitialized
              ? 'bg-blue-600 text-white shadow mx-2'
              : ragInitialized
              ? 'text-gray-600 hover:bg-blue-100'
              : 'text-gray-400 cursor-not-allowed'}
          `}
          onClick={() => ragInitialized && setActiveTab('chat')}
          disabled={!ragInitialized}
          title="Chat"
        >
          <MessageCircle className="w-7 h-7" />
        </button>
        <button
          className={`flex flex-col items-center w-full p-4 rounded-xl transition-all
            ${activeTab === 'history' && ragInitialized
              ? 'bg-blue-600 text-white shadow mx-2'
              : ragInitialized
              ? 'text-gray-600 hover:bg-blue-100'
              : 'text-gray-400 cursor-not-allowed'}
          `}
          onClick={() => ragInitialized && setActiveTab('history')}
          disabled={!ragInitialized}
          title="History"
        >
          <History className="w-7 h-7" />
        </button>
      </div>
    </nav>
  );
};

export default SidePanel;

