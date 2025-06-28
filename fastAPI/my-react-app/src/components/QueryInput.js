import React, { useState, useRef } from 'react';
import { Send, Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import MicrophoneButton from './MicrophoneButton';

const QueryInput = ({
  onSubmit,
  isLoading,
  placeholderText = '',
  isRagBuilding,
  assistantReply,
}) => {
  const [inputText, setInputText] = useState('');
  const textareaRef = useRef(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if ((inputText || '').trim() && !isLoading) {
      onSubmit((inputText || '').trim());
      setInputText('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !isLoading) {
      e.preventDefault();
      if ((inputText || '').trim()) {
        onSubmit((inputText || '').trim());
        setInputText('');
      }
    }
  };

  const handleTranscription = (transcription) => {
    setInputText(transcription);
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  };

  return (
    <div className="flex flex-col items-center space-y-4">
      {/* Show loader when building RAG system */}
      {isRagBuilding && (
        <div className="flex items-center space-x-2 text-blue-600 text-lg font-semibold mb-2">
          <Loader2 className="w-6 h-6 animate-spin" />
          <span>Building RAG system, please wait...</span>
        </div>
      )}

      {/* Microphone button */}
      <div className="w-full flex justify-center mb-2">
        <MicrophoneButton 
          onTranscription={handleTranscription}
          disabled={isLoading || isRagBuilding}
        />
      </div>

      {/* Text input and send button below */}
      <form onSubmit={handleSubmit} className="w-full flex flex-col space-y-2">
        <textarea
          ref={textareaRef}
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholderText || 'Enter your question here...'}
          className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-base"
          rows="2"
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={!((inputText || '').trim()) || isLoading}
          className={`px-6 py-3 rounded-lg font-medium transition-all flex items-center space-x-2 w-full justify-center
            ${!((inputText || '').trim()) || isLoading
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-blue-600 text-white hover:bg-blue-700 shadow-lg hover:shadow-xl'}
          `}
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Processing...</span>
            </>
          ) : (
            <>
              <Send className="w-4 h-4" />
              <span>Send Question</span>
            </>
          )}
        </button>
      </form>

      {/* Only render assistant reply as markdown if present */}
      {assistantReply && (
        <div className="w-full max-w-xl mt-4 prose lg:prose-xl">
          <ReactMarkdown>{assistantReply}</ReactMarkdown>
        </div>
      )}
    </div>
  );
};

export default QueryInput;