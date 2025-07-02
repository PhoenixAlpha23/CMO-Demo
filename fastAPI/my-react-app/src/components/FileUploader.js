import React, { useState, useRef } from 'react';
import { Upload, File, FileText, X, CheckCircle, Loader2, Plus } from 'lucide-react';

const FileUploader = ({ onUpload, isLoading, uploadedFiles }) => {
  const [dragActive, setDragActive] = useState(false);
  // Store only one file for pdf and txt
  const [files, setFiles] = useState({ pdf: null, txt: null });
  const [isInitializing, setIsInitializing] = useState(false);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const droppedFiles = e.dataTransfer.files;
    if (droppedFiles && droppedFiles.length > 0) {
      let pdf = null;
      let txt = null;
      Array.from(droppedFiles).forEach((file) => {
        if (file.type === 'application/pdf' && !pdf) {
          pdf = file;
        } else if (file.type === 'text/plain' && !txt) {
          txt = file;
        }
      });
      setFiles(prev => ({
        pdf: pdf || prev.pdf,
        txt: txt || prev.txt
      }));
    }
  };

  const handleFileSelect = (e) => {
    const selectedFiles = e.target.files;
    if (selectedFiles && selectedFiles.length > 0) {
      let pdf = null;
      let txt = null;
      Array.from(selectedFiles).forEach((file) => {
        if (file.type === 'application/pdf' && !pdf) {
          pdf = file;
        } else if (file.type === 'text/plain' && !txt) {
          txt = file;
        }
      });
      setFiles(prev => ({
        pdf: pdf || prev.pdf,
        txt: txt || prev.txt
      }));
    }
  };

  const removeFile = (type) => {
    setFiles(prev => ({ ...prev, [type]: null }));
  };

  const handleUpload = async () => {
    if (!files.pdf && !files.txt) {
      alert('Please select at least one file to upload.');
      return;
    }
    setIsInitializing(true);
    const result = await onUpload(files.pdf, files.txt);
    setIsInitializing(false);
    if (result.success) {
      // Files uploaded successfully
    } else {
      alert(result.message);
    }
  };

  // Auto-initialize AI system after files are selected
  React.useEffect(() => {
    if ((files.pdf || files.txt) && !isLoading) {
      handleUpload();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [files.pdf, files.txt]);

  return (
    <div className="flex flex-col items-center justify-center w-full">
      {/* Compact Upload Bar */}
      <div
        className={`flex items-center gap-2 bg-white border border-gray-300 rounded-2xl shadow-sm px-4 py-2 max-w-lg w-full transition-all ${
          dragActive
            ? 'border-blue-500 bg-blue-50'
            : 'hover:border-blue-400 hover:bg-gray-50'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        style={{ cursor: 'pointer', minHeight: 48 }}
      >
        <Upload className="w-5 h-5 text-gray-400" />
        <span className="flex-1 text-sm text-gray-600 text-left">
          {files.pdf?.name || files.txt?.name
            ? `${files.pdf?.name || ''} ${files.txt?.name || ''}`.trim()
            : 'Select or drag PDF/TXT file'}
        </span>
        <button
          type="button"
          onClick={e => {
            e.stopPropagation();
            fileInputRef.current?.click();
          }}
          className="bg-blue-600 text-white px-2 py-1 rounded-lg hover:bg-blue-700 transition-colors flex items-center space-x-1 text-xs"
          title="Add file"
        >
          <Plus className="w-4 h-4" />
        </button>
        {(files.pdf || files.txt) && (
          <button
            onClick={e => {
              e.stopPropagation();
              setFiles({ pdf: null, txt: null });
            }}
            className="ml-2 text-gray-400 hover:text-red-500"
            title="Remove file"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,.txt"
        className="hidden"
        multiple
        onChange={handleFileSelect}
      />

      {/* Upload Status */}
      {uploadedFiles.pdf || uploadedFiles.txt ? (
        <div className="bg-green-50 border border-green-200 rounded-lg p-2 mt-2 w-full max-w-lg text-xs flex items-center gap-2">
          <CheckCircle className="w-4 h-4 text-green-600" />
          <span className="font-medium text-green-800">AI RAG System Initialized Successfully!</span>
        </div>
      ) : null}

      {/* Loading icon during AI initialization */}
      {isInitializing && (
        <div className="flex items-center justify-center mt-2 text-blue-600 text-xs font-semibold w-full max-w-lg">
          <Loader2 className="w-4 h-4 animate-spin mr-2" />
          <span>AI RAG System Initializing...</span>
        </div>
      )}
    </div>
  );
};

export default FileUploader;