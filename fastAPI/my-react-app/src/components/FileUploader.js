import React, { useState, useRef } from 'react';
import { Upload, File, FileText, X, CheckCircle, Loader2 } from 'lucide-react';

const FileUploader = ({ onUpload, isLoading, uploadedFiles }) => {
  const [dragActive, setDragActive] = useState(false);
  // Store only one file for pdf and txt
  const [files, setFiles] = useState({ pdf: null, txt: null });
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
    const result = await onUpload(files.pdf, files.txt);
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
    <div className="space-y-6">
      {/* Drag and Drop Area */}
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-all ${
          dragActive
            ? 'border-blue-500 bg-blue-50'
            : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        style={{ cursor: 'pointer' }}
      >
        <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
        <p className="text-lg font-medium text-gray-700 mb-2">
          Drag and drop your PDF or TXT files here
        </p>
        <p className="text-gray-500 mb-4">or click to select files</p>
        <div className="flex justify-center space-x-4">
          <button
            type="button"
            onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors flex items-center space-x-2"
          >
            <FileText className="w-4 h-4" />
            <span>Select PDF/TXT</span>
          </button>
        </div>
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

      {/* Selected Files */}
      {(files.pdf || files.txt) && (
        <div className="space-y-3">
          <h3 className="font-medium text-gray-700">Selected Files:</h3>
          {files.pdf && (
            <div className="flex items-center justify-between p-3 bg-red-50 rounded-lg border border-red-200">
              <div className="flex items-center space-x-3">
                <FileText className="w-5 h-5 text-red-600" />
                <span className="text-sm font-medium text-gray-700">{files.pdf.name}</span>
                <span className="text-xs text-gray-500">
                  ({(files.pdf.size / 1024 / 1024).toFixed(2)} MB)
                </span>
              </div>
              <button
                onClick={() => removeFile('pdf')}
                className="text-red-600 hover:text-red-800"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          )}
          {files.txt && (
            <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg border border-green-200">
              <div className="flex items-center space-x-3">
                <File className="w-5 h-5 text-green-600" />
                <span className="text-sm font-medium text-gray-700">{files.txt.name}</span>
                <span className="text-xs text-gray-500">
                  ({(files.txt.size / 1024).toFixed(2)} KB)
                </span>
              </div>
              <button
                onClick={() => removeFile('txt')}
                className="text-green-600 hover:text-green-800"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>
      )}

      {/* Upload Status */}
      {uploadedFiles.pdf || uploadedFiles.txt ? (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <CheckCircle className="w-5 h-5 text-green-600" />
            <span className="font-medium text-green-800">AI System Initialized Successfully!</span>
          </div>
          <p className="text-sm text-green-600 mt-1">
            You can now proceed to the Chat tab to start asking questions.
          </p>
        </div>
      ) : null}
    </div>
  );
};

export default FileUploader;