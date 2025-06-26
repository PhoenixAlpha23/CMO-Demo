import React from 'react';
import { CheckCircle, XCircle, Clock, Wifi, WifiOff } from 'lucide-react';

const StatusBar = ({ isApiAvailable, ragInitialized }) => {
  return (
    <div className="flex items-center space-x-4">
      {/* API Status */}
      <div className="flex items-center space-x-2">
        {isApiAvailable ? (
          <>
            <Wifi className="w-4 h-4 text-green-500" />
            <span className="text-sm text-green-600 font-medium">API Connected</span>
          </>
        ) : (
          <>
            <WifiOff className="w-4 h-4 text-red-500" />
            <span className="text-sm text-red-600 font-medium">API Disconnected</span>
          </>
        )}
      </div>

      {/* RAG Status */}
      <div className="flex items-center space-x-2">
        {ragInitialized ? (
          <>
            <CheckCircle className="w-4 h-4 text-green-500" />
            <span className="text-sm text-green-600 font-medium">AI Ready</span>
          </>
        ) : (
          <>
            <Clock className="w-4 h-4 text-yellow-500" />
            <span className="text-sm text-yellow-600 font-medium">Upload Files</span>
          </>
        )}
      </div>
    </div>
  );
};

export default StatusBar;