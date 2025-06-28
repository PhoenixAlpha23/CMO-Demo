# CMRF AI Agent Frontend

A React-based frontend for the CMRF AI Agent, featuring an intelligent RAG (Retrieval-Augmented Generation) system with Text-to-Speech (TTS) and Speech-to-Text (STT) support.

## Features

- **File Upload**: Support for PDF and TXT file uploads
- **Chat Interface**: Interactive chat with the AI agent
- **Multi-language Support**: Marathi, Hindi, and English
- **Text-to-Speech**: Audio generation for responses
- **Speech-to-Text**: Voice input capability
- **Chat History**: Persistent conversation history
- **Real-time Status**: API availability and RAG system status monitoring

## Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- FastAPI backend running on http://localhost:8000

## Installation

1. Install dependencies:
```bash
npm install
# or
yarn install
```

2. Start the development server:
```bash
npm start
# or
yarn start
```

The app will open at [http://localhost:3000](http://localhost:3000).

## Available Scripts

- `npm start` - Runs the app in development mode
- `npm run build` - Builds the app for production
- `npm test` - Launches the test runner
- `npm run eject` - Ejects from Create React App (one-way operation)

## Project Structure

```
src/
├── components/          # React components
│   ├── AnswerSection.js
│   ├── ChatHistory.js
│   ├── FileUploader.js
│   ├── MicrophoneButton.js
│   ├── QueryInput.js
│   └── StatusBar.js
├── services/           # API client and services
│   └── ApiClient.js
├── App.js             # Main application component
├── index.js           # Application entry point
└── index.css          # Global styles
```

## Technologies Used

- React 18
- Tailwind CSS
- Lucide React (icons)
- Axios (HTTP client)
- React Markdown
- XLSX (Excel file support)

## Backend Integration

This frontend connects to a FastAPI backend that provides:
- File upload and processing
- RAG system initialization
- AI query processing
- Text-to-Speech generation
- Chat history management

## License

This project is part of the CMRF AI Agent system.
