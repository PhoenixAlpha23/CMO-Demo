import axios from 'axios';

class ApiClient {
  constructor() {
    this.baseURL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: 120000, // 2 minutes timeout
    });
  }

  async healthCheck() {
    try {
      const response = await this.client.get('/health/');
      return response.status === 200;
    } catch (error) {
      return false;
    }
  }

  async uploadFiles(pdfFile, txtFile) {
    const formData = new FormData();
    if (pdfFile) {
      formData.append('pdf_file', pdfFile);
    }
    if (txtFile) {
      formData.append('txt_file', txtFile);
    }

    try {
      const response = await this.client.post('/upload/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      if (error.response?.data?.error) {
        throw new Error(error.response.data.error);
      }
      throw new Error('Upload failed');
    }
  }

  async query(inputText, modelKey, enhancedMode = true, voiceLangPref = 'auto') {
    try {
      const response = await this.client.post('/query/', {
        input_text: inputText,
        model_key: modelKey, // <-- Send model_key
        enhanced_mode: enhancedMode,
        voice_lang_pref: voiceLangPref,
      });
      return response.data;
    } catch (error) {
      if (error.response?.status === 429) {
        throw new Error('Rate limited. Please wait a moment before trying again.');
      }
      if (error.response?.data?.error) {
        throw new Error(error.response.data.error);
      }
      throw new Error('Query failed');
    }
  }

  async getChatHistory() {
    try {
      const response = await this.client.get('/chat-history/');
      return response.data.chat_history || [];
    } catch (error) {
      console.error('Failed to fetch chat history:', error);
      return [];
    }
  }

  async generateTTS(text, langPreference = 'auto') {
    const formData = new FormData();
    formData.append('text', text);
    formData.append('lang_preference', langPreference);

    try {
      const response = await this.client.post('/tts/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      if (error.response?.data?.error) {
        throw new Error(error.response.data.error);
      }
      throw new Error('TTS generation failed');
    }
  }

  async transcribeAudio(audioBlob) {
    const formData = new FormData();
    formData.append('audio_file', audioBlob, 'audio.wav');

    try {
      const response = await this.client.post('/transcribe/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      if (error.response?.data?.error) {
        throw new Error(error.response.data.error);
      }
      throw new Error('Transcription failed');
    }
  }
}

export default ApiClient;