This is the Demo application for our CMRF AI project. You can view the application [here](https://cmrf-rag-agent.streamlit.app/).
The knowledge base is used to provide information about various government schemes.

## Features

### Speech-to-Text Functionality
*   Accurately understands citizen queries during calls using Whisper.
*   Searches the knowledge base based on the query and suggests relevant solutions.
*   Recommends necessary next steps to resolve the citizen's issue.

### Language Support
*   Supports Marathi, Hindi, and English.
*   Translates other languages into Marathi, Hindi, or English in real-time.
*   Handles regional and rural pronunciation variations.

### Query Tracking
*   Tracks query history and previous interactions using a citizen’s query ID.
*   Enables personalized, context-aware assistance during recurring calls.

### RAG Implementation
*   Uses a Retrieval-Augmented Generation (RAG) approach for improved response generation.
*   Leverages TF-IDF for efficient document retrieval.
*   Utilizes Groq's LLM for generating contextually relevant and informative responses.

### Voice Chat
*   Allows users to ask questions using voice input.
*   Transcribes audio queries using Whisper.

### Download Chat History
*   Enables users to download their chat history in Excel format.

### Next Steps

*   Add CRM autofill to our POC
*   Train transformer model(RoBERT or BERT for error correction in audio transcriptions, especially Marathi and Hindi.)
