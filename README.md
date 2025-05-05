# Sleep Assistant Projects

This repository contains two innovative projects focused on improving sleep quality through AI-powered assistance:

## Projects Overview

1. **RAG Project** (Task 1): A retrieval-augmented generation system for sleep advice
   - Option of using FAISS or Pinecone for efficient similarity search
   - Implements Gradio for user interface
   - Supports both OpenAI and Huggingface embeddings
   - Includes pre-built FAISS indices for quick deployment

2. **Voice Project** (Task 2): A voice-based interface for sleep coaching
   - Implements voice-to-text and text-to-voice capabilities
   - Uses OpenAI's Whisper for speech recognition
   - Features a caching database for storing conversation history
   - Implements Gradio for user interface

## Project Structure

### RAG Project
```
rag_project/
├── main.py               # Main application entry point
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── utils/                # Utility functions and helpers
├── faiss_index_openai/   # Pre-built FAISS index for OpenAI embeddings
├── faiss_index_generic/  # Pre-built FAISS index for generic embeddings
├── datasets/             # Training and test datasets
└── .gradio/              # Gradio configuration and cache
```

### Voice Project
```
voice_project/
├── main.py               # Main application entry point
├── alternate_main.py     # Alternative implementation
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── utils/                # Utility functions and helpers
├── db/                   # Database files and configurations
└── datasets/             # Training and test datasets
```

## Prerequisites

- Python 3.10 or higher
- Git
- Docker (recommended)
- OpenAI API key (for both projects)
- Pinecone API key (for rag project)
- Sufficient disk space for FAISS indices and datasets

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yash2002vardhan/naptick_assignment.git
   cd naptick_assignment
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies for each project:
   ```bash
   # For RAG Project
   cd rag_project
   pip install -r requirements.txt

   # For Voice Project
   cd ../voice_project
   pip install -r requirements.txt
   ```

## Environment Variables Setup

Create a `.env` file in each project directory with the following variables:

### RAG Project (.env)
```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

### Voice Project (.env)
```
OPENAI_API_KEY=your_openai_api_key
```

## Running the Projects

### Method 1: Using Python directly

1. For RAG Project:
   ```bash
   cd rag_project
   python main.py
   ```
   The application will start a Gradio interface accessible at `http://localhost:7860`

2. For Voice Project:
   ```bash
   cd voice_project
   python main.py
   ```
   The application will start a Gradio server accessible at `http://localhost:7860`

### Method 2: Using Docker (Recommended)

1. For RAG Project:
   ```bash
   cd rag_project
   docker compose up --build
   ```

2. For Voice Project:
   ```bash
   cd voice_project
   docker compose up --build
   ```

## Features

### RAG Project
- Semantic search for sleep-related queries
- Support for multiple embedding types
- Interactive Gradio interface
- Pre-built FAISS indices for quick deployment

### Voice Project
- Voice-to-text conversion using OpenAI Whisper
- Text-to-voice synthesis
- Conversation history tracking
- Multiple interface options (web and voice)

## Notes
- Both projects require an OpenAI API key for full functionality
- The RAG project includes pre-built FAISS indices for quick deployment
- The Voice project requires sufficient disk space for audio processing
- Docker deployment is recommended for consistent environment setup
