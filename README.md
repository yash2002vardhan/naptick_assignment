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
├── .env                  # Environment variables configuration
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
├── .env                  # Environment variables configuration
├── bgm.mp3               # Background music file
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── utils/                # Utility functions and helpers
├── openai_embeddings     # Openai embeddings for the alternate_main.py
└── datasets/             # Training and test datasets
```

## Prerequisites

- Python 3.10
- Git
- Docker (recommended)
- OpenAI API key (for both projects)
- Pinecone API key (for rag project)
- Sufficient disk space for FAISS indices and datasets
- FFmpeg (required for audio processing in Voice Project)
  - For macOS: 
   ```bash
   brew install ffmpeg
   ```
  - For Windows: Download from [FFmpeg official website](https://ffmpeg.org/download.html) and add to system PATH

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yash2002vardhan/naptick_assignment.git
   cd naptick_assignment
   ```

2. Create and activate a virtual environment (optional but recommended) with python 3.10 as the interpreter:
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
   cd voice_project
   pip install -r requirements.txt
   ```

## Environment Variables Setup

Copy the `.env.example` file to create a new `.env` file in each project directory and update the variables with your API keys:

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

   Please note  that you may need to change the port for gradio if port 7860 is not available. This can be done by configuring the gradio interface in `main.py` to use another port.
   ``` bash
   if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(server_name="0.0.0.0", server_port=<your_preferred_port>)
   ```

2. For Voice Project:
   ```bash
   cd voice_project
   python main.py
   ```
   The application will start a Gradio server accessible at `http://localhost:7860`

   Please note  that you may need to change the port for gradio if port 7860 is not available. This can be done by configuring the gradio interface in `main.py` to use another port.
   ``` bash
   if __name__ == "__main__":
    # Create Gradio interface
    iface = gr.Interface(
        fn=process_audio,
        inputs=gr.Audio(type="filepath"),
        outputs=[
            gr.Textbox(label="Your Question"),
            gr.Audio(label="AI Response")
         ],
        title="Voice Assistant",
        description="Ask a question about sleep and health through voice"
    )

    iface.launch(server_name="0.0.0.0", server_port=<your_preferred_port>)
   ```
   There's an alternative implementation of this project that uses RAG approach instead of dynamically accessing documents during runtime (as done in main.py). This version leverages `crewai` along with the `RagTool` (from crewai) to streamline the process.

   To try it out, follow these steps:

   ```bash
   python alternate_main.py
   ```
   The application will start a Gradio server accessible at `http://localhost:7860`

   Please note  that you may need to change the port for gradio if port 7860 is not available. This can be done by configuring the gradio interface in `main.py` to use another port.
   ``` bash
   if __name__ == "__main__":
    # Create Gradio interface
    iface = gr.Interface(
        fn=process_audio,
        inputs=gr.Audio(type="filepath"),
        outputs=[
            gr.Textbox(label="Your Question"),
            gr.Audio(label="AI Response")
        ],
        title="Voice Assistant",
        description="Ask a question about sleep and health through voice"
    )

    iface.launch(server_name="0.0.0.0", server_port=<your_preferred_port>)
   ```



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

Please note that Gradio may take a few minutes to initialize inside the Docker container. You can monitor the container logs to track the startup progress. 

## Features and Other details
This can be accessed from the Google doc present at this link: [Naptick_Assignment](https://docs.google.com/document/d/1L-lmqtymVzcR5taQ-d3nLUIdj7yirM4yEspezLH29aw/edit?usp=sharing)
