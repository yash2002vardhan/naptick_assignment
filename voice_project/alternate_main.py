"""
Alternative main module for the Voice Assistant application using CrewAI.

This module implements a voice-based question-answering system using CrewAI's
agent framework. It provides similar functionality to main.py but uses a more
structured approach with specialized agents and tasks.

Key features:
1. Uses CrewAI for agent-based question answering
2. Implements RAG (Retrieval-Augmented Generation) using Chroma vector store
3. Supports both CSV data analysis and research paper processing
4. Includes caching for improved response times

Dependencies:
    - crewai: For agent-based task execution
    - crewai_tools: For RAG implementation
    - whisper: For speech-to-text transcription
    - gtts: For text-to-speech conversion
    - gradio: For web interface
"""

from crewai import Crew, Task, Agent, LLM
from crewai_tools import RagTool
import os
from dotenv import load_dotenv
import whisper
import gradio as gr
from gtts import gTTS
from utils.check_cache import get_cached_response
from ast import literal_eval
from utils.add_bgm import create_tts_with_music

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize models
model = whisper.load_model("base")
llm = LLM(model = "openai/gpt-4o", api_key = openai_api_key)

# Load response cache
with open("utils/responses.txt", "r") as file:
    content = file.read().strip()
    cache = literal_eval(content)
    print("Cache loaded successfully")

# Configure vector store settings
embeddings_path = "openai_embeddings"
collection_name = "sleep_collection"

def process_audio(audio_path):
    """
    Process audio input and generate a voice response using CrewAI.
    
    This function:
    1. Transcribes the audio input using Whisper
    2. Checks for cached responses
    3. Gets a new response using CrewAI agents if no cache hit
    4. Converts the response to speech with background music
    
    Args:
        audio_path (str): Path to the input audio file
    
    Returns:
        tuple: (query, response_audio_path)
            - query: The transcribed text
            - response_audio_path: Path to the generated audio response
    """
    # Transcribe audio using whisper
    try:
        transcription = model.transcribe(audio_path)
        query = str(transcription["text"])
        
        # Get answer from agent
        cached_response = get_cached_response(query = query, cache = cache)
        if cached_response == None:
            response = str(get_answer(query))
            cache.append({"query": query, "response": response})
            with open("utils/responses.txt", "w") as file:
                file.write(str(cache))
                print("Cache updated successfully")
        else:
            response = cached_response
        
        # Convert response to speech
        tts = gTTS(text=response, lang='en')
        audio_response = "temp_response.mp3"
        tts.save(audio_response)
        final_response = create_tts_with_music(audio_response)
        os.remove(audio_response)
        print("Audio processing successful")
        
        return query, final_response
    
    except Exception as e:
        return f"Error processing audio: {str(e)}", None

# Configure RAG tool settings
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o",
        }
    },
    "embedding_model": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small"
        }
    },
    "vectordb": {
        "provider": "chroma",
        "config": {
            "dir": embeddings_path,
            "collection_name": collection_name
        }
    }
}

# Initialize RAG tool
rag_tool = RagTool(config=config) #type: ignore

# Create vector database if it doesn't exist
if os.path.exists(embeddings_path):
    print("Embeddings already exist")
else:
    print("Embeddings do not exist, creating new vector database")
    rag_tool.add(data_type = "directory", source = "/Users/yashvardhan/Documents/Desktop_Folders/ProjectsAndTutorials/naptick_assignment/voice_project/datasets")

def get_answer(query):
    """
    Get answer using CrewAI agents and RAG.
    
    This function:
    1. Creates a sleep coach agent with RAG capabilities
    2. Defines a task for answering the query
    3. Executes the task using the agent
    4. Returns the processed response
    
    Args:
        query (str): The user's question
    
    Returns:
        str: The generated response
    
    Note:
        The agent is configured to provide sleep-related advice based on
        both research papers and user data analysis
    """
    try:
        # Create sleep coach agent
        sleep_agent = Agent(
            role = "A verified sleep coach",
            goal  = " You need to advice the user on how to improve their sleep quality",
            backstory = "You are a sleep coach with a deep understanding of sleep science and psychology. You are able to provide personalized advice to users based on their sleep patterns, goals and queries",
            verbose = True,
            allow_delegation = True,
            llm = llm,
            tools = [rag_tool],
            max_retry_limit = 5
        )

        # Create task for the agent
        task  = Task(
            description = query,
            expected_output = "a summarised response which covers all the major information such as any figures, or any other information that is relevant to the user query. The output should be in a conversational tone and should be easy to understand. Also there should be no special formatting or markdown in the output",
            agent = sleep_agent
        )

        # Execute task
        crew = Crew(agents = [sleep_agent], tasks = [task], verbose = True)
        task_output = crew.kickoff()

        return task_output
    
    except Exception as e:
        return f"Error getting answer: {str(e)}"

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

    iface.launch(server_name="0.0.0.0", server_port=7860)
