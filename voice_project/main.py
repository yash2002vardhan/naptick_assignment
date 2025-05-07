"""
Main module for the Voice Assistant application.

This module implements a voice-based question-answering system that:
1. Transcribes user's voice input using Whisper
2. Processes the query using an AI agent
3. Converts the response to speech with background music
4. Provides a Gradio web interface for interaction

The system includes caching functionality to improve response times for similar queries.

Dependencies:
    - whisper: For speech-to-text transcription
    - gtts: For text-to-speech conversion
    - gradio: For web interface
    - pydub: For audio processing
"""

from utils.agentic_tools import get_answer
import whisper
import gradio as gr
from gtts import gTTS
from utils.check_cache import get_cached_response
from ast import literal_eval
from utils.add_bgm import create_tts_with_music
import os

# Initialize Whisper model for speech recognition
model = whisper.load_model("base")

# Load response cache from file
with open("utils/responses.txt", "r") as file:
    content = file.read().strip()
    cache = literal_eval(content)
    print("Cache loaded successfully")

def process_audio(audio_path):
    """
    Process audio input and generate a voice response.
    
    This function:
    1. Transcribes the audio input using Whisper
    2. Checks for cached responses
    3. Gets a new response if no cache hit
    4. Converts the response to speech
    5. Adds background music to the speech
    
    Args:
        audio_path (str): Path to the input audio file
    
    Returns:
        tuple: (query, response_audio_path)
            - query: The transcribed text
            - response_audio_path: Path to the generated audio response
    
    Note:
        Temporary files are automatically cleaned up after processing
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
        print("Audio processing successful")
        os.remove(audio_response)
        
        return query, final_response
    
    except Exception as e:
        return f"Error processing audio: {str(e)}", None

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
