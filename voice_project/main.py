from utils.agentic_tools import get_answer
import whisper
import gradio as gr
from gtts import gTTS
from utils.check_cache import get_cached_response
from ast import literal_eval

model = whisper.load_model("base")

with open("utils/responses.txt", "r") as file:
    content = file.read().strip()
    cache = literal_eval(content)
    print("Cache loaded successfully")

def process_audio(audio_path):
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
        print("Audio processing successful")
        
        return query, audio_response
    
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

    iface.launch()
