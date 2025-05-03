from utils.agentic_tools import get_answer
import whisper
import gradio as gr
from gtts import gTTS

model = whisper.load_model("base")

def process_audio(audio_path):
    # Transcribe audio using whisper
    try:
        transcription = model.transcribe(audio_path)
        query = str(transcription["text"])
        
        # Get answer from agent
        response = str(get_answer(query))
        
        # Convert response to speech
        tts = gTTS(text=response, lang='en')
        audio_response = "temp_response.mp3"
        tts.save(audio_response)
        
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
