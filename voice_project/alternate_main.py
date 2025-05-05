from crewai import Crew, Task, Agent, LLM
from crewai_tools import RagTool
import os
from dotenv import load_dotenv
import whisper
import gradio as gr
from gtts import gTTS
from utils.check_cache import get_cached_response
from ast import literal_eval

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

model = whisper.load_model("base")
llm = LLM(model = "openai/gpt-4o", api_key = openai_api_key)

with open("utils/responses.txt", "r") as file:
    content = file.read().strip()
    cache = literal_eval(content)
    print("Cache loaded successfully")



embeddings_path = "voice_project/db"
collection_name = "4ce43c40-eb0d-49bf-ad1e-0d9234131000"

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



# TODO: create embeddings using the openai api, currently using ollama with 384 embedding size
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o",
        }
    },
    "embedding_model": {
        "provider": "ollama",
        "config": {
            "model": "all-minilm:latest"
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



rag_tool = RagTool(config=config) #type: ignore


if os.path.exists(embeddings_path):
    print("Embeddings already exist")
else:
    print("Embeddings do not exist")
    rag_tool.add(data_type = "directory", source = "/Users/yashvardhan/Documents/Desktop_Folders/ProjectsAndTutorials/naptick_assignment/voice_project/datasets")


def get_answer(query):

    try:

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


        task  = Task(
            description = query,
            expected_output = "a summarised response which covers all the major information such as any figures, or any other information that is relevant to the user query",
            agent = sleep_agent
        )

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

    iface.launch()
