# Import necessary libraries
import os  # For environment variables and file operations

from langchain.retrievers import EnsembleRetriever  # For combining multiple retrievers
from dotenv import load_dotenv  # For loading environment variables
from langchain.prompts import PromptTemplate  # For creating prompt templates
from langchain.memory import ConversationBufferMemory  # For maintaining conversation history
import gradio as gr  # For creating web interface
from utils.get_vector_store import initialize_vector_stores
from langchain_openai import ChatOpenAI
from langchain.retrievers import EnsembleRetriever

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")  # Get OpenAI API key
pinecone_api_key = os.getenv("PINECONE_API_KEY")  # Get Pinecone API key
brave_api_key = os.getenv("BRAVE_API_KEY")  # Get Brave Search API key
llm_model = "gpt-4.1"  # Specify the language model to use

# Initialize vector stores
faiss_store, pinecone_store = initialize_vector_stores("openai", "faiss_index_openai", batch_size=100)

def select_retriever(faiss_store, pinecone_store, retriever_type):
    if pinecone_store is not None and retriever_type == "pinecone":
        retrievers = [store.as_retriever() for store in pinecone_store.values()]
        ensemble_retriever = EnsembleRetriever(
            retrievers=retrievers,
            weights=[1/len(retrievers)] * len(retrievers)
        )
        return ensemble_retriever
    elif faiss_store is not None and retriever_type == "faiss":
        return faiss_store.as_retriever()
    else:
        raise ValueError("No retriever type selected")

def get_memory():
    """
    Initialize and return a ConversationBufferMemory instance.
    Used to maintain conversation history.
    
    Returns:
        ConversationBufferMemory: The initialized memory
    """
    return ConversationBufferMemory(
        memory_key = "chat_history",
        input_key = "question",
        output_key = "answer",
        return_messages=True
    )

# Initialize memory
memory = get_memory()


def process_query(query, retriever_type, history):
    # Get appropriate retriever
    retriever = select_retriever(faiss_store, pinecone_store, retriever_type)

    # Update memory with history
    for user_msg, bot_msg in history:
        memory.save_context({"question": user_msg}, {"answer": bot_msg})

    # Retrieve context
    docs = retriever.get_relevant_documents(query) if retriever else []
    if not docs:
        return "I couldn't find any relevant information. Could you try rephrasing your query?"

    # Format retrieved documents
    formatted_docs = "\n".join([f"DOCUMENT {i+1}:\n{doc.page_content}\n\n" for i, doc in enumerate(docs[:8])])

    # Initialize LLM
    llm = ChatOpenAI(model=llm_model, api_key=openai_api_key) #type: ignore

    # Create prompt template
    template = """
    You are a helpful assistant. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    CONVERSATION HISTORY:
    {chat_history}

    RETRIEVED CONTEXT:
    {formatted_docs}

    Question: {question}

    Answer:"""
    
    prompt = PromptTemplate.from_template(template)

    # Run chain
    chain = prompt | llm
    result = chain.invoke({
        "formatted_docs": formatted_docs,
        "question": query,
        "chat_history": "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in memory.load_memory_variables({}).get("chat_history", [])])
    }).content
    
    # Save context for future reference
    memory.save_context({"question": query}, {"answer": result})
    
    return result

def create_gradio_interface():
    """
    Create a simple Gradio interface for the chatbot.
    """
    with gr.Blocks(title="Assistant") as interface:
        gr.Markdown("# Assistant")
        gr.Markdown("Ask me anything! I'll try to help you with your questions.")
        
        # Create chat interface
        chatbot = gr.Chatbot(height=500)
        msg = gr.Textbox(label="Your message", placeholder="What would you like to know?")
        retriever_type = gr.Dropdown(choices=["faiss", "pinecone"], label="Retriever Type", value="faiss")
        clear = gr.Button("Clear")

        def respond(message, retriever_type, history):
            bot_message = process_query(message, retriever_type, history)
            chat_history = history + [(message, bot_message)]
            return "", chat_history
        
        msg.submit(respond, [msg, retriever_type, chatbot], [msg, chatbot])
        clear.click(lambda: [], None, chatbot, queue=False)
    
    return interface

# Main entry point of the application
if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(share=True)  # Start the Gradio interface
