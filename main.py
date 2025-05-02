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
        llm=ChatOpenAI(model=llm_model, api_key=openai_api_key),  # type: ignore
        return_messages=True
    )

# Initialize memory
memory = get_memory()

def process_query(query, history,retriever_type):
    """
    Process user queries using the selected retrievers and return a response.
    
    Args:
        query (str): User's query
        history (list): List of previous conversation turns
        
    Returns:
        str: Response to the query
    """
    # Get appropriate retriever
    retriever = select_retriever(faiss_store, pinecone_store, retriever_type)
    
    # Get conversation history
    memory_variables = memory.load_memory_variables({})
    conversation_history = memory_variables.get("history", "")
    
    try:
        docs = retriever.get_relevant_documents(query) #type: ignore
    except Exception as e:
        print(f"Error with retriever: {e}")
        docs = []
    
    if not docs:
        return "I couldn't find any relevant information. Could you try rephrasing your query?"
    
    # Format retrieved documents
    formatted_docs = ""
    for i, doc in enumerate(docs[:8]):
        formatted_docs += f"DOCUMENT {i+1}:\n"
        formatted_docs += f"{doc.page_content}\n\n"
    
    # Initialize language model
    llm = ChatOpenAI(model=llm_model, api_key=openai_api_key) #type: ignore
    
    # Create prompt template
    template = """
    You are a helpful assistant. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    CONVERSATION HISTORY:
    {conversation_history}
    
    RETRIEVED CONTEXT:
    {formatted_docs}
    
    Question: {question}
    
    Answer:"""
    
    PROMPT = PromptTemplate.from_template(template)
    
    # Create and run chain
    chain = PROMPT | llm
    result = chain.invoke({
        "formatted_docs": formatted_docs, 
        "question": query,
        "conversation_history": conversation_history
    }).content
    
    # Save conversation context
    memory.save_context({"input": query}, {"output": result}) #type: ignore
    
    return result

def create_gradio_interface():
    """
    Creates a Gradio interface for the chatbot.
    This function handles:
    - Setting up the chat interface
    - Managing chat history 
    - Processing user input and displaying responses
    """
    # Create the Gradio interface
    with gr.Blocks(title="Assistant") as interface:
        gr.Markdown("# Assistant")
        gr.Markdown("Ask me anything! I'll try to help you with your questions.")
        
        # Create chat interface
        chatbot = gr.Chatbot(height=500)
        msg = gr.Textbox(label="Your message", placeholder="What would you like to know?")
        retriever_type = gr.Dropdown(choices=["faiss", "pinecone"], label="Retriever Type", value="faiss")
        clear = gr.Button("Clear")
        
        def respond(message, chat_history, retriever_type):
            bot_message = process_query(message, chat_history, retriever_type)
            chat_history.append((message, bot_message))
            return "", chat_history
        
        msg.submit(respond, [msg, chatbot, retriever_type], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)
    
    return interface

# Main entry point of the application
if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(share=True)  # Start the Gradio interface
