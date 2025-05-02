from langchain_huggingface import HuggingFaceEmbeddings  # For text embeddings using HuggingFace models
from langchain_openai import OpenAIEmbeddings  # For OpenAI's language models and embeddings
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def get_embeddings(model: str):
    """
    Get text embeddings based on specified model
    Args:
        model (str): Either 'openai' or 'generic' for HuggingFace
    Returns:
        Embeddings object for the specified model
    """
    if model == "openai":
        return OpenAIEmbeddings(model = "text-embedding-3-small", api_key = openai_api_key) #type: ignore
    else:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
