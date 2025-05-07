"""
Embedding generation module for the RAG system.

This module provides functionality to generate text embeddings using either OpenAI's
text-embedding-3-small model or HuggingFace's sentence-transformers model. The embeddings
are used to create vector representations of text for similarity search in the RAG system.

Dependencies:
    - langchain_huggingface: For HuggingFace embeddings
    - langchain_openai: For OpenAI embeddings
    - dotenv: For environment variable management
"""

from langchain_huggingface import HuggingFaceEmbeddings  # For text embeddings using HuggingFace models
from langchain_openai import OpenAIEmbeddings  # For OpenAI's language models and embeddings
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def get_embeddings(model: str):
    """
    Get text embeddings based on specified model.
    
    This function initializes and returns an embeddings model based on the specified type.
    It supports two types of embeddings:
    1. OpenAI's text-embedding-3-small model (1536 dimensions)
    2. HuggingFace's sentence-transformers/all-MiniLM-L6-v2 model (384 dimensions)
    
    Args:
        model (str): The type of embedding model to use. Must be either:
            - 'openai': Uses OpenAI's text-embedding-3-small model
            - 'generic': Uses HuggingFace's sentence-transformers model
    
    Returns:
        An embeddings object that can be used to generate vector representations of text
    
    Note:
        The OpenAI model requires a valid API key to be set in the environment variables
    """
    if model == "openai":
        return OpenAIEmbeddings(model = "text-embedding-3-small", api_key = openai_api_key) #type: ignore
    else:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
