"""
Response caching module for the Voice Assistant.

This module provides functionality to cache and retrieve responses using
semantic similarity matching. It uses sentence transformers to find similar
queries and their corresponding responses, improving response times for
repeated or similar questions.

The module supports three types of matching:
1. Exact substring matching
2. Fuzzy string matching
3. Semantic similarity matching using embeddings

Dependencies:
    - sentence_transformers: For semantic similarity matching
    - torch: For tensor operations
    - difflib: For fuzzy string matching
"""

from sentence_transformers import SentenceTransformer, util
import difflib
import torch

# Initialize sentence transformer model for semantic similarity
paraphrase_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Set device for model inference
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

def get_cached_response(query, cache: list[dict], similarity_threshold = 0.85):
    """
    Get a cached response for a query using multiple matching strategies.
    
    This function tries three methods to find a matching response:
    1. Exact substring matching
    2. Fuzzy string matching (using difflib)
    3. Semantic similarity matching (using sentence transformers)
    
    Args:
        query (str): The user's question
        cache (list[dict]): List of cached query-response pairs
        similarity_threshold (float, optional): Threshold for semantic similarity.
            Defaults to 0.85
    
    Returns:
        str or None: The cached response if a match is found, None otherwise
    
    Note:
        The function uses a hierarchical approach, trying exact matches first,
        then fuzzy matches, and finally semantic similarity matching
    """
    # Try exact substring matching
    for entry in cache:
        if query.lower() in entry['query'].lower():
            print("Found cached response")
            return entry['response']
    
    # Try fuzzy string matching
    queries = [entry['query'] for entry in cache]
    close_matches = difflib.get_close_matches(query, queries, n = 1, cutoff = 0.7)
    if close_matches:
        print("Fuzzy match found.")
        match_query = close_matches[0]
        for entry in cache:
            if entry['query'] == match_query:
                return entry['response']

    # Try semantic similarity matching
    query_embedding = paraphrase_model.encode(query, convert_to_tensor = True).to(device)
    for entry in cache:
        cached_embedding = paraphrase_model.encode(entry['query'], convert_to_tensor = True).to(device)
        similarity = util.pytorch_cos_sim(query_embedding, cached_embedding).item()
        if similarity > similarity_threshold:
            print("Semantic match found")
            return entry['response']
    
    print("No match found")
    return None
