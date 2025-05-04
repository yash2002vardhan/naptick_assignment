from sentence_transformers import SentenceTransformer, util
import difflib
import torch

paraphrase_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")


device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

def get_cached_response(query, cache: list[dict],similarity_threshold = 0.85):
    for entry in cache:
        if query.lower() in entry['query'].lower():
            print("Found cached response")
            return entry['response']
    
    queries = [entry['query'] for entry in cache]
    close_matches = difflib.get_close_matches(query, queries, n = 1, cutoff = 0.7)
    if close_matches:
        print("Fuzzy match found.")
        match_query = close_matches[0]
        for entry in cache:
            if entry['query'] == match_query:
                return entry['response']

    query_embedding = paraphrase_model.encode(query, convert_to_tensor = True).to(device)
    for entry in cache:
        cached_embedding = paraphrase_model.encode(entry['query'], convert_to_tensor = True).to(device)
        similarity = util.pytorch_cos_sim(query_embedding, cached_embedding).item()
        if similarity > similarity_threshold:
            print("Semantic match found")
            return entry['response']
    
    print("No match found")
    return None
