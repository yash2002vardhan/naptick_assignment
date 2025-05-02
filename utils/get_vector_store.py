from utils.generate_embeddings import get_embeddings
import pinecone
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore  
from langchain_community.vectorstores import FAISS
from utils.process_txt import flatten_conversations
from langchain.docstore.document import Document
import os

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")

def initialize_vector_stores(model:str, faiss_docs_dir:str):
    """
    Initialize and return vector stores (FAISS and Pinecone).
    If Pinecone index already exists, skip document ingestion.
    
    Args:
        model (str): Embedding model to use ('openai' or 'generic')
        faiss_docs_dir (str): Directory for FAISS index storage
        
    Returns:
        tuple: (faiss_store, pinecone_store) - The initialized vector stores
    """
    # Get embeddings based on model choice
    embeddings = get_embeddings(model)
    if model == "openai":
        embedding_size = 1536  # OpenAI embedding dimension
    else:
        embedding_size = 384  # HuggingFace embedding dimension
    
    # Define data sources and categories
    file_paths = ['datasets/fitbit_data/merged_hourly_data.csv', 'datasets/fitbit_data/heartrate_seconds_merged.csv', 'datasets/fitbit_data/merged_minute_data.csv', 'datasets/fitbit_data/merged_daily_data.csv', 'datasets/fitbit_data/weightLogInfo_merged.csv', "datasets/geotag_data.csv", "datasets/nutrition_data.csv", "datasets/user_data.csv", 'datasets/chats.txt']

    namespaces = ["fitbit", "geotag", "nutrition", "user", "chat"]

    faiss_docs = []
    pinecone_docs = {"fitbit": [], "geotag": [], "nutrition": [], "user": [], "chat": []}

    # Initialize Pinecone client
    pc = pinecone.Pinecone(api_key=pinecone_api_key)

    # Set index name based on model
    if model == "openai":
        index_name = "rag-naptick-openai"
    else:
        index_name = "rag-naptick-hf"

    index_exists = False

    # Check if Pinecone index exists
    try:
        indexes = pc.list_indexes()
        index_exists = any(index.name == index_name for index in indexes)
        
        if index_exists:
            print(f"Index '{index_name}' already exists. Skipping document ingestion.")
        else:
            print(f"Index '{index_name}' does not exist. Will create and ingest documents.")
            pc.create_index(index_name, dimension= embedding_size, spec = pinecone.ServerlessSpec(cloud="aws", region="us-east-1"))
    except Exception as e:
        print(f"Error checking Pinecone index: {e}")
        print("Will proceed with document processing for FAISS.")

    # Process documents for vector stores
    for file in file_paths:

        if file.endswith(".txt"):
            data_to_embed = flatten_conversations(file)
            documents = [Document(page_content=item["text"], metadata={"source": item["id"]}) for item in data_to_embed]
            split_docs = documents

        loader = CSVLoader(file_path=file)
        documents = loader.load()

        # Split documents into chunks for better processing
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)
        
        if file == "datasets/geotag_data.csv":
            namespace = "geotag"

        elif file == "datasets/nutrition_data.csv":
            namespace = "nutrition"

        elif file == "datasets/user_data.csv":
            namespace = "user"
        
        elif file == "datasets/chats.txt":
            namespace = "chat"
        
        else:
            namespace = "fitbit"
            
        # Add category metadata and prepare documents for stores
        for doc in split_docs:
            faiss_docs.append(doc)
            
            if not index_exists:
                pinecone_docs[namespace].append(doc)

    # Initialize FAISS store
    if os.path.exists(faiss_docs_dir) and faiss_docs_dir == "faiss_index_openai":
        print(f"Loading OpenAI FAISS store from {faiss_docs_dir}")
        faiss_store = FAISS.load_local(faiss_docs_dir, embeddings, allow_dangerous_deserialization=True)   
    elif os.path.exists(faiss_docs_dir) and faiss_docs_dir == "faiss_index_generic":
        print(f"Loading Generic FAISS store from {faiss_docs_dir}")
        faiss_store = FAISS.load_local(faiss_docs_dir, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"Creating new FAISS store in {faiss_docs_dir}")
        faiss_store = FAISS.from_documents(faiss_docs, embeddings)
        faiss_store.save_local(faiss_docs_dir)

    # Initialize Pinecone stores for each category
    pinecone_store = {}
    for namespace in namespaces:
        vector_store = PineconeVectorStore(
            index_name=index_name, 
            embedding=embeddings, 
            text_key="text", 
            namespace=namespace
        )
        
        # Add documents if index is new
        if not index_exists and namespace in pinecone_docs and pinecone_docs[namespace]:
            vector_store.add_documents(pinecone_docs[namespace])
        
        pinecone_store[namespace] = vector_store
    
    return faiss_store, pinecone_store

if __name__ == "__main__":
    initialize_vector_stores("generic", "faiss_index_generic")
