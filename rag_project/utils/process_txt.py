"""
Text processing module for handling chat conversations in the RAG system.

This module provides functionality to process and structure chat conversations
from text files. It extracts conversations from text blocks, flattens them into
a format suitable for embedding, and maintains the conversation structure with
system prompts, user messages, and assistant responses.

The module expects chat conversations to be stored in a specific format where
each conversation is a separate block of text containing a Python dictionary
representation of the conversation turns.
"""

from pathlib import Path

def extract_conversations(file_path):
    """
    Extract conversations from a text file.
    
    This function reads a text file and extracts individual conversations
    from text blocks. Each block should contain a valid Python dictionary
    representation of a conversation with turns.
    
    Args:
        file_path (str): Path to the text file containing conversations
    
    Returns:
        list: List of conversation dictionaries, each containing conversation turns
    
    Note:
        Invalid blocks (those that can't be evaluated as Python dictionaries)
        are silently skipped
    """
    text = Path(file_path).read_text()
    blocks = text.split('\n\n') 
    conversations = []
    for block in blocks:
        try:
            convo = eval(block.strip())
            conversations.append(convo)
        except (SyntaxError, ValueError):
            continue
    return conversations


def flatten_conversations(file_path):
    """
    Flatten conversations into a format suitable for embedding.
    
    This function processes conversations and combines them into a structured
    format that includes system prompts, user messages, and assistant responses.
    Each conversation is assigned a unique ID and formatted as a single text block.
    
    Args:
        file_path (str): Path to the text file containing conversations
    
    Returns:
        list: List of dictionaries, each containing:
            - id: Unique identifier for the conversation
            - text: Combined text of the conversation including system prompt,
                   user messages, and assistant responses
    
    Example output format:
        [
            {
                "id": "conversation-0",
                "text": "SYSTEM: <system_prompt>\n\nUSER:\n<user_message>\n\nASSISTANT:\n<assistant_response>"
            },
            ...
        ]
    """
    data_to_embed = []
    conversations = extract_conversations(file_path)
    for idx, convo in enumerate(conversations):
        user_messages = []
        assistant_messages = []
        system_prompt = ""

        for turn in convo:
            if turn["role"] == "system":
                system_prompt = turn["content"]
            elif turn["role"] == "user" and turn["content"]:
                user_messages.append(turn["content"])
            elif turn["role"] == "assistant" and turn["content"]:
                assistant_messages.append(turn["content"])

        # Combine into a structured chunk
        combined = f"SYSTEM: {system_prompt}\n\nUSER:\n" + "\n".join(user_messages) + "\n\nASSISTANT:\n" + "\n".join(assistant_messages)
        data_to_embed.append({
            "id": f"conversation-{idx}",
            "text": combined,
        })
    return data_to_embed
