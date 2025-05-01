import json
from pathlib import Path

def extract_conversations(file_path):
    text = Path(file_path).read_text()
    # Split based on assumption: multiple conversation blocks per file
    # Each block is a list starting with '[', ending with ']'
    blocks = text.split('\n\n')  # or use regex if not clearly separated
    conversations = []
    for block in blocks:
        try:
            convo = json.loads(block.strip())
            conversations.append(convo)
        except json.JSONDecodeError:
            continue
    return conversations


if __name__ == "__main__":
    conversations = extract_conversations("datasets/chats.txt")
    print(len(conversations))
