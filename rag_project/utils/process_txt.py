from pathlib import Path

def extract_conversations(file_path):
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
