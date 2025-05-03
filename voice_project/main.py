from utils.agentic_tools import get_answer

if __name__ == "__main__":
    query = "plan a daily schedule for me to have a good sleep in night"
    answer = get_answer(query)
    print(answer)
