
from utils.get_relevant_docs import choose_doc, parse_research_content, model
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from utils.prompts import system_prompts

def get_answer(query:str):
    chosen_docs = choose_doc(query)
    print(f"Chosen docs: {chosen_docs}")

    response_list = []
    research_content = ""

    research_prompt = system_prompts['research_prompt']

    summary_prompt = system_prompts['summary_prompt']

    for doc in chosen_docs:
        if doc.endswith(".csv"):
            try:
                agent_executor = create_pandas_dataframe_agent(
                    llm=model,
                    df=pd.read_csv(doc),
                    verbose=True,
                    allow_dangerous_code=True,
                    handle_parsing_errors=True,
                )
                response = agent_executor.invoke({"input": query})
                response_list.append(response)
            except:
                continue

        if doc.endswith(".md"):
            print("Going to research mode")
            content = parse_research_content(doc)
            research_content += content
            response = model.invoke(research_prompt.format(research_content=research_content, query=query))
            response_list.append(response.content)


    final_response = ""

    if len(response_list) > 1:
        for conversation in response_list:
            try:
                answer = conversation['output']
            except:
                answer = conversation
            final_response += answer

    summary = model.invoke(summary_prompt.format(content=final_response, query=query))
    return summary.content
