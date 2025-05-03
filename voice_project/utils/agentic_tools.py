
from utils.get_relevant_docs import choose_doc, parse_research_content, model
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent

def get_answer(query:str):
    chosen_docs = choose_doc(query)
    print(f"chosen_docs: {chosen_docs}")
    response_list = []

    research_content = ""
    content = ""

    csv_prompt = "There should be no pandas or any other code in the response. The response should just contain the answer to the question."

    research_prompt = f'''
            You are an expert in the field of sleep and sleep disorders. You have access to the following research papers:
            {research_content}
            
            User query: {query}
            
            Based on the user query, give me the most relevant information from the research papers in the simplest form possible.
            Do not include any other text tnat the answer.
            '''
    for doc in chosen_docs:
        if doc.endswith(".csv"):
            try:
                agent_executor = create_pandas_dataframe_agent(
                    llm=model,
                    df=pd.read_csv(doc),
                    verbose=True,
                    allow_dangerous_code=True,
                    handle_parsing_errors=True,
                    prompt=csv_prompt
                )
                response = agent_executor.invoke({"input": query})
                response_list.append(response)
            except:
                continue

        if doc.endswith(".md"):
            content = parse_research_content(doc)
            research_content += content
            response = model.invoke(research_prompt.format(content=research_content, query=query))
            response_list.append(response.content)

    summary_prompt = f'''You are a skilled expert at synthesizing complex information into clear, actionable advice. 

Given the following content: {content}

And the user's question: {query}

Please provide a focused summary that:
1. Directly answers the user's question in simple, everyday language
2. Includes specific, practical recommendations they can implement
3. Preserves all key data points, statistics, and research findings
4. Organizes information in a clear, logical flow
5. Uses a friendly, encouraging tone
6. The final response should be about 100 - 150 words in a paragraph format.

Focus only on the most relevant information to their question. Your response should contain just the summary itself, with no additional text or meta-commentary.'''

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
