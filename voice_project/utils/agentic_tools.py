"""
Agent-based tools module for the Voice Assistant.

This module provides functionality for processing user queries using a combination
of data analysis and research paper processing. It uses pandas agents for CSV data
analysis and language models for research paper content processing.

The module supports:
1. CSV data analysis using pandas agents
2. Research paper content processing
3. Response summarization and synthesis

Dependencies:
    - langchain_experimental: For pandas dataframe agents
    - pandas: For data analysis
    - langchain: For language model integration
"""

from utils.get_relevant_docs import choose_doc, parse_research_content, model
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from utils.prompts import system_prompts

def get_answer(query:str):
    """
    Process a query using a combination of data analysis and research.
    
    This function:
    1. Selects relevant documents based on the query
    2. Processes CSV files using pandas agents
    3. Processes research papers using language models
    4. Combines and summarizes the responses
    
    Args:
        query (str): The user's question
    
    Returns:
        str: A summarized response combining insights from both data analysis
            and research papers
    
    Note:
        The function handles both structured data (CSV) and unstructured data
        (research papers) to provide comprehensive answers
    """
    # Select relevant documents
    chosen_docs = choose_doc(query)
    print(f"Chosen docs: {chosen_docs}")

    response_list = []
    research_content = ""

    # Get prompts for research and summary
    research_prompt = system_prompts['research_prompt']
    summary_prompt = system_prompts['summary_prompt']

    # Process each selected document
    for doc in chosen_docs:
        if doc.endswith(".csv"):
            try:
                # Create pandas agent for CSV analysis
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
            # Process research paper content
            content = parse_research_content(doc)
            research_content += content
            response = model.invoke(research_prompt.format(research_content=research_content, query=query))
            response_list.append(response.content)

    # Combine and summarize responses
    final_response = ""

    if len(response_list) > 1:
        for conversation in response_list:
            try:
                answer = conversation['output']
            except:
                answer = conversation
            final_response += answer

    # Generate final summary
    summary = model.invoke(summary_prompt.format(content=final_response, query=query))
    return summary.content
