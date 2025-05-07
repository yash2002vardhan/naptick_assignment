"""
Document selection module for the Voice Assistant.

This module provides functionality to select relevant documents based on user queries.
It maintains descriptions of available documents and uses language models to choose
the most appropriate ones for answering specific questions.

The module supports two types of documents:
1. CSV files containing structured data (Fitbit data, sleep metrics)
2. Research papers in markdown format

Dependencies:
    - langchain_openai: For language model integration
    - os: For environment variable access
"""

from typing import Union, Any
from langchain_openai import ChatOpenAI
from ast import literal_eval
from utils.prompts import system_prompts
import os 

# Initialize OpenAI model
openai_api_key = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key) #type:ignore    

def choose_doc(query: str) -> Union[str, list[str], dict[Any, Any]]:
    """
    Choose the most relevant documents based on the query.
    
    This function uses a language model to select the most appropriate documents
    from a predefined set of CSV files and research papers. Each document has
    a description that helps in determining its relevance to the query.
    
    Args:
        query (str): The user's question
    
    Returns:
        Union[str, list[str], dict[Any, Any]]: List of selected document paths
    
    Note:
        The function uses predefined descriptions of documents to help the
        language model make informed decisions about document relevance
    """
    # Document descriptions for CSV files
    research_papers_description = {
        "datasets/research_papers/1.md": "Clinical guidelines for using actigraphy in evaluating sleep disorders and circadian rhythm sleep-wake disorders in both adults and children",
        "datasets/research_papers/2.md": "A comprehensive position statement on the essential role of sleep in health, covering its impact on physical and mental well-being, public safety, and recommendations for sleep duration",
        "datasets/research_papers/3.md": "A detailed exploration of sleep's importance, focusing on its impact on cognitive function, attention, mood, and the detrimental effects of sleep deprivation on health and safety",
        "datasets/research_papers/4.md": "A scientific review of the role of glial cells in regulating sleep and circadian rhythms, examining their contributions to synaptic plasticity, metabolism, and immune responses"
    }

    # Document descriptions for research papers
    csv_description = {
        "datasets/fitbit_data/heartrate_seconds_merged.csv": "Heart rate data recorded in seconds intervals from a Fitbit device",
        "datasets/fitbit_data/merged_daily_data.csv": "Aggregated daily activity and sleep metrics from Fitbit device",
        "datasets/fitbit_data/merged_minute_data.csv": "Activity and movement data recorded in minute intervals from Fitbit device",
        "datasets/fitbit_data/weightLogInfo_merged.csv": "Weight tracking and body composition data from Fitbit device",
        "datasets/fitbit_data/merged_hourly_data.csv": "Hourly aggregated activity and sleep metrics from Fitbit device",
        "datasets/sleep_data/combined_sleep_data.csv": "Comprehensive sleep metrics combining various sleep parameters and quality indicators",
        "datasets/sleep_data/wearable_tech_sleep_quality.csv": "Sleep quality metrics collected from wearable technology devices"
    }

    # Get document selection prompt
    prompt = system_prompts['docs_prompt']

    # Get model's document selection
    response = model.invoke(prompt.format(csv_description=csv_description, research_papers_description=research_papers_description, query=query))
    response_list = literal_eval(response.content) #type: ignore
    return response_list

def parse_research_content(article_path: str):
    """
    Parse content from a research paper markdown file.
    
    Args:
        article_path (str): Path to the markdown file
    
    Returns:
        str: The first 5000 characters of the article content
    
    Note:
        This function is used to limit the amount of text processed
        for each research paper to avoid token limits
    """
    with open(article_path, "r") as file:
        content = file.read()
    return content[:5000]
