from typing import Union, Any
from langchain_openai import ChatOpenAI
from ast import literal_eval
from utils.prompts import system_prompts
import os 

openai_api_key = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key) #type:ignore    


def choose_doc(query: str) -> Union[str, list[str], dict[Any, Any]]:
    """
    Choose the most relevant document from the list of documents based on the query.
    """


    research_papers_description = {
    "datasets/research_papers/1.md": "Clinical guidelines for using actigraphy in evaluating sleep disorders and circadian rhythm sleep-wake disorders in both adults and children",
    "datasets/research_papers/2.md": "A comprehensive position statement on the essential role of sleep in health, covering its impact on physical and mental well-being, public safety, and recommendations for sleep duration",
    "datasets/research_papers/3.md": "A detailed exploration of sleep's importance, focusing on its impact on cognitive function, attention, mood, and the detrimental effects of sleep deprivation on health and safety",
    "datasets/research_papers/4.md": "A scientific review of the role of glial cells in regulating sleep and circadian rhythms, examining their contributions to synaptic plasticity, metabolism, and immune responses"
    }

    csv_description = {
        "datasets/fitbit_data/heartrate_seconds_merged.csv": "Heart rate data recorded in seconds intervals from a Fitbit device",
        "datasets/fitbit_data/merged_daily_data.csv": "Aggregated daily activity and sleep metrics from Fitbit device",
        "datasets/fitbit_data/merged_minute_data.csv": "Activity and movement data recorded in minute intervals from Fitbit device",
        "datasets/fitbit_data/weightLogInfo_merged.csv": "Weight tracking and body composition data from Fitbit device",
        "datasets/fitbit_data/merged_hourly_data.csv": "Hourly aggregated activity and sleep metrics from Fitbit device",
        "datasets/sleep_data/combined_sleep_data.csv": "Comprehensive sleep metrics combining various sleep parameters and quality indicators",
        "datasets/sleep_data/wearable_tech_sleep_quality.csv": "Sleep quality metrics collected from wearable technology devices"
    }

    prompt = system_prompts['docs_prompt']

    response = model.invoke(prompt.format(csv_description=csv_description, research_papers_description=research_papers_description, query=query))
    response_list = literal_eval(response.content) #type: ignore
    return response_list

def parse_research_content(article_path: str):
    with open(article_path, "r") as file:
        content = file.read()
    return content[:5000]
