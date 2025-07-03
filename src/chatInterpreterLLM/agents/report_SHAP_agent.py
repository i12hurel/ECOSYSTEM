from crewai import Agent, LLM
from src.chatInterpreterLLM.env import enviroment

def return_report_SHAP_agent():
    API_KEY = enviroment()
    report_SHAP_agent = Agent(
        role='AI Report Generator SHAP',
        goal='Generate an assessment report integrating the SHAP explanation with additional information from the info_text to provide the final expert with a clear summary of the factors influencing the prediction.',
        backstory="""You are an expert in analyzing machine learning models and preparing assessment reports. Your task is to combine the SHAP-based explanation (in JSON format already structured) with additional details about the database (e.g., title, sources, description, statistics, etc.).
        You must:
        1. Extract the three most important positive and negative factors from the explanation.
        2. Compare the probability of the predicted class versus the other classes.
        3. Incorporate and contextualize the information from the database, highlighting relevant data that can help interpret the prediction.

        Generate a clear and structured report for the final expert. For example, in the context of dropping out of school, you could write:
        'As you can see, he hasn't shown up for anything in the first semester, and we're already in the second. He probably won't show up for any subjects in the second semester either. If he had at least submitted Activity L3, he could have passed. Everything points to him dropping out of school.'
        Similarly, adapt the language to the context of the problem and the information available.""",
        verbose=True,
        allow_delegation=False,
        llm=LLM(
            model="gemini/gemini-2.0-flash-lite",
            temperature = 0.0,
            key = API_KEY,
            #vertex_credentials=vertex_credentials_json
        ) 
    )
    return report_SHAP_agent