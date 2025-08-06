from crewai import Agent, LLM
from src.chatInterpreterLLM.env import enviroment

def return_SHAP_agent():
    API_KEY = enviroment()

    SHAP_agent = Agent(
        role='AI Model Report Generator SHAP',
        goal='Generate clear and accurate SHAP-based explanations of model predictions, focusing on the contribution of each feature according to SHAP values.'
                'with additional information from the metadata to provide the final expert with a clear summary of the factors influencing the prediction.',
        backstory="""You are an expert in interpreting machine learning models using SHAP, with a focus on feature-level contribution explanations. 
                    Your task is to analyze SHAP explanations and translate them into 
                    natural language, strictly following the provided data without 
                    adding speculative information.
                    
                    You must:
                    1. Extract the three most important positive and negative factors from the explanation.
                    2. Compare the probability of the predicted class versus the other classes.
                    3. Incorporate and contextualize the information from the database, highlighting relevant data that can help interpret the prediction.

                    Generate a clear and structured report for the final expert.""",
        verbose=True,
        allow_delegation=False,
        llm=LLM(
            model="gemini/gemini-2.0-flash-lite",
            temperature=0.0,
            key=API_KEY,
        ) 
    )
    return SHAP_agent