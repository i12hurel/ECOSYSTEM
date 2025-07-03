from crewai import Agent, LLM
from src.chatInterpreterLLM.env import enviroment

def return_explanation_SHAP_agent():
    API_KEY = enviroment()

    explanation_SHAP_agent = Agent(
        role='AI Model Explainer SHAP',
        goal='Generate clear and accurate SHAP-based explanations of model predictions, focusing on the contribution of each feature according to SHAP values.',
        backstory="""You are an expert in interpreting machine learning models using SHAP, with a focus on feature-level contribution explanations. 
                    Your task is to analyze SHAP explanations and translate them into 
                    natural language, strictly following the provided data without 
                    adding speculative information.""",
        verbose=True,
        allow_delegation=False,
        #tool = explicacion_SHAP,
        llm = LLM(
            model="gemini/gemini-2.0-flash-lite",
            temperature = 0.0,
            key = API_KEY,
            #vertex_credentials=vertex_credentials_json
        ) 
    )
    return explanation_SHAP_agent