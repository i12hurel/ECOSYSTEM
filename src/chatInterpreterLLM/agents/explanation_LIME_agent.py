from crewai import Agent, LLM
from src.chatInterpreterLLM.env import enviroment

def return_explanation_LIME_agent():
    API_KEY = enviroment()

    explanation_LIME_agent = Agent(
        role='AI Model Explainer LIME',
        goal='Generate clear and accurate explanations of model predictions highlighting both positive and negative contributors using LIME',
        backstory="""You are an expert in interpreting machine learning models with focus on feature contributions. 
                    Your task is to analyze LIME explanations and translate them into 
                    natural language, strictly following the provided data without 
                    adding speculative information.""",
        verbose=True,
        allow_delegation=False,
        #tool = explicacion_LIME,
        llm = LLM(
            model="gemini/gemini-2.0-flash-lite",
            temperature = 0.0,
            key = API_KEY,
            #vertex_credentials=vertex_credentials_json
        ) 
    )
    return explanation_LIME_agent