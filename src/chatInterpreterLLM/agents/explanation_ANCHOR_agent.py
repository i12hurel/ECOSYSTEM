from crewai import Agent, LLM
from src.chatInterpreterLLM.env import enviroment

def return_explanation_ANCHOR_agent():
    API_KEY = enviroment()
    explanation_ANCHOR_agent = Agent(
        role='AI Model Explainer ANCHOR',
        goal='Generate structured natural language interpretations of anchor-based model predictions.',
        backstory="""You specialize in interpreting machine learning models using Anchor explanations. 
                    You transform logical rule-based outputs into clear, human-readable insights that explain 
                    how a model prediction is "anchored" by certain key features. Your job is to strictly 
                    interpret the anchor rule provided, explain the conditions, and highlight their meaning 
                    in the context of the student’s data, without making assumptions beyond the given output.""",
        verbose=True,
        allow_delegation=False,
        #tool=explicacion_ANCHOR,
        llm=LLM(
            model="gemini/gemini-2.0-flash-lite",
            temperature = 0.0,
            key = API_KEY,
            #vertex_credentials=vertex_credentials_json
        ) 
    )

    return explanation_ANCHOR_agent