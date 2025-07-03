from crewai import Agent, LLM
from src.chatInterpreterLLM.env import enviroment

def return_integrador_agent():
    API_KEY = enviroment()
    integrador_agent = Agent(
        role='AI Integrated Explanation Expert',
        goal='Integrate interpretations from LIME, SHAP, and ANCHOR into a unified conclusion.',
        backstory="""You are skilled in synthesizing different machine learning interpretability techniques 
                    into coherent, actionable insights. You compare feature importance consistency across methods
                    and explain clearly why the combined findings lead to a final conclusion.""",
        verbose=True,
        allow_delegation=False,
        llm=LLM(
            model="gemini/gemini-2.0-flash-lite",
            temperature = 0.0,
            key = API_KEY,
            #vertex_credentials=vertex_credentials_json
        ) 
    )
    return integrador_agent