from crewai import Agent, LLM
from src.chatInterpreterLLM.env import enviroment
from src.chatInterpreterLLM.tools.explanation_LIME import ExplanationLIMETool, RecibirInstanciaDatasetTool, RecibirInstanciaNuevaTool

def return_LIME_agent():
    """Returns an LLM agent for generating reports based on LIME explanations."""
    
    API_KEY = enviroment()

    LIME_agent = Agent(
        role = 'AI Model Report Generator LIME',
        goal = 'Generate clear and accurate explanations of model predictions highlighting both positive and negative contributors using LIME '
                'with additional information from the metadata to provide the final expert with a clear summary of the factors influencing the prediction.',
        backstory = """You are an expert in analyzing machine learning models with focus on feature contributions and preparing assessment reports.
                    Your task is to analyze LIME explanations and translate them into 
                    natural language, strictly following the provided data without 
                    adding speculative information.
                    
                    You must:
                    1. Compare the probability of the predicted class versus the other classes.
                    2. Extract the three most important positive and negative factors from the explanation.

                    Generate a clear and structured report for the final expert.""",
        verbose=True,
        allow_delegation=False,
        tools= [
            ExplanationLIMETool(),
            RecibirInstanciaDatasetTool(),
            RecibirInstanciaNuevaTool()
        ],
        llm=LLM(
            model="gemini/gemini-2.0-flash-lite",
            temperature = 0.0,
            key = API_KEY
        )
    )

    return LIME_agent

#                    3. Only if you have been given additional information about the dataset (metadata) or additional information from the expert (expert_notes), incorporate and contextualize the information from the database, highlighting relevant data that can help interpret the prediction.

        #                    3. Explain why the factors are considered positive or negative, using the metadata if available.
