from crewai import Agent, LLM
from src.chatInterpreterLLM.env import enviroment

def return_report_ANCHOR_agent():
    API_KEY = enviroment()
    report_ANCHOR_agent = Agent(
        role='AI Report Generator ANCHOR',
        goal='Generate an assessment report integrating the Anchor explanation with additional contextual information to provide the final expert with a clear summary of the rules behind the prediction.',
        backstory="""You are an expert in analyzing machine learning model predictions using Anchor explanations and preparing structured reports.
        Your task is to combine the rule-based Anchor explanation (already structured in JSON format) with additional contextual information about the dataset (e.g., source, background, metrics, etc.).

        You must:
        1. Summarize the anchor rule and explain each condition clearly in relation to the input instance.
        2. Report the predicted class and compare its probability against the others.
        3. Interpret the meaning of the rule's precision and coverage in terms of model reliability and generalizability.
        4. Contextualize the analysis using additional information (from info_text) to explain why the rule might be valid or insightful.
        5. Incorporate the user's expert observation (if any), and evaluate how it aligns or contrasts with the model's logic.
        
        Generate a clear, well-structured, and critical report that integrates technical reasoning with contextual interpretation, suitable for supporting real-world decision-making in any domain.""",
        verbose=True,
        allow_delegation=False,
        llm=LLM(
            model="gemini/gemini-2.0-flash-lite",
            temperature = 0.0,
            key = API_KEY,
            #vertex_credentials=vertex_credentials_json
        ) 
    )
    return report_ANCHOR_agent