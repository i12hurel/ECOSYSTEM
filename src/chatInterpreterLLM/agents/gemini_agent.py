from crewai import Agent, LLM
from src.chatInterpreterLLM.env import enviroment
from src.chatInterpreterLLM.tools.gemini_tools import (
    RequestDatasetTool,
    RequestInstanceTool,
    UploadMetadataTool,
    AddContextTool,
    RunLIMEDatasetInstanceTool,
    RunLIMENewInstanceTool
)

def return_gemini_agent(user_message: str, system_state: dict):
    API_KEY = enviroment()
    gemini_agent = Agent(
        role="Conversational AI Assistant",
        goal=(
            "Understand the user's intent and autonomously take actions to assist in the explanation of machine learning model predictions. "
            "If the message is unrelated to technical actions (e.g., greetings or casual conversation), simply respond naturally as a friendly assistant without triggering any tool."
        ),
        backstory=(
            "You are the central decision-maker in a system that helps users understand machine learning predictions. "
            "You have access to tools that allow you to request datasets, explain instances using LIME, add contextual notes, or reset the session. "
            "You must interpret the user's request along with the current state of the system and decide which action (if any) is needed."
        ),
        llm=LLM(
            model="gemini/gemini-2.0-flash-lite",
            temperature=0.5,
            key=API_KEY
        ),
        tools=[
            RequestDatasetTool(),
            RequestInstanceTool(),
            UploadMetadataTool(),
            AddContextTool(),
            RunLIMEDatasetInstanceTool(),
            RunLIMENewInstanceTool(),
        ]
    )
    return gemini_agent