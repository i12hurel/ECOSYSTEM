from crewai import Task
from src.chatInterpreterLLM.agents.gemini_agent import return_gemini_agent

def return_task_interpretar_intencion(user_message, system_state: dict):
    serialized_state = str(system_state)
    task_interpretar_intencion = Task(
        description=f"""
            The user wrote: "{user_message}"

            This is the current system state:
            {serialized_state}

            Your responsibilities:
            - Understand the user's intention
            - If the user refers to an instance, extract the index (e.g., "Explain row 4" → index = 4)
            - Use the tool `execute_lime` passing the correct index
            - Always respond in a clear, conversational tone""",
        expected_output="A helpful assistant message or a tool-executed response.",
        agent=return_gemini_agent(user_message, system_state)
    )


    return task_interpretar_intencion
