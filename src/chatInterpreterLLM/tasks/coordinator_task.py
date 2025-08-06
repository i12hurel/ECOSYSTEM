from crewai import Task
from src.chatInterpreterLLM.agents.coordinator_agent import return_coordinator_agent

def return_task_coordinator(message):
    task_coordinator = Task(
    description="""
                You have received the following request: "{user_request}"

                Your tasks:
                - Evaluate the content of the request.
                - If it requires an interpretation using LIME or SHAP, delegate it to the appropriate expert agent.
                - If it does not require delegation or is very general, you can answer it yourself.

                IMPORTANT:
                When using the 'Delegate work to coworker' tool, ALWAYS provide plain text for the fields:
                {{
                    "task": "Describe the task simply here",
                    "context": "Provide all necessary context here",
                    "coworker": "LIME Interpretation Expert / SHAP Interpretation Expert"
                }}
                """,
    agent=return_coordinator_agent(),
    expected_output="A detailed interpretation or answer to the user's request.",
    inputs={"user_request": message }
)

    
    return task_coordinator