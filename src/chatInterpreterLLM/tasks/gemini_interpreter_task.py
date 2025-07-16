from crewai import Task
from src.chatInterpreterLLM.agents.gemini_agent import return_gemini_agent
import json

def return_task_interpretar_intencion(user_message, system_state: dict):
    serialized_state = json.dumps(system_state, indent=2)

    return Task(
        description=f"""
        The user said: "{user_message}"

        System state:
        {serialized_state}

        Your role:
        You are a smart conversational AI that interprets user requests and interacts with machine learning explanation tools.

        You can:
        - Ask the user to upload a dataset (if none is present)
        - Let them upload metadata from a `.txt` file in the sidebar
        - Accept additional comments or insights written by the user directly in the chat
        - Run LIME explanation on an instance (from the dataset or a new one)
        - Reset the session

        Available tools:
        - pedir_dataset
        - cargar_metadata
        - recibir_instancia_dataset
        - ejecutar_crew_LIME_dataset (index: int)
        - ejecutar_crew_LIME_nueva (instance_data: dict)
        - añadir_contexto (content: str)
        - reiniciar_sesion

        How to choose the correct tool:
        - If the user asks to explain an instance from the dataset and provides an index (e.g. "explain instance 5"), call:
        `ejecutar_crew_LIME_dataset(index=5)`

        - If the user message includes a new instance (i.e. a structured list of attribute values, such as a dictionary or JSON), or mentions "I want to explain this new instance", then:
            → Parse that data as a dictionary
            → Call: ejecutar_crew_LIME_nueva(instance_data=<parsed_instance>)
            → If the system state includes `instance_uploaded: true`, assume the instance is already available in memory and do not ask again.



        - If the user provides any textual insight, commentary, or clarification, add it to the system using:
        `añadir_contexto(content=<user_input>)`

        - Never call `ejecutar_crew_LIME_dataset` without a valid integer index.
        - Do not call any tool without the required argument.
        - Always respond clearly and conversationally after using a tool.
        """,
        expected_output="A helpful assistant message or a tool-executed response.",
        agent=return_gemini_agent(user_message, system_state)
    )
