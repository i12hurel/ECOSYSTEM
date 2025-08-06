from crewai import Crew
from src.chatInterpreterLLM.agents.coordinator_agent import return_coordinator_agent
from src.chatInterpreterLLM.tasks.coordinator_task import return_task_coordinator
from src.chatInterpreterLLM.agents.LIME_agent import return_LIME_agent
from src.chatInterpreterLLM.agents.SHAP_agent import return_SHAP_agent


def return_resultado_crew_coordinator(message):
    crew_coordinator = Crew(
        agents=[return_coordinator_agent(), return_LIME_agent(), return_SHAP_agent()],
        tasks=[return_task_coordinator(message)],
        verbose=True
    )

    resultado_coordinator = crew_coordinator.kickoff(inputs={"user_request": message})
    return resultado_coordinator