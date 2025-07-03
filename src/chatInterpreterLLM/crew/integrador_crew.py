from crewai import Crew
from src.chatInterpreterLLM.agents.interpreter_agent import return_integrador_agent
from src.chatInterpreterLLM.tasks.interpreter_task import return_task_integration
from src.chatInterpreterLLM.knowledge.info import return_info_experto, return_info_text
from src.chatInterpreterLLM.agents.interpreter_agent import return_integrador_agent
from src.chatInterpreterLLM.crew.results_crew import return_resultado_crew_LIME, return_resultado_crew_SHAP, return_resultado_crew_ANCHOR
import json

def return_resultado_crew_integrador():
    crew_integration = Crew(
        agents=[return_integrador_agent()],
        tasks=[return_task_integration()],
        verbose=True
    )

    resultado_integrador = crew_integration.kickoff(
        inputs={"resultado_LIME": json.dumps(return_resultado_crew_LIME().model_dump(), indent=2),
            "resultado_SHAP": json.dumps(return_resultado_crew_SHAP().model_dump(), indent=2),
            "resultado_ANCHOR": json.dumps(return_resultado_crew_ANCHOR().model_dump(), indent=2),
            "info_experto": return_info_experto(),
            "info_text": return_info_text()
        }
    )
    return resultado_integrador