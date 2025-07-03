from crewai import Crew
from src.chatInterpreterLLM.agents.explanation_SHAP_agent import return_explanation_SHAP_agent
from src.chatInterpreterLLM.agents.report_SHAP_agent import return_report_SHAP_agent
from src.chatInterpreterLLM.tasks.explanation_SHAP_task import return_task_explanation_SHAP
from src.chatInterpreterLLM.tasks.report_SHAP_task import return_task_report_SHAP
from src.chatInterpreterLLM.knowledge.info import return_info_experto, return_info_text
from src.chatInterpreterLLM.tools.explanation_SHAP import explicacion_SHAP

def return_resultado_crew_SHAP():
  crew_SHAP = Crew(
    agents=[return_explanation_SHAP_agent(), return_report_SHAP_agent()],
    tasks=[return_task_explanation_SHAP(), return_task_report_SHAP()],
      verbose=True
  )

  resultado_SHAP = crew_SHAP.kickoff(inputs={"shap_output": explicacion_SHAP(), "info_text": return_info_text(), "info_experto": return_info_experto()})
  return resultado_SHAP