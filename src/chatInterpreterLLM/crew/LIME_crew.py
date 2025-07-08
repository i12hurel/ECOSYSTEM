from crewai import Crew
from src.chatInterpreterLLM.agents.explanation_LIME_agent import return_explanation_LIME_agent
from src.chatInterpreterLLM.agents.report_LIME_agent import return_report_LIME_agent
from src.chatInterpreterLLM.tasks.explanation_LIME_task import return_task_explanation_LIME
from src.chatInterpreterLLM.tasks.report_LIME_task import return_task_report_LIME
from src.chatInterpreterLLM.knowledge.info import return_info_experto, return_info_text
from src.chatInterpreterLLM.tools.explanation_LIME import explicacion_LIME
def return_resultado_crew_LIME(model, x_train):
  crew_LIME = Crew(
    agents=[return_explanation_LIME_agent(), return_report_LIME_agent()],
    tasks=[return_task_explanation_LIME(model, x_train), return_task_report_LIME()],
      verbose=True
  )

  resultado_LIME = crew_LIME.kickoff(inputs={"lime_output": explicacion_LIME(model, x_train)})
  return resultado_LIME