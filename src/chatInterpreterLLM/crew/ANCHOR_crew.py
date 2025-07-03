from crewai import Crew
from src.chatInterpreterLLM.agents.explanation_ANCHOR_agent import return_explanation_ANCHOR_agent
from src.chatInterpreterLLM.agents.report_ANCHOR_agent import return_report_ANCHOR_agent
from src.chatInterpreterLLM.tasks.explanation_ANCHOR_task import return_task_explanation_ANCHOR
from src.chatInterpreterLLM.tasks.report_ANCHOR_task import return_task_report_ANCHOR
from src.chatInterpreterLLM.knowledge.info import return_info_experto, return_info_text
from src.chatInterpreterLLM.tools.explanation_ANCHOR import explicacion_ANCHOR

def return_resultado_crew_ANCHOR():
  crew_ANCHOR = Crew(
      agents=[return_explanation_ANCHOR_agent(), return_report_ANCHOR_agent()],
      tasks=[return_task_explanation_ANCHOR(), return_task_report_ANCHOR()],
      verbose=True
    )

  resultado_ANCHOR = crew_ANCHOR.kickoff(inputs={"anchor_output": explicacion_ANCHOR(), "info_text": return_info_text(), "info_experto": return_info_experto()})

  return resultado_ANCHOR