from crewai import Agent, Crew, Task, LLM
from src.chatInterpreterLLM.env import enviroment
from src.chatInterpreterLLM.agents.gemini_agent import return_gemini_agent
from src.chatInterpreterLLM.tasks.gemini_interpreter_task import return_task_interpretar_intencion

def return_resultado_crew_gemini(usser_message, system_state):
  crew_gemini = Crew(
    agents=[return_gemini_agent(usser_message, system_state)],
    tasks=[return_task_interpretar_intencion(usser_message, system_state)],
      verbose=True
  )

  resultado_gemini = crew_gemini.kickoff()
  return resultado_gemini