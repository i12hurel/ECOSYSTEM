from crewai import Task
from src.chatInterpreterLLM.knowledge.info import return_info_experto, return_info_text
from src.chatInterpreterLLM.agents.interpreter_agent import return_integrador_agent
from src.chatInterpreterLLM.crew.results_crew import return_resultado_crew_LIME, return_resultado_crew_SHAP, return_resultado_crew_ANCHOR
import json

def return_task_integration():

    task_integration = Task(
        description=""" Analyze ONLY the provided final reports from LIME, SHAP, and ANCHOR below. Generate a concise integrated report strictly based on these actual reports. 

        Provided Reports:
        ---- LIME REPORT ----
        {resultado_LIME}

        ---- SHAP REPORT ----
        {resultado_SHAP}

        ---- ANCHOR REPORT ----
        {resultado_ANCHOR}

        Instructions:
        - Explicitly copy predictions and probabilities exactly as presented in each report (LIME, SHAP, ANCHOR). DO NOT INVENT OR CHANGE THEM.
        - Identify ONLY factors explicitly mentioned by at least two methods.
        - Clearly highlight explicit discrepancies if any appear.
        - Integrate briefly the provided expert context ({info_experto}) to justify the significance of identified common factors.
        - Provide a clear conclusion based ONLY on the explicitly identified common factors and expert context.

        Format:
        Quick reference predictions (EXACT COPIES):
        - LIME: [Predicted class] ([Probability])
        - SHAP: [Predicted class] ([Probability])
        - ANCHOR: [Predicted class] ([Probability])

        Common factors identified:
        - [Factor]: Brief explanation.

        Significant discrepancies (only if explicitly present):
        - [Discrepant factor]: Explanation.

        Integration of expert context:
        - Brief justification from expert context.

        Final integrated conclusion:
        - Concise conclusion and specific recommendations.""",
        expected_output="""Exact predictions and probabilities explicitly from original reports, strictly no assumptions or invented data. Report based solely on explicitly provided texts.""",
        agent=return_integrador_agent(),
        inputs={
            "resultado_LIME": return_resultado_crew_LIME(),
            "resultado_SHAP": return_resultado_crew_SHAP(),
            "resultado_ANCHOR": json.dumps(return_resultado_crew_ANCHOR().model_dump(), indent=2),
            "info_experto": return_info_experto(),
            "info_text": return_info_text()
        }
    )
    return task_integration
