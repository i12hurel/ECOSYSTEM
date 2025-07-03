from crewai import Task
from src.chatInterpreterLLM.knowledge.info import return_info_experto, return_info_text
from src.chatInterpreterLLM.agents.report_ANCHOR_agent import return_report_ANCHOR_agent

def return_task_report_ANCHOR():

    task_report_ANCHOR = Task(
        description="""Using the output from the Anchor explanation agent and the contextual information provided in "{info_text}", write a detailed report that:

            1. Summarizes the prediction and compares probabilities of each class.
            2. Interprets the anchor rule:
                - Explain what each condition means in terms of academic behavior or student performance.
                - Discuss the reliability of the prediction using the reported precision and coverage.
            3. Integrates additional dataset context from the info_text.
            4. Incorporates the user's expert observation: "{info_experto}"
            5. Reconciles any contradiction between the anchor rule and the user’s interpretation.
            6. Suggests possible interventions or actions if needed.
            7. Keeps the report focused, structured, and aligned with a decision-making context.

            **Suggested Output Format:**
            Prediction: [Predicted class]  
            Probabilities: [List each class and %]

            Anchor Conditions:
            - [Condition 1] → [Interpretation]
            - ...

            Precision: X% | Coverage: X%  
            Explanation Summary: [Brief summary of anchor logic]

            Final Report:
            [Full narrative integrating data, anchor logic, and expert insight.]""",
        expected_output="Oriented report combining anchor explanation, student profile, and expert context.",
        agent=return_report_ANCHOR_agent(),
        inputs={
            "info_text": return_info_text(), 
            "info_experto": return_info_experto()
        }
    )
    return task_report_ANCHOR