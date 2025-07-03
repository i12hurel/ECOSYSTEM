from crewai import Task
from src.chatInterpreterLLM.agents.explanation_ANCHOR_agent import return_explanation_ANCHOR_agent
from src.chatInterpreterLLM.tools.explanation_ANCHOR import explicacion_ANCHOR

def return_task_explanation_ANCHOR():

    task_explanation_ANCHOR = Task(
        description="""Analyze the Anchor explanation JSON below and provide a detailed natural language interpretation based on the anchor rules.

            **Instructions:**
            1. Clearly state the predicted class and the associated probabilities for each class.
            2. List the exact anchor rule conditions (e.g., feature > value) and explain what each one means in context.
            3. Emphasize the meaning of precision and coverage:
                - Precision: What does it mean that the model is X% confident when the rule holds?
                - Coverage: How rare is this rule across the dataset?
            4. Discuss how the anchor conditions ensure the model's confidence in the prediction.
            5. DO NOT create additional reasoning not found in the anchor explanation. Stay faithful to the rule-based format.

            **Output Format:**
            Prediction: [Predicted class]  
            Probabilities: [Dropout: X%, Enrolled: Y%, Graduate: Z%]

            Anchor Rule:
            - [Condition 1] → [What it means for the student]
            - [Condition 2] → [What it means for the student]
            - ...

            Precision: X%  
            Coverage: X%

            Explanation: [Summarize what the rule is telling us about why the model made this prediction.]

            Anchor Output:
            {anchor_output}""",
        expected_output="""Textual interpretation explaining the predicted class, anchor rule, condition meanings, precision, and coverage. Must reflect the anchor structure and be grounded strictly in the data.""",
        agent=return_explanation_ANCHOR_agent(),
        inputs={"anchor_output": explicacion_ANCHOR()}
)
    return task_explanation_ANCHOR