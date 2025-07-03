from crewai import Task
from src.chatInterpreterLLM.knowledge.info import return_info_experto, return_info_text
from src.chatInterpreterLLM.agents.report_LIME_agent import return_report_LIME_agent

def return_task_report_LIME():
    task_report_LIME = Task(
        description="""Using the explanation agent's output and the additional information provided,
        generate an assessment report for the final expert.
            
        **Instructions:**
        1. Briefly summarize the prediction: indicate the predicted class and the relative probabilities of each class.
        2. Extract and list the 3 features with the highest positive weight (weight > 0) and the 3 with the highest negative weight (weight < 0). For each, provide the feature name, the value obtained, and the weight, along with a brief explanation of why they positively or negatively impact the prediction.
        3. Incorporate additional information from the info_text (e.g., title, sources, description, statistics) to contextualize the analysis.
        4. Write the final report in natural language, addressed to the final expert, explaining in a clear and structured manner the factors influencing the prediction and how the information from the info_text supports the analysis.
        5. Also consider the following user observation: {info_experto}. Si esta observación es relevante para reinterpretar los resultados del modelo, tenla en cuenta en el informe final.
        6. **Critically evaluate the coherence between the model's explanation and the user's input.**
        - If there is a perceived contradiction (e.g., strong engagement but low performance), clarify and explain it.
        - Reformulate the narrative to create a consistent explanation that merges model output with user context.
        7. Write the final report in a clear and structured way, balancing objective analysis with contextual understanding.

        **Suggested output format:**
        Prediction: [Predicted class]
        Rationale: [Comparison of probabilities and reason for choice]

        Top Positives:
        - [Feature 1] = [value]: +[weight]. [Brief explanation]
        - [Feature 2] = [value]: +[weight]. [Brief explanation]
        - [Feature 3] = [value]: +[weight]. [Brief explanation]

        Top Negatives:
        - [Feature 1] = [value]: -[weight]. [Brief explanation]
        - [Feature 2] = [value]: -[weight]. [Brief explanation]
        - [Feature 3] = [value]: -[weight]. [Brief explanation]


        Final Report:
        [Structured and coherent narrative that includes the explanation, the dataset context, and the user's observation, reformulated when necessary to resolve contradictions.]    """,
        expected_output="Coherent and context-aware final report integrating LIME explanation, dataset info, and user context. Reformulate when contradictions appear.",
        agent=return_report_LIME_agent(),
        inputs={
            "info_text" : return_info_text(),
            "info_experto": return_info_experto()
        }
    )
    return task_report_LIME