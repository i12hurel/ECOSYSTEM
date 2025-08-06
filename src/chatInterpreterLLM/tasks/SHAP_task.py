from crewai import Task, Agent, LLM
from src.chatInterpreterLLM.agents.SHAP_agent import return_SHAP_agent
from src.chatInterpreterLLM.tools.explanation_SHAP import explicacion_SHAP

def return_task_SHAP(model, x_train, instancia, metadata, expert_notes):

    task_SHAP = Task(
        description="""Analyze the SHAP explanation JSON: {shap_output} and provide a natural language interpretation and generate an assessment report adding the additional information provided for the final expert.
                    **Follow these rules:**
                      1. Use ONLY features and weights from the JSON.
                      2. Relate exact values.
                      3. Extract and list the 3 features with the highest positive weight (weight > 0) and the 3 with the highest negative weight (weight < 0). For each, provide the feature name, the value obtained, and the weight, along with a brief explanation of why they positively or negatively impact the prediction.
                      4. Explain why the prediction was chosen over other classes.
                      5. Explicitly compare probabilities between classes.
                      6. Briefly summarize the prediction: indicate the predicted class and the relative probabilities of each class.
                      7. Incorporate additional information from the {metadata} if provided (e.g., title, sources, description, statistics) to contextualize the analysis.
                      8. Consider any user comments if they have been entered: {expert_notes}. If this observation is relevant for reinterpreting the model's results, take it into account in the final report.
                      9. Write the final report in natural language, addressed to the final expert, explaining in a clear and structured manner the factors influencing the prediction and how the information from the {expert_notes} supports the analysis.


                    **Output Format (strict, no backticks):**
                    ### Prediction
                        [The predicted class]

                    ### Rationale
                        [Comparison of probabilities and reason for choice]
                    
                    ### Top Positive Ratings:
                    - [Feature 1] = [Value]: +[Weight]. Explanation : [Reason why it's positive]
                    - [Feature 2] = [Value]: +[Weight]. Explanation : [Reason why it's positive]
                    - [Feature 3] = [Value]: +[Weight]. Explanation : [Reason why it's positive]

                    ### Top Negative Ratings:
                    - [Feature 1] = [Value]: -[Weight]. Explanation : [Reason why it's negative]
                    - [Feature 2] = [Value]: -[Weight]. Explanation : [Reason why it's negative]
                    - [Feature 3] = [Value]: -[Weight]. Explanation : [Reason why it's negative]

                    ### Final Report:
                        [Structured and coherent narrative that includes the explanation, the dataset context, and the user's observation, reformulated when necessary to resolve contradictions.]    """,
        expected_output="Coherent and context-aware final report integrating SHAP explanation, dataset info, and user context. Reformulate when contradictions appear.",
        agent=return_SHAP_agent(),
        inputs={
            "shap_output": explicacion_SHAP(model, x_train, instancia),
            "metadata": metadata, 
            "expert_notes": expert_notes} 
    )

    return task_SHAP
