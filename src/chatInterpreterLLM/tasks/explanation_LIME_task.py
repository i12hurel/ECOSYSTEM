from crewai import Task
from src.chatInterpreterLLM.agents.explanation_LIME_agent import return_explanation_LIME_agent
from src.chatInterpreterLLM.tools.explanation_LIME import explicacion_LIME

def return_task_explanation_LIME():
  task_explanation_LIME = Task(
      description="""Analyze the LIME explanation JSON below and provide a natural language interpretation. 
                    **Follow these rules:**
                      1. Use ONLY features and weights from the JSON.
                      2. Relate exact values.
                      3. List TOP 3 features NEGATIVELY affecting prediction (weights < 0) and their impact.
                      4. List TOP 3 features POSITIVELY affecting it (weights > 0) and their impact.
                      5. Explain why the prediction was chosen over other classes.
                      6. Explicitly compare probabilities between classes.
                      
                      **Output Format:**
                    Prediction: [The class predicted]
                    Why?: [Explanation based on probabilities]
                    
                    Top Positive Ratings:
                    - [Feature 1] = [Value]: +[Weight]. Explanation : [Reason why it's positive]
                    - [Feature 2] = [Value]: +[Weight]. Explanation : [Reason why it's positive]
                    - [Feature 3] = [Value]: +[Weight]. Explanation : [Reason why it's positive]

                    Top Negative Ratings:
                    - [Feature 1] = [Value]: -[Weight]. Explanation : [Reason why it's negative]
                    - [Feature 2] = [Value]: -[Weight]. Explanation : [Reason why it's negative]
                    - [Feature 3] = [Value]: -[Weight]. Explanation : [Reason why it's negative]

                    LIME Output: 
                    {lime_output}""",
      expected_output="""Clear text explaining key factors. Structured text with 3 positive and 3 negative contributors + prediction rationale.
                      Text with format: [Feature] = [Value]: [Weight Explanation]. """,
      agent=return_explanation_LIME_agent(),
      inputs={"lime_output": explicacion_LIME()}, 
  )
  
  return task_explanation_LIME