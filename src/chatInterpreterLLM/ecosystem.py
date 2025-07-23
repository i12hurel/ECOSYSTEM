import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import re
import lime
import shap
from alibi.explainers import AnchorTabular
import time
from datetime import datetime
from litellm.exceptions import RateLimitError


from crewai import Agent, Task, Crew, LLM
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import os
import requests
from dotenv import load_dotenv
from pathlib import Path

# Cargar las variables del archivo .env
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

import warnings
warnings.filterwarnings('ignore', message="X does not have valid feature names")

def pedir_caracteristicas():
    caracteristicas = {
        "Marital status": [4],
        "Application mode": [39],
        "Application order": [1],
        "Course": [9130],
        "Daytime/evening attendance\t": [1],
        "Previous qualification": [1],
        "Previous qualification (grade)": [132],
        "Nacionality": [2],
        "Mother's qualification": [5],
        "Father's qualification": [3],
        "Mother's occupation": [5],
        "Father's occupation": [4],
        "Admission grade": [110],
        "Displaced": [1],
        "Educational special needs": [0],
        "Debtor": [0],
        "Tuition fees up to date": [1],
        "Gender": [0],
        "Scholarship holder": [1],
        "Age at enrollment": [31],
        "International": [0],
        "Curricular units 1st sem (credited)": [0],
        "Curricular units 1st sem (enrolled)": [6],
        "Curricular units 1st sem (evaluations)": [8],
        "Curricular units 1st sem (approved)": [1],
        "Curricular units 1st sem (grade)": [0],
        "Curricular units 1st sem (without evaluations)": [0],
        "Curricular units 2nd sem (credited)": [0],
        "Curricular units 2nd sem (enrolled)": [5],
        "Curricular units 2nd sem (evaluations)": [16],
        "Curricular units 2nd sem (approved)": [1],
        "Curricular units 2nd sem (grade)": [8],
        "Curricular units 2nd sem (without evaluations)": [0],
        "Unemployment rate": [11.1],
        "Inflation rate": [0.6],
        "GDP": [2.02]
    }
    return pd.DataFrame(caracteristicas, index=[0])


#ENTRENAMOS AL MODELO

current_dir = Path(__file__).resolve().parent
csv_path = current_dir.parent / 'chatInterpreterLLM' / 'knowledge' / 'clasificadores' / 'datasets' / 'abandono_escolar.csv'

data = pd.read_csv(csv_path, delimiter=';')

x = data.drop('Target', axis=1)
y = data['Target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)    

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)


#CARGAMOS LA INSATNCIA DEL EXPERTO

instancia = pedir_caracteristicas()
instancia.columns = x_train.columns

current_dir = Path(__file__).resolve().parent
info_text_path = current_dir / 'knowledge' / 'info_text.txt'
with open(info_text_path, 'r', encoding='utf-8') as file:
    info_text = file.read()


#info_experto = "I think the second-quarter subjects are slightly more complex."

info_experto = "The student failed more subjects in the second semester because he was going through a difficult family situation due to the loss of a close relative. This situation affected him emotionally, causing him to be more absent, distracted, and less focused in class, which could explain the decline in his academic performance during that period."



def explicacion_LIME(instancia, model=model):

    probabilidades = model.predict_proba(instancia)[0] #devuelve las probabilidades de cada clase
    pred_idx = np.argmax(probabilidades) # devuelve el índice de la clase con mayor probabilidad
    clase_predicha = model.classes_[pred_idx] #recoge la clase con mayor probabilidad

    explainer = LimeTabularExplainer(
    training_data = x_train.values, 
    class_names = model.classes_,
    feature_names=x_train.columns.tolist(), 
    mode = 'classification',
    random_state=42
    )
    
    exp = explainer.explain_instance(instancia.values[0], model.predict_proba, num_features=15, num_samples=5000)
    exp_list = exp.as_list()

    valoraciones = {
        "positive": [],
        "negative": []
    }

    explanation = {cls: [] for cls in model.classes_}

    for feat_str, weight in exp_list:
        tokens = re.split(r'<=|>=|<|>', feat_str)
        feature_name = next((token.strip() for token in tokens if token.strip() in x_train.columns), None)
        if not feature_name:
            continue
        
        explanation[clase_predicha].append({
            "feature": feature_name,
            "value": instancia.iloc[0][feature_name],
            "weight": round(weight, 2)
        })
            
        feature_value = instancia.iloc[0][feature_name]
        entry = {
            "feature": feature_name,
            "value": feature_value,
            "weight": round(weight, 2)
        }
        
        if weight >= 0:
            valoraciones["positive"].append(entry)
        else:
            valoraciones["negative"].append(entry)

      
    valoraciones["positive"].sort(key=lambda x: abs(x["weight"]), reverse=True)
    valoraciones["negative"].sort(key=lambda x: abs(x["weight"]), reverse=True)

    prob_dict = {label: round(prob, 2) for label, prob in zip(model.classes_, probabilidades)}
    
    return json.dumps({
        "predicted_class": clase_predicha,
        "probabilities": prob_dict,
        "valoraciones": valoraciones,
        "explanation": explanation,
    }, indent=2)



def explicacion_SHAP(instancia):

    predicted_class = model.predict(instancia)[0]

    class_index = np.where(model.classes_ == predicted_class)[0][0]
    clase_predicha = model.classes_[class_index] #recoge la clase con mayor probabilidad

    explainer = shap.Explainer(model, x_train)

    shap_values = explainer(instancia)

    valuesSHAP = np.array(shap_values.values[0]) 
    shap_values_for_class = valuesSHAP[:, class_index] 

    base_valuesSHAP = np.array(shap_values.base_values[0]) 
    base_values_for_class = base_valuesSHAP[class_index]

    dataSHAP=shap_values.data[0] 
    
    explanation = {cls: [] for cls in model.classes_}
    valoraciones = {
        "positive": [],
        "negative": []
    }

    for i, weight in enumerate(shap_values_for_class):
        feature = shap_values.feature_names[i]
        value = instancia.iloc[0][feature]
        entry = {
            "feature": feature,
            "value": value,
            "weight": round(weight, 4)
        }

        explanation[clase_predicha].append(entry)

        if weight >= 0:
            valoraciones["positive"].append(entry)
        else:
            valoraciones["negative"].append(entry)

    valoraciones["positive"].sort(key=lambda x: abs(x["weight"]), reverse=True)
    valoraciones["negative"].sort(key=lambda x: abs(x["weight"]), reverse=True)

    prob_dict = {label: round(prob, 2) for label, prob in zip(model.classes_, model.predict_proba(instancia)[0])}

    return json.dumps({
        "predicted_class": clase_predicha,
        "probabilities": prob_dict,
        "valoraciones": valoraciones,
        "explanation": explanation,
    }, indent=2)


def explicacion_ANCHOR(instancia):

    probabilidades = model.predict_proba(instancia)[0]
    pred_idx = np.argmax(probabilidades)
    clase_predicha = model.classes_[pred_idx]

    explainer = AnchorTabular(
        predictor=model.predict,
        feature_names=x_train.columns.tolist()
    )

    explainer.fit(x_train.values)

    exp = explainer.explain(instancia.to_numpy()[0])

    rules = exp.data.get("anchor", [])

    output = {
        "prediction": str(clase_predicha),
        "anchor": rules, #siempre que la instancia cumpla esas condiciones, el modelo predice la clase predicha.
        "precision": round(exp.precision, 2), #porcertaje de acierto
        "coverage": round(exp.coverage, 2)
    }

    detailed_conditions = []
    for condition in rules:
        match = re.match(r"([\w\s'\"()\[\]\-]+)", condition)
        if match:
            feature_name = match.group(1).strip()
            if feature_name in instancia.columns:
                detailed_conditions.append({
                    "feature": feature_name,
                    "value": instancia.iloc[0][feature_name],
                    "rule_string": condition
                })

    prob_dict = {label: round(prob, 2) for label, prob in zip(model.classes_, probabilidades)}

    return json.dumps({
        "predicted_class": clase_predicha,
        "probabilities": prob_dict,
        "anchor_explanation": output,
        "conditions": detailed_conditions #condiciones de anchor que cumple esta instancia
    }, indent=2)

    
lime_output = explicacion_LIME(instancia)
shap_output = explicacion_SHAP(instancia)
anchor_output = explicacion_ANCHOR(instancia)


agentes_LLM = LLM(
        model="gemini/gemini-2.0-flash-lite",
        temperature=0.2,
        key = API_KEY,
        #vertex_credentials=vertex_credentials_json
)

#AGENTES EXPLICADORES
explanation_LIME_agent = Agent(
    role='AI Model Explainer LIME',
    goal='Generate clear and accurate explanations of model predictions highlighting both positive and negative contributors using LIME',
    backstory="""You are an expert in interpreting machine learning models with focus on feature contributions. 
                Your task is to analyze LIME explanations and translate them into 
                natural language, strictly following the provided data without 
                adding speculative information.""",
    verbose=True,
    allow_delegation=False,
    #tool = explicacion_LIME,
    llm = agentes_LLM 
)

explanation_SHAP_agent = Agent(
    role='AI Model Explainer SHAP',
    goal='Generate clear and accurate SHAP-based explanations of model predictions, focusing on the contribution of each feature according to SHAP values.',
    backstory="""You are an expert in interpreting machine learning models using SHAP, with a focus on feature-level contribution explanations. 
                Your task is to analyze SHAP explanations and translate them into 
                natural language, strictly following the provided data without 
                adding speculative information.""",
    verbose=True,
    allow_delegation=False,
    #tool = explicacion_SHAP,
    llm = agentes_LLM 
)

explanation_ANCHOR_agent = Agent(
    role='AI Model Explainer ANCHOR',
    goal='Generate structured natural language interpretations of anchor-based model predictions.',
    backstory="""You specialize in interpreting machine learning models using Anchor explanations. 
                You transform logical rule-based outputs into clear, human-readable insights that explain 
                how a model prediction is "anchored" by certain key features. Your job is to strictly 
                interpret the anchor rule provided, explain the conditions, and highlight their meaning 
                in the context of the student’s data, without making assumptions beyond the given output.""",
    verbose=True,
    allow_delegation=False,
    #tool=explicacion_ANCHOR,
    llm=agentes_LLM 
)

#AGENTES REPORTE FINAL

report_LIME_agent = Agent(
    role='AI Report Generator LIME',
    goal='Generate an assessment report integrating the LIME explanation with additional information from the info_text to provide the final expert with a clear summary of the factors influencing the prediction.',
    backstory="""You are an expert in analyzing machine learning models and preparing assessment reports. Your task is to combine the explanation obtained with LIME (in JSON format already translated into natural language) with additional details about the database (e.g., title, sources, description, statistics, etc.).

    You must:
    1. Extract the three most important positive and negative factors from the explanation.
    2. Compare the probability of the predicted class versus the other classes.
    3. Incorporate and contextualize the information from the database, highlighting relevant data that can help interpret the prediction.

    Generate a clear and structured report for the final expert. For example, in the context of dropping out of school, you could write:
    'As you can see, he hasn't shown up for anything in the first semester, and we're already in the second. He probably won't show up for any subjects in the second semester either. If he had at least submitted Activity L3, he could have passed. Everything points to him dropping out of school.'
    Similarly, adapt the language to the context of the problem and the information available.""",
    verbose=True,
    allow_delegation=False,
    llm=agentes_LLM 
)

report_SHAP_agent = Agent(
    role='AI Report Generator SHAP',
    goal='Generate an assessment report integrating the SHAP explanation with additional information from the info_text to provide the final expert with a clear summary of the factors influencing the prediction.',
    backstory="""You are an expert in analyzing machine learning models and preparing assessment reports. Your task is to combine the SHAP-based explanation (in JSON format already structured) with additional details about the database (e.g., title, sources, description, statistics, etc.).
    You must:
    1. Extract the three most important positive and negative factors from the explanation.
    2. Compare the probability of the predicted class versus the other classes.
    3. Incorporate and contextualize the information from the database, highlighting relevant data that can help interpret the prediction.

    Generate a clear and structured report for the final expert. For example, in the context of dropping out of school, you could write:
    'As you can see, he hasn't shown up for anything in the first semester, and we're already in the second. He probably won't show up for any subjects in the second semester either. If he had at least submitted Activity L3, he could have passed. Everything points to him dropping out of school.'
    Similarly, adapt the language to the context of the problem and the information available.""",
    verbose=True,
    allow_delegation=False,
    llm=agentes_LLM 
)

report_ANCHOR_agent = Agent(
    role='AI Report Generator ANCHOR',
    goal='Generate an assessment report integrating the Anchor explanation with additional contextual information to provide the final expert with a clear summary of the rules behind the prediction.',
    backstory="""You are an expert in analyzing machine learning model predictions using Anchor explanations and preparing structured reports.
    Your task is to combine the rule-based Anchor explanation (already structured in JSON format) with additional contextual information about the dataset (e.g., source, background, metrics, etc.).

    You must:
    1. Summarize the anchor rule and explain each condition clearly in relation to the input instance.
    2. Report the predicted class and compare its probability against the others.
    3. Interpret the meaning of the rule's precision and coverage in terms of model reliability and generalizability.
    4. Contextualize the analysis using additional information (from info_text) to explain why the rule might be valid or insightful.
    5. Incorporate the user's expert observation (if any), and evaluate how it aligns or contrasts with the model's logic.
    
    Generate a clear, well-structured, and critical report that integrates technical reasoning with contextual interpretation, suitable for supporting real-world decision-making in any domain.""",
    verbose=True,
    allow_delegation=False,
    llm=agentes_LLM 
)


#TAREAS EXPLICACION

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
    agent=explanation_LIME_agent,
    inputs={"lime_output": lime_output}, 
)

task_explanation_SHAP = Task(
    description="""Analyze the SHAP explanation JSON below and provide a natural language interpretation based explicitly on SHAP values. 
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

                  SHAP Output: 
                  {shap_output}""",
    expected_output="""Clear text explaining key factors. Structured text with 3 positive and 3 negative contributors + prediction rationale.
                    Text with format: [Feature] = [Value]: [Weight Explanation]. """,
    agent=explanation_SHAP_agent,
    inputs={"shap_output": shap_output},
)

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
    agent=explanation_ANCHOR_agent,
    inputs={"anchor_output": anchor_output}, 
)

#TAREAS REPORTE FINAL
task_report_LIME = Task(
    description="""Using the explanation agent's output and the additional information provided,
    generate an assessment report for the final expert.
        
    **Instructions:**
    1. Briefly summarize theI'm going to give you some extra information about the dataset. prediction: indicate the predicted class and the relative probabilities of each class.
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
    agent=report_LIME_agent,
    inputs={
        "info_text" : info_text,
        "info_experto": info_experto
    }
)

task_report_SHAP = Task(
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
    expected_output="Coherent and context-aware final report integrating SHAP explanation, dataset info, and user context. Reformulate when contradictions appear.",
    agent=report_SHAP_agent,
    inputs={
        "info_text" : info_text,
        "info_experto": info_experto
    }
)

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
    agent=report_ANCHOR_agent,
    inputs={
        "info_text": info_text, 
        "info_experto": info_experto
    }
)

crew_LIME = Crew(
  agents=[explanation_LIME_agent, report_LIME_agent],
  tasks=[task_explanation_LIME, task_report_LIME],
    verbose=True
)

resultado_LIME = crew_LIME.kickoff(inputs={"lime_output": lime_output, "info_text": info_text, "info_experto": info_experto})
print(f"RESULTADO LIME: {resultado_LIME}")

time.sleep(60)

crew_SHAP = Crew(
    agents=[explanation_SHAP_agent, report_SHAP_agent],
    tasks=[task_explanation_SHAP, task_report_SHAP],
    verbose=True
)

resultado_SHAP = crew_SHAP.kickoff(inputs={"shap_output": shap_output, "info_text": info_text, "info_experto": info_experto})
print(f"RESULTADO SHAP: {resultado_SHAP}")

time.sleep(60)

crew_ANCHOR = Crew(
    agents=[explanation_ANCHOR_agent, report_ANCHOR_agent],
    tasks=[task_explanation_ANCHOR, task_report_ANCHOR],
    verbose=True
)

resultado_ANCHOR = crew_ANCHOR.kickoff(inputs={"anchor_output": anchor_output, "info_text": info_text, "info_experto": info_experto})
print(f"RESULTADO ANCHOR: {resultado_ANCHOR}")

time.sleep(60)

#ESCRIBIMOS
with open("resultados_crew_LIME.txt", "w", encoding="utf-8") as f:
    #f.write(f"RESULTADO LIME\n")
    f.write(f"{resultado_LIME}\n")

with open("resultados_crew_SHAP.txt", "w", encoding="utf-8") as f:
    #f.write(f"RESULTADO SHAP:\n")
    f.write(f"{resultado_SHAP}\n")

with open("resultados_crew_ANCHOR.txt", "w", encoding="utf-8") as f:
    #f.write(f"RESULTADO ANCHOR:\n")
    f.write(f"{resultado_ANCHOR}\n")

#AGENTE INTEGRADOR
integrador_agent = Agent(
    role='AI Integrated Explanation Expert',
    goal='Integrate interpretations from LIME, SHAP, and ANCHOR into a unified conclusion.',
    backstory="""You are skilled in synthesizing different machine learning interpretability techniques 
                into coherent, actionable insights. You compare feature importance consistency across methods
                and explain clearly why the combined findings lead to a final conclusion.""",
    verbose=True,
    allow_delegation=False,
    llm=agentes_LLM
)

#TAREA INTEGRADORA
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
    agent=integrador_agent,
    inputs={
        "resultado_LIME": json.dumps(resultado_LIME.model_dump(), indent=2),
        "resultado_SHAP": json.dumps(resultado_SHAP.model_dump(), indent=2),
        "resultado_ANCHOR": json.dumps(resultado_ANCHOR.model_dump(), indent=2),
        "info_experto": info_experto,
        "info_text": info_text
    }
)

time.sleep(60)

crew_integration = Crew(
    agents=[integrador_agent],
    tasks=[task_integration],
    verbose=True
)

resultado_integrador = crew_integration.kickoff(
    inputs={ "resultado_LIME": json.dumps(resultado_LIME.model_dump(), indent=2),
        "resultado_SHAP": json.dumps(resultado_SHAP.model_dump(), indent=2),
        "resultado_ANCHOR": json.dumps(resultado_ANCHOR.model_dump(), indent=2),
        "info_experto": info_experto,
        "info_text": info_text }
)

print(f"RESULTADO INTEGRADOR: {resultado_integrador}")

with open("resultado_integrador.txt", "w", encoding="utf-8") as f:
    f.write(f"{resultado_integrador}\n")
