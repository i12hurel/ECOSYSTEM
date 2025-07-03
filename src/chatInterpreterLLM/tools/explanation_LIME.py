from src.chatInterpreterLLM.training.model import value_model, value_x_train
from src.chatInterpreterLLM.knowledge.info import pedir_caracteristicas
import re
from lime.lime_tabular import LimeTabularExplainer 
import json
import pandas as pd
import numpy as np

def explicacion_LIME():
    model = value_model() 
    x_train = value_x_train()
    instancia = pedir_caracteristicas()
    instancia.columns = x_train.columns

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
