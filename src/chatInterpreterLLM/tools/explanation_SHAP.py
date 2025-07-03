from src.chatInterpreterLLM.training.model import value_model, value_x_train
from src.chatInterpreterLLM.knowledge.info import pedir_caracteristicas
import shap
import json
import pandas as pd
import numpy as np

def explicacion_SHAP():

    model = value_model() 
    x_train = value_x_train()
    instancia = pedir_caracteristicas()
    instancia.columns = x_train.columns

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