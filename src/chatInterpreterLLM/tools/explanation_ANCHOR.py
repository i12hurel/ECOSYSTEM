from src.chatInterpreterLLM.training.model import value_model, value_x_train
from src.chatInterpreterLLM.knowledge.info import pedir_caracteristicas
from alibi.explainers import AnchorTabular
import json
import re
import pandas as pd
import numpy as np

def explicacion_ANCHOR():

    model = value_model() 
    x_train = value_x_train()
    instancia = pedir_caracteristicas()
    instancia.columns = x_train.columns


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
