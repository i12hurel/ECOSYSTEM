from src.chatInterpreterLLM.training.model import value_model
from src.chatInterpreterLLM.knowledge.info import pedir_caracteristicas
import re
from lime.lime_tabular import LimeTabularExplainer 
import json
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import streamlit as st
from crewai.tools import BaseTool

class ExplanationLIMETool(BaseTool):
    name: str = "Generate LIME Explanation"
    description: str = "Generates a LIME explanation for a given instance. Returns the predicted class, probabilities, and the most influential features."

    def _run(self):
        """
        Args:
            instancia: A DataFrame containing the instance to explain. If None, it will use the instance from st.session_state.
        Returns:
            JSON with predicted class, probabilities, positive/negative contributions, and feature explanations.
        Notes:
            - The model and x_train are retrieved from st.session_state.
            - This tool is designed to be used by an agent without needing to manually pass all arguments.
        """

        model, x_train = st.session_state.model, st.session_state.x_train
        instancia = st.session_state.instance
            
        print(f"{instancia}")
        instancia.columns = x_train.columns

        # Predict probabilities and determine the predicted class
        probabilidades = model.predict_proba(instancia)[0]
        pred_idx = np.argmax(probabilidades)
        clase_predicha = model.classes_[pred_idx]

        # Initialize LIME explainer
        explainer = LimeTabularExplainer(
            training_data=x_train.values,
            class_names=model.classes_,
            feature_names=x_train.columns.tolist(),
            mode='classification',
            random_state=42,
        )

        # Generate LIME explanation for the instance
        exp = explainer.explain_instance(
            instancia.values[0],
            model.predict_proba,
            num_features=15,
            num_samples=5000,
            labels=[pred_idx]
        )
        exp_list = exp.as_list(label=pred_idx)

        # Prepare structured explanation
        valoraciones = {"positive": [], "negative": []}
        explanation = {cls: [] for cls in model.classes_}

        # Parse features and weights from the LIME output
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

            entry = {
                "feature": feature_name,
                "value": instancia.iloc[0][feature_name],
                "weight": round(weight, 2)
            }

            # Separate features into positive and negative contributions
            if weight >= 0:
                valoraciones["positive"].append(entry)
            else:
                valoraciones["negative"].append(entry)

        # Sort features by absolute importance
        valoraciones["positive"].sort(key=lambda x: abs(x["weight"]), reverse=True)
        valoraciones["negative"].sort(key=lambda x: abs(x["weight"]), reverse=True)

        # Prepare probabilities as a dict
        prob_dict = {label: round(prob, 2) for label, prob in zip(model.classes_, probabilidades)}

        # Return the explanation as JSON
        return json.dumps({
            "predicted_class": clase_predicha,
            "probabilities": prob_dict,
            "valoraciones": valoraciones,
            "explanation": explanation,
        }, indent=2)
    

class RecibirInstanciaDatasetTool(BaseTool):
    name: str ="Select Dataset Instance by Index"
    description: str = "Selects a specific row from the dataset by index and stores it as a single-row DataFrame in session_state.instancia for further processing (e.g., for LIME explanations)."
    
    def _run(self, index: int):
        """
        Args:
            index (int): Row index to extract from the dataset.
        Returns:
            JSON with confirmation and the selected instance.
        """
        if index < 0 or index >= len(st.session_state.dataset):
                return f"⚠️ Index out of range. The dataset contains {len(st.session_state.dataset)} rows."

        instance = st.session_state.dataset.iloc[[index]]
        #instance = instance.iloc[:, 1:]
        instance = instance.drop(columns=[st.session_state.dataset.columns[-1]], errors='ignore')  # Drop target column if it exists
        st.session_state.instance = instance

        return instance
    
class RecibirInstanciaNuevaTool(BaseTool):
    name: str = "Receive New Instance"
    description: str = "Receives a new instance as a single-row DataFrame for further processing (e.g., for LIME explanations)."

    def _run(self):
        """
        Args:
            instance (pd.DataFrame): A single-row DataFrame representing the new instance.
        Returns:
            JSON with confirmation and the received instance.
        """
        if st.session_state.instance_uploaded == True:
            instance = st.session_state.new_instance
            instance = instance.drop(columns=[st.session_state.dataset.columns[-1]], errors='ignore')  # Drop target column if it exists
            st.session_state.instance = instance
        else:
            return "⚠️ No instance uploaded or available in session state."

        return instance
    
class RecibirMetadataTool(BaseTool):
    name: str = "Receive Metadata"
    description: str = "Receives metadata information to provide context for the dataset."

    def _run(self):
        """
        Args:
            metadata (str): Metadata information as a string.
        Returns:
            JSON with confirmation and the received metadata.
        """
        if st.session_state.metadata is None:
            return "⚠️ No metadata provided or available in session state."
        
        metadata = st.session_state.metadata
        return metadata
    
class RecibirNotasExpertosTool(BaseTool):
    name: str = "Receive Expert Notes"
    description: str = "Receives expert notes to provide additional context for the explanations."

    def _run(self):
        """
        Args:
            notes (list): List of expert notes.
        Returns:
            JSON with confirmation and the received expert notes.
        """
        if not st.session_state.expert_notes:
            return "⚠️ No expert notes provided or available in session state."

        st.session_state.expert_notes.append(st.session_state.user_input)

        return st.session_state.expert_notes

