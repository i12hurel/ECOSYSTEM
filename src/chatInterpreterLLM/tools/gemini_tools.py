from crewai.tools import BaseTool
import streamlit as st
import pandas as pd
import json
from src.chatInterpreterLLM.crew.LIME_crew import return_resultado_crew_LIME
import io
import matplotlib.pyplot as plt


class RequestDatasetTool(BaseTool):
    name: str = "request_dataset"
    description: str = "Asks the user to upload a dataset to train the model."

    def _run(self, **kwargs):
        return "📥 Please upload a `.csv` dataset from the sidebar."


class RequestInstanceTool(BaseTool):
    name: str = "request_instance"
    description: str = "Asks the user to upload a new instance as a `.csv` file with a single row."

    def _run(self, **kwargs):
        return "📤 Please upload a `.csv` file with a single row from the sidebar to generate the explanation."


class RunLIMEDatasetInstanceTool(BaseTool):
    name: str = "run_lime_dataset_instance"
    description: str = "Runs the LIME explanation for an instance from the existing dataset."

    def _run(self, index: int = None, **kwargs):
        try:
            if st.session_state.dataset is None or st.session_state.model is None:
                return "❌ No dataset or model loaded."
            if index is None:
                return "⚠️ Please provide a valid index."

            if index < 0 or index >= len(st.session_state.dataset):
                return f"⚠️ Index out of range. The dataset contains {len(st.session_state.dataset)} rows."

            instance = st.session_state.dataset.iloc[[index]]

            target_column = st.session_state.dataset.columns[-1]
            if target_column in instance.columns:
                instance = instance.drop(columns=[target_column])
            else:
                return f"❌ Target column '{target_column}' not found in the selected instance."

            st.session_state.x_train = st.session_state.dataset.drop(columns=[target_column])

            expert_notes = st.session_state.get("expert_notes", [])
            #expert_notes = "\n\nExpert notes:\n" + "\n".join(f"- {note}" for note in notes) if notes else ""

            metadata = st.session_state.get("metadata", "No metadata provided.")

            lime_result = return_resultado_crew_LIME(st.session_state.model, st.session_state.x_train, instance, metadata, expert_notes)
            st.session_state.lime_output = lime_result
            return f"🧠 Explanation for instance {index}:\n```{lime_result}```"

        except Exception as e:
            return f"❌ Error in 'run_lime_dataset_instance' tool: {str(e)}"


class RunLIMENewInstanceTool(BaseTool):
    name: str = "run_lime_new_instance"
    description: str = "Ejecuta LIME sobre la instancia cargada por el usuario."

    def _run(self, **kwargs):
        if st.session_state.model is None:
            return "❌ No hay modelo cargado."

        df = st.session_state.get("dataset_instance")
        if df is None or df.shape[0] != 1:
            return "⚠️ No se encontró una instancia válida cargada."

        instance_data = df.iloc[0].to_dict()

        try:
            df_instance = pd.DataFrame([instance_data])
            target_column = st.session_state.dataset.columns[-1]

            if target_column in df_instance.columns:
                df_instance = df_instance.drop(columns=[target_column])

            st.session_state.x_train = st.session_state.dataset.drop(columns=[target_column])

            expert_notes = st.session_state.get("expert_notes", [])
            #expert_notes = "\n\nExpert notes:\n" + "\n".join(f"- {note}" for note in notes) if notes else ""

            metadata = st.session_state.get("metadata", "No metadata provided.")

            lime_result = return_resultado_crew_LIME(st.session_state.model, st.session_state.x_train, df_instance, metadata, expert_notes)
            st.session_state.lime_output = lime_result
            return f"✅ Explicación generada:\n\n```{lime_result}```"

        except Exception as e:
            return f"❌ Error generando la explicación: {str(e)}"


class UploadMetadataTool(BaseTool):
    name: str = "upload_metadata"
    description: str = "Allows the user to upload a `.txt` metadata file from the sidebar to provide additional dataset information."

    def _run(self, **kwargs):
        return "⚠️ Please upload a metadata file in `.txt` format from the sidebar."


class AddContextTool(BaseTool):
    name: str = "add_context"
    description: str = "Adds a user-provided note or additional information to enrich the analysis, but only when explicitly indicated."

    def _run(self, content: str = None, **kwargs):
        if not content:
            return "⚠️ No text provided to add to context."

        if "expert_notes" not in st.session_state:
            st.session_state.expert_notes = []
            
        st.session_state.expert_notes.append(content)
        return "✅ Comment successfully added to system context."


class ResetSessionTool(BaseTool):
    name: str = "reset_session"
    description: str = "Resets the entire session and clears the state."

    def _run(self, **kwargs):
        st.session_state.clear()
        return "🔄 Session reset. You can start over."

