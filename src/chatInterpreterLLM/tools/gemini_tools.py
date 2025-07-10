from crewai.tools import BaseTool
import streamlit as st
import pandas as pd
import json
from src.chatInterpreterLLM.crew.LIME_crew import return_resultado_crew_LIME
from src.chatInterpreterLLM.tools.explanation_LIME import explicacion_LIME

class PedirDatasetTool(BaseTool):
    name: str = "pedir_dataset"
    description: str = "Pide al usuario que suba un dataset para entrenar el modelo."
    
    def _run(self, **kwargs):
        return "📥 Por favor, sube un archivo .csv desde la barra lateral."

class EjecutarCrewLIMETool(BaseTool):
    name: str = "ejecutar_crew_LIME"
    description: str = "Ejecuta la explicación de LIME para una instancia del dataset."

    def _run(self, index: int = None, **kwargs):
        if st.session_state.dataset is None or st.session_state.model is None:
            return "❌ No hay dataset ni modelo cargado."
        if index is None:
            return "⚠️ I did not receive a valid index. Please specify which instance to explain."
        target_column = st.session_state.dataset.columns[-1]  # o pon el nombre directamente
        instancia = st.session_state.dataset.iloc[[index]].drop(columns=[target_column])
        resultado_lime = return_resultado_crew_LIME(st.session_state.model, st.session_state.x_train, instancia)
        st.session_state.lime_output = resultado_lime
        return f"🧠 Explicación de la instancia {index}:\n```{resultado_lime}```"

class AñadirContextoTool(BaseTool):
    name: str ="añadir_contexto"
    description: str ="Añade una observación del experto al sistema para mejorar la interpretación."
    
    def _run(self, content=None, **kwargs):
        if content:
            st.session_state.expert_notes.append(content)
            return "✅ Observación añadida correctamente."
        return "❌ No se proporcionó contenido."

class ReiniciarTool(BaseTool):
    name: str ="reiniciar_sesion"
    description: str = "Reinicia toda la sesión y limpia el estado."

    def _run(self, **kwargs):
        st.session_state.clear()
        return "🔄 Sesión reiniciada. Puedes comenzar desde cero."
