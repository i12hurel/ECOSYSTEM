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


class RecibirInstanciaDatasetTool(BaseTool):
    name: str = "recibir_instancia_dataset"
    description: str = "Pide al usuario que suba una instancia en un archivo .csv para su explicación."

    def _run(self, **kwargs):
        return "📤 Por favor, sube un archivo `.csv` con una única fila representando la nueva instancia."


class EjecutarCrewLIMEToolInstanciaDataset(BaseTool):
    name: str = "ejecutar_crew_LIME_dataset"
    description: str = "Ejecuta la explicación de LIME para una instancia del dataset existente."

    def _run(self, index: int = None, **kwargs):
        try:
            if st.session_state.dataset is None or st.session_state.model is None:
                return "❌ No hay dataset ni modelo cargado."
            if index is None:
                return "⚠️ Por favor, proporciona un índice válido."

            if index < 0 or index >= len(st.session_state.dataset):
                return f"⚠️ Índice fuera de rango. El dataset tiene {len(st.session_state.dataset)} filas."

            # Paso 1: preparar instancia
            target_column = st.session_state.dataset.columns[-1]
            instance = st.session_state.dataset.iloc[[index]]
            if target_column in instance.columns:
                instance = instance.drop(columns=[target_column])
            else:
                return f"❌ La columna objetivo '{target_column}' no está en la instancia seleccionada."

            # Paso 2: preparar x_train si es necesario
            st.session_state.x_train = st.session_state.dataset.drop(columns=[target_column])

            # Paso 3: ejecutar explicación
            resultado_lime = return_resultado_crew_LIME(st.session_state.model, st.session_state.x_train, instance)
            st.session_state.lime_output = resultado_lime
            return f"🧠 Explicación de la instancia {index}:\n```{resultado_lime}```"

        except Exception as e:
            return f"❌ Error en la tool 'ejecutar_crew_LIME_dataset': {str(e)}"


class EjecutarCrewLIMEToolInstanciaNueva(BaseTool):
    name: str = "ejecutar_crew_LIME_nueva"
    description: str = "Ejecuta la explicación de LIME para una nueva instancia pasada por el usuario."

    def _run(self, instance_data: dict = None, **kwargs):
        if st.session_state.model is None:
            return "❌ No hay modelo cargado."

        if instance_data is None:
            return "⚠️ Por favor, proporciona los datos de la nueva instancia como un diccionario."

        try:
            instancia = pd.DataFrame([instance_data])
            st.session_state.x_train = instancia

            lime_output = explicacion_LIME(
                model=st.session_state.model,
                x_train=st.session_state.x_train,
                instance=instancia
            )
            st.session_state.lime_output = lime_output

            resultado = return_resultado_crew_LIME(
                lime_output=lime_output,
                info_text=json.dumps(st.session_state.metadata) if st.session_state.metadata else "",
                info_experto="\n".join(st.session_state.expert_notes)
            )

            return f"🧠 Explicación de la instancia proporcionada:\n```{resultado}```"
        except Exception as e:
            return f"❌ No se pudo procesar la instancia: {str(e)}"


class CargarMetadataTool(BaseTool):
    name: str = "cargar_metadata"
    description: str = "Permite al usuario subir un archivo de metadata (.txt) desde la barra lateral para dar más infromación sobre el dataset."

    def _run(self, **kwargs):
        txt_file = st.sidebar.file_uploader("📎 Subir metadata (.txt)", type=["txt"], key="metadata_file")
        if txt_file:
            try:
                content = txt_file.read().decode("utf-8")
                st.session_state.metadata = content
                return "🧠 Metadata cargada correctamente desde el archivo."
            except Exception as e:
                return f"❌ Error al leer el archivo .txt: {e}"
        return "⚠️ Por favor, sube un archivo de metadata en formato .txt desde la barra lateral."

class AñadirContextoTool(BaseTool):
    name: str = "añadir_contexto"
    description: str = "Añade una observación o información adicional escrita por el usuario para enriquecer el análisis."

    def _run(self, content: str = None, **kwargs):
        if not content:
            return "⚠️ No se proporcionó ningún texto para añadir al contexto."

        if "expert_notes" not in st.session_state:
            st.session_state.expert_notes = []
        st.session_state.expert_notes.append(content)
        return "✅ Comentario añadido al contexto del sistema correctamente."



class ReiniciarTool(BaseTool):
    name: str = "reiniciar_sesion"
    description: str = "Reinicia toda la sesión y limpia el estado."

    def _run(self, **kwargs):
        st.session_state.clear()
        return "🔄 Sesión reiniciada. Puedes comenzar desde cero."
