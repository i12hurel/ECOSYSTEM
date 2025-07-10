"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PASO 1: Experto pasa un modelo de aprendizaje y un dataset

PASO 2: (Opcionalmente) el experto pasa información sobre el dataset, bien a través del chat o bien a través de un fichero de metadatos

PASO 3: (Análisis de explicaciones por instancias) Mientras el experto desee:  # Este bucle no es el de interacción. Simplemente permite al experto seleccionar diferentes instancias para que sean explicadas

3.a. El experto observa las predicciones del modelo de aprendizaje sobre las instancias del dataset y selecciona una
   3.b. El sistema aplica LIME, SHAP, y Breakdown y muestra los resultados y el reporte interpretado
   3.c. (Interacción para las explicaciones de una instancia) Mientras el experto desee:
        3.c.I. El experto añade información a tener en cuenta para la explicación
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import streamlit as st
import pandas as pd
import json
from src.chatInterpreterLLM.training.model import value_model
from src.chatInterpreterLLM.tools.explanation_LIME import explicacion_LIME
from src.chatInterpreterLLM.crew.LIME_crew import return_resultado_crew_LIME

st.set_page_config(page_title="Chatbot IA", page_icon="🧠")

# Inicialización de estado
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "lime_output" not in st.session_state:
    st.session_state.lime_output = None
if "expert_notes" not in st.session_state:
    st.session_state.expert_notes = []
if "metadata" not in st.session_state:
    st.session_state.metadata = None

model = value_model()

st.title("💬 Chat de explicación ML")

# Subida de archivo
st.chat_message("assistant").markdown("👋 ¡Hola! Soy tu asistente para interpretar modelos de machine learning. Comparta el dataset sobre el que desea hacer predicciones.")
dataset_file = st.sidebar.file_uploader("📄 Dataset (.csv)", type=["csv"])
if dataset_file:
    st.session_state.dataset = pd.read_csv(dataset_file, delimiter=";")
    st.session_state.lime_output = None
    st.session_state.expert_notes = []
    st.chat_message("assistant").markdown("✅ Dataset cargado correctamente.")
    st.markdown("### 📊 Vista previa del dataset:")
    st.dataframe(st.session_state.dataset)
    with st.chat_message("assistant"):
        info_extra_input = st.chat_input("ℹ️ Si tienes información adicional sobre el dataset, puedes cargarla a continuación o añadirla directamente en el chat.")
        if info_extra_input:
            st.chat_message("user").markdown(info_extra_input)
            if info_extra_input.lower() == "no":
                st.chat_message("assistant").markdown("De acuerdo, no se cargará información adicional.")
            else:
                file_info = st.sidebar.file_uploader("📁 Metadatos (.json)", type=["json"])
                if file_info:
                    st.session_state.metadata = json.load(file_info)
                    st.chat_message("assistant").markdown("🧠 Metadatos cargados.")

st.chat_message("assistant").markdown("Ahor que?")
user_input = st.chat_input("Escribe una instrucción...")

if user_input:
    st.chat_message("user").markdown(user_input)

    if "explica" in user_input.lower():
        # Buscar índice en el mensaje
        import re
        match = re.search(r"\d+", user_input)
        if match and st.session_state.dataset is not None:
            index = int(match.group())
            lime_output = explicacion_LIME(
                model=model,
                dataset=st.session_state.dataset,
                instance_index=index
            )
            st.session_state.lime_output = lime_output
            resultado = return_resultado_crew_LIME(
                lime_output=lime_output,
                info_text=json.dumps(st.session_state.metadata) if st.session_state.metadata else "",
                info_experto="\n".join(st.session_state.expert_notes)
            )
            st.chat_message("assistant").markdown(f"✅ Explicación para la instancia {index}:\n```\n{resultado}\n```")
        else:
            st.chat_message("assistant").markdown("❌ No entendí qué fila quieres explicar.")
    elif "añade" in user_input.lower():
        # Añadir observación libre
        st.session_state.expert_notes.append(user_input)
        if st.session_state.lime_output:
            resultado = return_resultado_crew_LIME(
                lime_output=st.session_state.lime_output,
                info_text=json.dumps(st.session_state.metadata) if st.session_state.metadata else "",
                info_experto="\n".join(st.session_state.expert_notes)
            )
            st.chat_message("assistant").markdown(f"🔄 Explicación refinada:\n```\n{resultado}\n```")
        else:
            st.chat_message("assistant").markdown("ℹ️ Observación añadida. No hay una explicación previa activa.")
    else:
        st.chat_message("assistant").markdown("🧠 Puedes pedirme que explique una fila (ej: *Explica la instancia 3*) o añadir observaciones (ej: *Añade que la edad influye*).")
