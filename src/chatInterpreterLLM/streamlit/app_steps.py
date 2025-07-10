# ✅ VERSIÓN 1 – Estructurada con pasos (step)
import streamlit as st
import pandas as pd
import json
from src.chatInterpreterLLM.training.model import value_model
from src.chatInterpreterLLM.crew.LIME_crew import return_resultado_crew_LIME

st.set_page_config(page_title="Chatbot IA", page_icon="💬")
st.title("💬 Chat de explicación ML")

# Estado de pasos y datos
if "step" not in st.session_state:
    st.session_state.step = 0
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "metadata" not in st.session_state:
    st.session_state.metadata = None
if "expert_notes" not in st.session_state:
    st.session_state.expert_notes = []
if "lime_output" not in st.session_state:
    st.session_state.lime_output = None

# Paso 0 – Introducción
if st.session_state.step == 0:
    with st.chat_message("assistant"):
        st.markdown("👋 ¡Hola! Soy tu asistente para interpretar modelos de ML.Sube un dataset para empezar.")
    st.session_state.step = 1

# Paso 1 – Subir dataset
model = ""
if st.session_state.step == 1:
    with st.chat_message("user"):
        dataset_file = st.file_uploader("📄 Dataset (.csv)", type=["csv"])
    if dataset_file:
        st.session_state.dataset = pd.read_csv(dataset_file, delimiter=";")
        with st.chat_message("assistant"):
            model, x_train = value_model(st.session_state.dataset)  # Entrenamos el modelo con el dataset
            st.session_state.model = model
            st.session_state.x_train = x_train
            st.markdown("✅ Dataset cargado correctamente.")
            st.markdown("### 📊 Vista previa del dataset:")
            st.dataframe(st.session_state.dataset)
            met = st.chat_input("¿Deseas añadir contexto? Puedes escribir o subir metadatos.")
            if met:
                st.chat_message("user").markdown(met)
                if met.lower() == "no":
                    st.chat_message("assistant").markdown("De acuerdo, no se añadirá contexto adicional.")
                    st.session_state.step = 3
                else:
                    st.session_state.step = 2

# Paso 2 – Añadir contexto
if st.session_state.step == 2:
    with st.chat_message("user"):
        text_info = st.text_area("📝 Texto contextual (opcional)")
        file_info = st.file_uploader("📁 Metadatos (.json)", type=["json"])
    if text_info or file_info:
        if text_info:
            st.session_state.expert_notes.append(text_info)
        if file_info:
            st.session_state.metadata = json.load(file_info)
        with st.chat_message("assistant"):
            st.markdown("✅ Contexto recibido. Ahora selecciona una instancia a explicar.")
        st.session_state.step = 3

#HASTA AQUI BIEN



# Paso 3 – Seleccionar y explicar instancia
if st.session_state.step == 3:
    with st.chat_message("user"):
        index = st.number_input("🔍 Índice a explicar", 0, len(st.session_state.dataset)-1, step=1)
        confirmar = st.button("🧠 Explicar")
    if confirmar:
        instance = st.session_state.dataset.iloc[[index]]
        resultado_lime = return_resultado_crew_LIME(st.session_state.model, st.session_state.x_train)
        st.session_state.lime_output = resultado_lime

        with st.chat_message("assistant"):
            st.markdown(f"✅ Explicación generada:\n```{resultado_lime}\n```")
            st.markdown("¿Deseas refinar con nueva observación o analizar otra?")
        st.session_state.step = 4

# Paso 4 – Refinar o reiniciar
if st.session_state.step == 4:
    with st.chat_message("user"):
        nueva_obs = st.text_area("✏️ Añadir observación (opcional)")
        refinar = st.button("🔄 Refinar")
        otra = st.button("📊 Otra instancia")
    if refinar and nueva_obs:
        st.session_state.expert_notes.append(nueva_obs)
        resultado = return_resultado_crew_LIME(
            lime_output=st.session_state.lime_output,
            info_text=json.dumps(st.session_state.metadata) if st.session_state.metadata else "",
            info_experto="\n".join(st.session_state.expert_notes)
        )
        with st.chat_message("assistant"):
            st.markdown(f"✅ Explicación refinada:\n```{resultado}\n```")
    if otra:
        st.session_state.step = 3

st.sidebar.button("🔁 Reiniciar", on_click=lambda: st.session_state.clear())