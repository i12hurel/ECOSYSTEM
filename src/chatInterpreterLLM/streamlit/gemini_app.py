# app_gemini_chat.py – Conversational interface using Gemini as intelligent controller
import streamlit as st
import pandas as pd
import json

from src.chatInterpreterLLM.training.model import value_model
from src.chatInterpreterLLM.crew.gemini_crew import return_resultado_crew_gemini

st.set_page_config(page_title="ML Explanation Chatbot", page_icon="💬")

# Initialize session state
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "model" not in st.session_state:
    st.session_state.model = None
if "lime_output" not in st.session_state:
    st.session_state.lime_output = None
if "expert_notes" not in st.session_state:
    st.session_state.expert_notes = []
if "metadata" not in st.session_state:
    st.session_state.metadata = None

# --- Sidebar: Dataset and metadata upload ---
with st.sidebar:
    st.markdown("### 📂 Upload your data")
    dataset_file = st.file_uploader("📄 Dataset (.csv)", type=["csv"])
    metadata_file = st.file_uploader("🧾 Metadata (.txt)", type=["txt"])

    if dataset_file:
        st.session_state.dataset = pd.read_csv(dataset_file, delimiter=";")
        st.session_state.model, st.session_state.x_train = value_model(st.session_state.dataset)
        st.chat_message("assistant").markdown("✅ Dataset uploaded and model trained.")
        st.chat_message("assistant").dataframe(st.session_state.dataset)

    if metadata_file:
        st.session_state.metadata = metadata_file.read().decode("utf-8")
        st.chat_message("assistant").markdown("🧠 Metadata loaded.")

# --- Chat Interface ---
st.title("💬 Machine Learning Explanation Assistant")

user_input = st.chat_input("Write your instruction here...")
if user_input:
    st.chat_message("user").markdown(user_input)

    # Build current system state
    system_state = {
        "dataset_loaded": st.session_state.dataset is not None,
        "model_trained": st.session_state.model is not None,
        "lime_explained": st.session_state.lime_output is not None,
        "metadata_present": st.session_state.metadata is not None,
        "notes_count": len(st.session_state.expert_notes),
    }

    # Call the Gemini reasoning crew
    crew = return_resultado_crew_gemini(user_input, system_state)

    st.chat_message("assistant").markdown(crew)
