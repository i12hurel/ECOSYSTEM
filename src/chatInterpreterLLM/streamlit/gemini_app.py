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
if "x_train" not in st.session_state:
    st.session_state.x_train = None
if "lime_output" not in st.session_state:
    st.session_state.lime_output = None
if "expert_notes" not in st.session_state:
    st.session_state.expert_notes = []
if "metadata" not in st.session_state:
    st.session_state.metadata = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "dataset_uploaded" not in st.session_state:
    st.session_state.dataset_uploaded = False
if "show_instance_uploader" not in st.session_state:
    st.session_state.show_instance_uploader = False
if "dataset_instance_uploaded" not in st.session_state:
    st.session_state.dataset_instance_uploaded = None
if "dataset_instance" not in st.session_state:
    st.session_state.dataset_instance = None

# --- Sidebar: Upload only ---
with st.sidebar:
    st.markdown("### 📂 Upload your data")
    dataset_file = st.file_uploader("📄 Dataset (.csv)", type=["csv"])
    metadata_file = st.file_uploader("🧾 Metadata (.txt)", type=["txt"])
    instance_file = st.file_uploader("📄 Instancia a explicar (.csv)", type=["csv"])


# --- Main content ---
st.title("💬 Machine Learning Explanation Assistant")

# Procesar dataset (una sola vez)
if dataset_file and not st.session_state.dataset_uploaded:
    st.session_state.dataset = pd.read_csv(dataset_file, delimiter=";")
    st.session_state.model, st.session_state.x_train = value_model(st.session_state.dataset)
    st.session_state.dataset_uploaded = True  # ✅ Solo mostrar una vez
    st.session_state.messages.append({"role": "assistant", "content": "✅ Dataset uploaded and model trained."})
    st.session_state.messages.append({"role": "assistant", "content": "__SHOW_DATASET__"})  # marcador especial


# Procesar metadata
if metadata_file and st.session_state.metadata is None:
    st.session_state.metadata = metadata_file.read().decode("utf-8")
    st.session_state.messages.append({"role": "assistant", "content": "🧠 Metadata loaded."})

if instance_file and st.session_state.dataset_instance_uploaded is None:
    st.session_state.dataset_instance = pd.read_csv(instance_file, delimiter=";")
    st.session_state.dataset_instance_uploaded = True
    st.session_state.messages.append({"role": "assistant", "content": "📊 Instance data loaded."})
    st.session_state.messages.append({"role": "assistant", "content": "__SHOW_INSTANCE__"})


# --- Display chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["content"] == "__SHOW_DATASET__":
            st.markdown("### 📊 Dataset preview:")
            st.dataframe(st.session_state.dataset)
        elif msg["content"] == "__SHOW_INSTANCE__":
            if st.session_state.dataset_instance_uploaded:
                st.markdown("### 🔍 Instance to explain:")
                st.dataframe(st.session_state.dataset_instance)
            else:
                st.markdown("⚠️ No instance available to display.")
        else:
            st.markdown(msg["content"])

# --- Chat input ---
user_input = st.chat_input("Write your instruction here...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)


    if "nueva instancia" in user_input.lower() or "subir una instancia" in user_input.lower():
        st.session_state.show_instance_uploader = True
    # Current system state
    system_state = {
        "dataset_loaded": st.session_state.dataset is not None,
        "model_trained": st.session_state.model is not None,
        "lime_explained": st.session_state.lime_output is not None,
        "metadata_present": st.session_state.metadata is not None,
        "notes_count": len(st.session_state.expert_notes),
        "expert_notes": st.session_state.expert_notes,
        "instance_uploaded": st.session_state.dataset_instance is not None,
    }

    # Crew execution
    response = return_resultado_crew_gemini(user_input, system_state)

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

