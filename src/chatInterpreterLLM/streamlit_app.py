# estilo_whatsapp.py
import streamlit as st

st.set_page_config(page_title="Chat estilo WhatsApp", layout="centered")

css = """
<style>
.user-bubble {
    background-color: #dcf8c6;
    padding: 10px;
    border-radius: 10px;
    margin-left: auto;
    margin-bottom: 10px;
    max-width: 70%;
}
.assistant-bubble {
    background-color: #f1f0f0;
    padding: 10px;
    border-radius: 10px;
    margin-right: auto;
    margin-bottom: 10px;
    max-width: 70%;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

if "chat" not in st.session_state:
    st.session_state.chat = [
        ("assistant", "👋 ¡Hola! ¿Cómo estás hoy?")
    ]
    st.session_state.respondido = False

# Mostrar conversación con estilos personalizados
for role, content in st.session_state.chat:
    bubble_class = "assistant-bubble" if role == "assistant" else "user-bubble"
    st.markdown(f"<div class='{bubble_class}'>{content}</div>", unsafe_allow_html=True)

# Entrada del usuario
if not st.session_state.respondido:
    respuesta = st.text_input("✍️ Tu respuesta:")
    if respuesta:
        st.session_state.chat.append(("user", respuesta))
        st.session_state.respondido = True
        st.experimental_rerun()
