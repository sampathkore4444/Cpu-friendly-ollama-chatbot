import streamlit as st
import subprocess
import json

MODEL_NAME = "qwen2.5:1.5b"

st.set_page_config(page_title="Qwen2.5 Chat", page_icon="🤖")
st.title("🤖 Qwen2.5:1.5B Chat (Ollama)")

# -----------------------------
# Cache model download on first run
# -----------------------------
@st.cache_resource
def download_model(model_name):
    """Download the model using Ollama CLI"""
    try:
        subprocess.run(
            ["ollama", "pull", model_name],
            check=True,
            capture_output=True
        )
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to download model: {e.stderr.decode()}")
        return False

download_model(MODEL_NAME)

# -----------------------------
# Chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# User input
# -----------------------------
prompt = st.chat_input("Type your message...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build conversation context
    conversation = ""
    for m in st.session_state.messages:
        role = "User" if m["role"] == "user" else "Assistant"
        conversation += f"{role}: {m['content']}\n"
    conversation += "Assistant:"

    # -----------------------------
    # Call Ollama CLI to generate response
    # -----------------------------
    try:
        result = subprocess.run(
            ["ollama", "generate", MODEL_NAME, "--json", "--prompt", conversation],
            capture_output=True,
            text=True,
            check=True
        )
        output = json.loads(result.stdout)
        answer = output.get("response", "")
    except Exception as e:
        st.error(f"Failed to generate response: {e}")
        answer = "Oops! Something went wrong."

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})