import streamlit as st
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:1.5b"

st.set_page_config(page_title="Qwen2.5 Chat", page_icon="🤖")

st.title("🤖 Qwen2.5:1.5B Chat (Ollama)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
prompt = st.chat_input("Type your message...")

if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build conversation context
    conversation = ""
    for m in st.session_state.messages:
        role = "User" if m["role"] == "user" else "Assistant"
        conversation += f"{role}: {m['content']}\n"

    payload = {
        "model": MODEL_NAME,
        "prompt": conversation + "Assistant:",
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
        },
    }

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            answer = response.json()["response"]

            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
