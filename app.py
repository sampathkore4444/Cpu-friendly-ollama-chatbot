import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

st.set_page_config(page_title="Qwen2.5 Chat", page_icon="🤖")
st.title("🤖 Qwen2.5:1.5B Chat (Streamlit Cloud)")

# -----------------------------
# Cache model loading
# -----------------------------
@st.cache_resource
def load_model():
    model_name = "Qwen/Qwen-2.5B-chat"  # Hugging Face model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    chat_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1  # CPU, use device=0 for GPU if available
    )
    return chat_pipeline

chatbot = load_model()

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
    # Append user message
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
    # Generate response (streaming-style)
    # -----------------------------
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response_text = ""

        # Generate tokens step by step
        for output in chatbot(conversation, max_new_tokens=200, do_sample=True, temperature=0.7):
            # The pipeline returns full text, we can display it all at once
            response_text = output["generated_text"][len(conversation):]  # only new text
            message_placeholder.markdown(response_text)

        # Append assistant message
        st.session_state.messages.append({"role": "assistant", "content": response_text})