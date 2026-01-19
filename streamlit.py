import streamlit as st
import requests


VALID_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "mistral-saba-24b",
    "qwen-qwq-32b"
]
VALID_RAG_TYPES = ["Basic", "Hybrid", "Advance"]
VALID_PREPROCESSING = ["Basic", "Custom"]
VALID_COHERE_OPTIONS_HYBRID = ["Allow Long Context Reordering", "No Reordering"]
VALID_COHERE_OPTIONS_ADVANCE = ["Allow Re-Ranking by Default"]

# Sidebar config
st.sidebar.header("Configuration")
model_name = st.sidebar.selectbox("Choose model", VALID_MODELS)
rag_type = st.sidebar.selectbox("Choose RAG Type", VALID_RAG_TYPES)
preprocessing_type = st.sidebar.selectbox("Choose Pre-processed Data", VALID_PREPROCESSING)
summary_flag = st.sidebar.checkbox("Activate Context Summarizer", value=False)
summary_flag = 1 if summary_flag else 0

cohere_flag = 0

if rag_type == "Hybrid":
    cohere_option = st.sidebar.selectbox("Select CoHere Reranker Option", VALID_COHERE_OPTIONS_HYBRID)
    cohere_flag = 1 if cohere_option == "Allow Long Context Reordering" else 0

elif rag_type == "Advance":
    cohere_option = st.sidebar.selectbox("CoHere Reranker Option", VALID_COHERE_OPTIONS_ADVANCE)
    cohere_flag = 1 if cohere_option == "Allow Re-Ranking by Default" else 0

# Chat title
st.markdown("<h2 style='text-align: center;'>🛡️ Compliance AI (GDPR & EU AI Act)</h2>", unsafe_allow_html=True)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat container
chat_container = st.container()

# User input
query = st.chat_input("Enter your question")

if query:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": query})

    # Make request to FastAPI
    try:
        response = requests.post(
            "http://127.0.0.1:8000/complaince/bot/ask",
            json={
                "query": query,
                "model_name": model_name,
                "rag_type": rag_type,
                "preprocessing_type": preprocessing_type,
                "cohere_hybrid": cohere_flag,
                "summary_flag": summary_flag
            },
        )
        bot_response = response.text.strip()
    except Exception as e:
        print(f"Error: {e}")
        bot_response = f"Something went wrong. Try changing the input parameters in side bar."

    # Store bot message
    st.session_state.messages.append({"role": "bot", "content": bot_response})

# Render chat
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])
