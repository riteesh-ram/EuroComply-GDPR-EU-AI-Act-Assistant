import streamlit as st
import asyncio
import os

# --- DIRECT IMPORT (Bypassing FastAPI to save memory) ---
try:
    from API.service.appService import Service
except ImportError as e:
    st.error(f"❌ Could not import Backend Service. Check file structure: {e}")

# Configuration Constants
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

# --- SIDEBAR CONFIGURATION ---
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

# --- MAIN PAGE ---
st.markdown("<h2 style='text-align: center;'>🛡️ Compliance AI (Direct Mode)</h2>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

chat_container = st.container()
query = st.chat_input("Enter your question")

# --- DIRECT BACKEND CALL FUNCTION ---
async def get_ai_response(user_query, model, r_type, p_type, c_flag, s_flag):
    """
    Directly calls the Service layer, bypassing FastAPI request overhead.
    """
    try:
        # Call the static method directly on the class
        # Matches the signature in API/service/appService.py
        response = await Service.ask_bot(
            query=user_query,
            model_name=model,
            rag_type=r_type,
            preprocessing_type=p_type,
            cohere_hybrid=c_flag,
            summary_flag=s_flag
        )
        return response
        
    except Exception as e:
        return f"❌ Backend Error: {str(e)}"

# --- CHAT LOGIC ---
if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Thinking..."):
        # We use asyncio.run to execute the async backend logic
        try:
            bot_response = asyncio.run(
                get_ai_response(
                    query, 
                    model_name, 
                    rag_type, 
                    preprocessing_type, 
                    cohere_flag, 
                    summary_flag
                )
            )
        except Exception as e:
            bot_response = f"Critical Error: {str(e)}"

    st.session_state.messages.append({"role": "bot", "content": bot_response})

# Render chat
with chat_container:
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "assistant"
        st.chat_message(role).write(msg["content"])