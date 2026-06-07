import streamlit as st
import asyncio
import os

try:
    from API.service.appService import Service
except ImportError as e:
    st.error(f"❌ Could not import Backend Service. Check file structure: {e}")

VALID_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]
VALID_RAG_TYPES = ["Hybrid", "Advance"]

st.sidebar.header("Configuration")
model_name = st.sidebar.selectbox("Choose model", VALID_MODELS)
rag_type = st.sidebar.selectbox("Choose RAG Type", VALID_RAG_TYPES)

st.markdown("<h2 style='text-align: center;'>🛡️ EuroComply: GDPR & EU AI Act Assistant</h2>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

chat_container = st.container()
query = st.chat_input("Ask about GDPR or EU AI Act...")

async def get_ai_response(user_query, model, r_type):
    try:
        response = await Service.ask_bot(
            query=user_query,
            model_name=model,
            rag_type=r_type,
        )
        return response
    except Exception as e:
        return f"❌ Backend Error: {str(e)}"

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.spinner("Thinking..."):
        try:
            bot_response = asyncio.run(get_ai_response(query, model_name, rag_type))
        except Exception as e:
            bot_response = f"Critical Error: {str(e)}"
    st.session_state.messages.append({"role": "bot", "content": bot_response})

with chat_container:
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            if role == "assistant" and isinstance(msg["content"], dict):
                data = msg["content"]
                st.markdown(data.get("response", "No response."))
                relevance = data.get("relevance score", "N/A")
                faithfulness = data.get("faithfulness score", "N/A")
                st.markdown(
                    f"<div style='margin-top:8px; font-size:0.8em; color:gray;'>"
                    f"📊 {relevance} &nbsp;|&nbsp; 🎯 {faithfulness}"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.write(msg["content"])
