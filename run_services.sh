#!/bin/bash

# 1. Start the Backend (FastAPI) in the background
# We bind to 0.0.0.0 to ensure internal visibility
python app.py &

# 2. Wait 15 seconds for the database and model to load
echo "Waiting for Backend to initialize..."
sleep 15

# 3. Start the Frontend (Streamlit)
# We strictly bind to port 8501 and address 0.0.0.0 for Render
echo "Starting Streamlit..."
streamlit run streamlit.py --server.port 8501 --server.address 0.0.0.0