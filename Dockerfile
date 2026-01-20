# 1. Use Python 3.11 (Crucial for compatibility)
FROM python:3.11-slim

# 2. Set working directory
WORKDIR /app

# 3. Install system dependencies (Required for ChromaDB & Building)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the application code
COPY . .

# 6. Build the Knowledge Base (ChromaDB)
# Note: This runs during build. Ensure 'config/chroma_client.py' uses the new Singleton fix!
RUN python run_pipeline.py

# 7. Expose the port used by Streamlit
EXPOSE 8501

# 8. Start the application (Updated for Standalone Mode)
# We set the address to 0.0.0.0 so external users can connect
CMD ["streamlit", "run", "streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]