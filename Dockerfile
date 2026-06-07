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

# 6. Build the Knowledge Base (ChromaDB) from JSONL/CSV sources
RUN python run_pipeline.py

# 7. Expose the port used by Streamlit (HF Spaces requires 7860)
EXPOSE 7860

# 8. Start the application (Updated for Standalone Mode)
# We set the address to 0.0.0.0 so external users can connect
CMD ["streamlit", "run", "main_ui.py", "--server.port=7860", "--server.address=0.0.0.0"]