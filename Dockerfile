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
# We run this during the build so the database is ready inside the container
RUN python run_pipeline.py

# 7. Expose the port used by Streamlit
EXPOSE 8501

# 8. Setup the startup script
COPY run_services.sh .
RUN chmod +x run_services.sh

# 9. Start the application
CMD ["./run_services.sh"]