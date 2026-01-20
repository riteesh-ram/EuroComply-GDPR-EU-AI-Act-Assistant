import os
import chromadb

# 1. Disable Telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# --- 🚀 GLOBAL SINGLETON (Build-Safe) ---
# We use a simple global variable because this file is used by
# 'run_pipeline.py' during the Docker build, where Streamlit is not running.
_SHARED_EMBEDDING_MODEL = None

def get_shared_embedding_model():
    global _SHARED_EMBEDDING_MODEL
    if _SHARED_EMBEDDING_MODEL is None:
        print("🧠 DEBUG: Loading Embedding Model (Global Singleton)...")
        _SHARED_EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return _SHARED_EMBEDDING_MODEL

class ChromadDBConfig:
    
    def __init__(self, collection_name="gdpr_euAI_complainces_basic_preprocess"):
        print(f"🔍 DEBUG: Initializing Config for: '{collection_name}'")
        self.collection_name = collection_name
        
        # Use the global singleton
        self.embedding_function = get_shared_embedding_model()
        
        self.vectorstore = None 
        
    async def get_vectorstore(self):
        if self.vectorstore is None:
            try:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                persist_dir = os.path.join(base_dir, "..", "chromadb")
                persist_dir = os.path.normpath(persist_dir)
                
                self.vectorstore = Chroma(
                    collection_name=self.collection_name, 
                    embedding_function=self.embedding_function, 
                    persist_directory=persist_dir
                )
                print(f"✅ DEBUG: Connected to ChromaDB: {self.collection_name}")
                
            except Exception as e:
                print(f"❌ DEBUG: Failed to connect to ChromaDB: {e}")
                raise e
            
        return self.vectorstore

    async def get_retriever(self, k=5):
        vs = await self.get_vectorstore()
        return vs.as_retriever(search_kwargs={"k": k})
    
    async def get_retriever_with_metadata_filter(self, k=5, policy="GDPR"):
        vs = await self.get_vectorstore()
        return vs.as_retriever(search_kwargs={"k": k, "filter": {"policy": policy}})
    
    async def add_documents(self, documents):
        vs = await self.get_vectorstore()
        if vs is None:
            raise ValueError("Critical Error: Could not initialize Vectorstore.")
        await vs.aadd_documents(documents)
        return True