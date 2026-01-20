import os
import chromadb

# 1. Disable Telemetry to reduce log noise
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

class ChromadDBConfig:
    
    def __init__(self, collection_name="gdpr_euAI_complainces_basic_preprocess"):
        print(f"🔍 DEBUG: Initializing ChromadDBConfig for collection: '{collection_name}'")
        self.collection_name = collection_name
        
        print("🔍 DEBUG: Loading Embedding Model (MiniLM-L6)...")
        self.embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        
        self.vectorstore = None 
        
    async def get_vectorstore(self):
        """
        Lazy loader with logs to track connection status.
        """
        if self.vectorstore is None:
            print(f"🔌 DEBUG: Vectorstore is None. Attempting to connect to '{self.collection_name}'...")
            
            try:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                persist_dir = os.path.join(base_dir, "..", "chromadb")
                persist_dir = os.path.normpath(persist_dir)
                
                print(f"📂 DEBUG: Database persistence directory: {persist_dir}")

                # Initialize Chroma
                self.vectorstore = Chroma(
                    collection_name=self.collection_name, 
                    embedding_function=self.embedding_function, 
                    persist_directory=persist_dir
                )
                print("✅ DEBUG: ChromaDB Connection Successful!")
                
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
    
    async def retrieval_for_BM25(self, retriever, query):
        return await retriever.ainvoke(input=query)
    
    async def retrieval_for_advance_rag(self, retriever, query):
        return await retriever.ainvoke(input=query)
    
    async def add_documents(self, documents):
        print(f"📝 DEBUG: add_documents called with {len(documents)} chunks.")
        
        # Ensure connection
        vs = await self.get_vectorstore()
        
        # FIX: Explicit check for None (handles the 'truthiness' bug)
        if vs is None:
            print("❌ DEBUG: Vectorstore is None after initialization attempt.")
            raise ValueError("Critical Error: Could not initialize Vectorstore.")
        
        print(f"💾 DEBUG: Saving {len(documents)} documents to {self.collection_name}...")
        try:
            await vs.aadd_documents(documents)
            print("✅ DEBUG: Documents saved successfully.")
        except Exception as e:
            print(f"❌ DEBUG: Error during aadd_documents: {e}")
            raise e
            
        return True