from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import chromadb

# DISABLE TELEMETRY to stop the "capture() takes 1 positional argument" warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"

load_dotenv()

class ChromadDBConfig:
    
    def __init__(self, collection_name="gdpr_euAI_complainces_basic_preprocess"):
        self.collection_name = collection_name
        # Keep the model consistent
        self.embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        self.vectorstore = None 
        
    async def get_vectorstore(self):
        """
        Lazy loader: Connects to the database only when asked.
        """
        if self.vectorstore is None:
            print(f"🔌 Connecting to ChromaDB collection: {self.collection_name}...")
            
            base_dir = os.path.dirname(os.path.abspath(__file__))
            # Navigate up one level to root, then into chromadb
            persist_dir = os.path.join(base_dir, "..", "chromadb")
            persist_dir = os.path.normpath(persist_dir)

            self.vectorstore = Chroma(
                collection_name=self.collection_name, 
                embedding_function=self.embedding_function, 
                persist_directory=persist_dir
            )
            print("✅ ChromaDB Connected.")
            
        return self.vectorstore

    async def get_retriever(self, k=5):
        vs = await self.get_vectorstore()
        retriever = vs.as_retriever(search_kwargs={"k": k})
        return retriever
    
    async def get_retriever_with_metadata_filter(self, k=5, policy="GDPR"):
        vs = await self.get_vectorstore()
        retriever = vs.as_retriever(search_kwargs={"k": k, "filter": {"policy": policy}})
        return retriever
    
    async def retrieval_for_BM25(self, retriever, query):
        semantic_docs = await retriever.ainvoke(input=query)
        return semantic_docs
    
    async def retrieval_for_advance_rag(self, retriever, query):
        semantic_docs = await retriever.ainvoke(input=query)
        return semantic_docs
    
    async def add_documents(self, documents):
        # FORCE connection before adding
        vs = await self.get_vectorstore()
        
        if not vs:
            raise ValueError("Critical Error: Could not initialize Vectorstore.")
        
        await vs.aadd_documents(documents)
        return True