from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

class ChromadDBConfig:
    
    def __init__(self, collection_name="gdpr_euAI_complainces_basic_preprocess"):
        self.embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        persist_dir = os.path.join(base_dir, "..", "chromadb")
        persist_dir = os.path.normpath(persist_dir)

        self.vectorstore = Chroma(collection_name=collection_name, embedding_function=self.embedding_function, persist_directory=persist_dir)
        
    async def get_retriever(self, k=5):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        
        return retriever
    
    async def get_retriever_with_metadata_filter(self, k=5, policy="GDPR"):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k, "filter": {"policy": policy}})
        
        return retriever
    
    async def retrieval_for_BM25(self, retriever, query):
        # semantic_docs = await retriever.aget_relevant_documents(query=query)
        semantic_docs = await retriever.ainvoke(input=query)
        
        return semantic_docs
    
    async def retrieval_for_advance_rag(self, retriever, query):
        # semantic_docs = await retriever.aget_relevant_documents(query=query)
        semantic_docs = await retriever.ainvoke(input=query)
        
        return semantic_docs
    
    async def add_documents(self, documents):
        if not self.vectorstore:
            raise ValueError("Vectorstore is required.")
        
        await self.vectorstore.aadd_documents(documents)

        return True