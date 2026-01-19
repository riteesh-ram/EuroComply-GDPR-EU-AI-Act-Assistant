from RAG.rag_chains import BasicRag, HybridRag, AdvanceRag

class Service:
    async def ask_bot(query: str, model_name: str, rag_type: str, preprocessing_type: str, cohere_hybrid: int=1, summary_flag: int=0):
        if preprocessing_type.lower() == "basic":
            collection_name = "gdpr_euAI_complainces_basic_preprocess"
        else:
            collection_name = "gdpr_euAI_complainces_custom_preprocess"

        if rag_type.lower() == "basic":
            response = await BasicRag.executor(query, model_name, collection_name, summary_flag)
        elif rag_type.lower() == "hybrid":
            response = await HybridRag.executor(query, model_name, collection_name, cohere_hybrid, summary_flag)
        elif rag_type.lower() == "advance":
            response = await AdvanceRag.executor(query, model_name, collection_name, cohere_hybrid, summary_flag)
        else:
            raise ValueError("Invalid RAG type. Choose 'basic', 'hybrid' or 'advance'.")
        
        return response
