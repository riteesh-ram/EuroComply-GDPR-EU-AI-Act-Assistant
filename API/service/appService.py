from RAG.rag_chains import HybridRag, AdvanceRag

class Service:
    async def ask_bot(query: str, model_name: str, rag_type: str):
        collection_name = "gdpr_euAI_complainces_custom_preprocess"

        if rag_type.lower() == "hybrid":
            response = await HybridRag.executor(query, model_name, collection_name)
        elif rag_type.lower() == "advance":
            response = await AdvanceRag.executor(query, model_name, collection_name)
        else:
            raise ValueError("Invalid RAG type. Choose 'hybrid' or 'advance'.")

        return response
