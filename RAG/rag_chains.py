from config.chroma_client import ChromadDBConfig
from config.groq_client import Groq
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableMap
from langchain_community.retrievers import BM25Retriever
# Notice we removed ".ensemble" and ".contextual_compression"
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
import json
import re

captured_context = {}

# Function to capture context flowing through the chain
def capture_context(inputs):
    captured_context['context'] = inputs['context']
    return inputs

class Prompts:

    async def get_chat_prompt():
        chat_template = """
        Answer the question based only on the following context:
        {context}

        Question: {question}

        Instructions:
        - If the context clearly contains information that answers the question, provide a concise and accurate response.
        - If the context contains information that partially answers the question, provide a response that reflects that partial information.

        Important:
        - Do not use any external knowledge.
        - Do not make assumptions.
        - Only use the context provided.
        - Dont give reasoning.
        
        Answer:
        """
        chat_prompt = ChatPromptTemplate.from_template(chat_template)
        return chat_prompt
        
    async def get_validator_prompt():
        validator_template = """
        You are a validator. Determine if the given answer is valid based on the user's question.
        - If the answer has enough relevancy to the question, output the answer itself.
        - If the answer mentions something about not enough context, output: "Sorry, I cannot answer that at the moment.".
        - Dont give any reasoning.

        Question: {question}
        Answer: {answer}
        """
        validator_prompt = ChatPromptTemplate.from_template(validator_template)
        return validator_prompt

    async def get_summary_prompt():
        summary_template = """
        Given the context summarize it in a detailed manner preserving all key points and relevant information.
        Context: {context}

        Instructions
        - Only output the context summary

        Summary:
        """
        summary_prompt = ChatPromptTemplate.from_template(summary_template)
        return summary_prompt
    
    async def get_relevance_prompt():
        relevance_template = """
        You are a STRICT expert RAG evaluator.
        Evaluate how well the answer addresses the question. 
        - Is the answer directly answering what was asked?
        - Is it useful and on-topic?

        Question:
        {question}

        Answer:
        {answer}

        Respond with: "Relevance Score: X" where X is a number from 1 (irrelevant) to 10 (highly relevant).
        Be strict with scoring.
        Dont give reasoning.
        Only Output the relevance score.
        """
        relevance_prompt = ChatPromptTemplate.from_template(relevance_template)
        return relevance_prompt
    
    async def get_faithfulness_prompt():
        faithfulness_template = """
        You are a STRICT expert RAG evaluator.
        Evaluate whether the answer is factually consistent with the context above. 
        - If all information in the answer is supported by the context, it is faithful.
        - If there are hallucinations or fabricated content, it is unfaithful.

        Context:
        {context}

        Answer:
        {answer}

        Respond with: "Faithfulness Score: X" where X is a number from 1 (unfaithful) to 10 (completely faithful).
        Be strict with scoring.
        Dont give reasoning.
        Only Output the relevance score.
        """
        faithfulness_prompt = ChatPromptTemplate.from_template(faithfulness_template)
        return faithfulness_prompt

    async def get_metadata_policy_name_prompt():
        metadata_policy_name_template = """
        Given the user question, determine the policy name ("GDPR" or "EU AI Act") only if it is mentioned in the question.

        Question:
        {question}

        Instructions
        - Only output the policy name.
        - If the policy name is mentioned in the question, respond with: "Policy: GDPR" or "Policy: EU AI Act".
        - If the policy name is not mentioned in the question, respond with: "Policy: Unknown".
        - Do not provide any reasoning or additional information.
        
        Policy: 
        """
        metadata_policy_name_prompt = ChatPromptTemplate.from_template(metadata_policy_name_template)
        return metadata_policy_name_prompt
    
    async def get_self_query_prompt():
        self_query_template = """
        You are a RAG expert.
        Given the user question, generate 5 distinct and optimized retrieval queries.

        Question:
        {question}

        Instructions
        - Dont give reasoning.
        - Only output the retrieval queries and originlay query in json format.
        - Each query should be distinct and optimized for retrieval.
        
        Output format

        "original query":
        "query 1":
        "query 2":
        "query 3":
        "query 4":
        "query 5":
        """
        self_query_prompt = ChatPromptTemplate.from_template(self_query_template)
        return self_query_prompt

    def extract_page_content(docs):
        return [doc.page_content for doc in docs]

class Evaluations:

    async def evaluate(context, answer, question):
        relevance_prompt = await Prompts.get_relevance_prompt()
        faithfulness_prompt = await Prompts.get_faithfulness_prompt()

        groq = Groq(model="llama-3.3-70b-versatile")
        llm = await groq.get_llm()

        relevance_chain = (
            {"question": RunnablePassthrough(), "answer": RunnablePassthrough()}
            | relevance_prompt
            | llm
            | StrOutputParser()
        )

        faithfulness_chain = (
            {"context": RunnablePassthrough(), "answer": RunnablePassthrough()}
            | faithfulness_prompt
            | llm
            | StrOutputParser()
        )

        relevance_score = await relevance_chain.ainvoke({"question": question, "answer": answer})
        faithfulness_score = await faithfulness_chain.ainvoke({"context": context, "answer": answer})

        return {"response": answer, "relevance score": relevance_score, "faithfulness score": faithfulness_score}
    
    
    async def get_metadata(query):
        metadata_policy_name_prompt = await Prompts.get_metadata_policy_name_prompt()
        groq = Groq(model="llama-3.3-70b-versatile")
        llm = await groq.get_llm()

        metadata_chain = (
            {"question": RunnablePassthrough()}
            | metadata_policy_name_prompt
            | llm
            | StrOutputParser()
        )

        policy = await metadata_chain.ainvoke({"question": query})
        
        if "GDPR" in policy:
            policy = "GDPR"
        elif "EU AI Act" in policy:
            policy = "EU AI Act"
        else:
            policy = "Unknown"

        return policy


class BasicRag:
    
    @traceable
    async def executor(query: str, model_name: str="llama-3.3-70b-versatile", collection_name: str="gdpr_euAI_complainces_basic_preprocess", summary_flag: int=0):
        
        chromadb = ChromadDBConfig(collection_name=collection_name)

        policy = await Evaluations.get_metadata(query=query)

        if policy == "GDPR" or policy == "EU AI Act":
            retriever = await chromadb.get_retriever_with_metadata_filter(policy=policy)
        else:
            retriever = await chromadb.get_retriever()

        groq = Groq(model=model_name)
        llm = await groq.get_llm()

        extract_page_content_runnable = RunnableLambda(Prompts.extract_page_content)

        chat_prompt = await Prompts.get_chat_prompt()
        summary_prompt = await Prompts.get_summary_prompt()
        # validator_prompt = await Prompts.get_validator_prompt()

        if summary_flag == 1:
            summary_chain = (
                {"context": retriever | extract_page_content_runnable}
                | summary_prompt
                | llm
                | StrOutputParser()
            )

            basic_rag_chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | chat_prompt
                | llm
                # | {"answer": RunnablePassthrough(), "question": RunnablePassthrough()}
                # | validator_prompt
                # | llm
                | StrOutputParser()
            )

            summary = await summary_chain.ainvoke(query)
            response = await basic_rag_chain.ainvoke({"context": summary, "question": query})
            result = await Evaluations.evaluate(context=summary, answer=response, question=query)
            return result

        else:
            capture_context_runnable = RunnableLambda(capture_context)

            basic_rag_chain = (
                {"context": retriever | extract_page_content_runnable, "question": RunnablePassthrough()}
                | capture_context_runnable
                | chat_prompt
                | llm
                # | {"answer": RunnablePassthrough(), "question": RunnablePassthrough()}
                # | validator_prompt
                # | llm
                | StrOutputParser()
            )

            response = await basic_rag_chain.ainvoke(query)
            context = captured_context.get("context", [])
            result = await Evaluations.evaluate(context=context, answer=response, question=query)
            return result


class HybridRag:
        
        @traceable
        async def executor(query: str, model_name: str="llama-3.3-70b-versatile", collection_name: str="gdpr_euAI_complainces_basic_preprocess", cohere_flag: int=1, summary_flag: int=0):
            chromadb = ChromadDBConfig(collection_name=collection_name)
            
            policy = await Evaluations.get_metadata(query=query)

            if policy == "GDPR" or policy == "EU AI Act":
                # getting normal retriever for chromadb with metadata policy name filtering
                semantic_retriever = await chromadb.get_retriever_with_metadata_filter(k=10, policy=policy)
                
                # getting a chromadb retriever for bm25 with metadata policy name filtering keeping k=10 for chroma and then bm25 will be applied on the docs to fetch top 5 best match docs
                chroma_retriever_BM25 = await chromadb.get_retriever_with_metadata_filter(k=20, policy=policy)
            else:
                # getting normal retriever for chromadb
                semantic_retriever = await chromadb.get_retriever(k=10)
                
                # getting a chromadb retriever for bm25 keeping k=10 for chroma and then bm25 will be applied on the docs to fetch top 5 best match docs
                chroma_retriever_BM25 = await chromadb.get_retriever(k=20)
            
            # this gets docs with semantic retrieval from chroma for bm25
            semantic_docs_for_BM25 = await chromadb.retrieval_for_BM25(retriever=chroma_retriever_BM25, query=query)
            
            # bm25 is intialized and applied
            bm25_retriever = BM25Retriever.from_documents(documents=semantic_docs_for_BM25, kwargs={"k": 10})

            # initializing the ensemble retriever
            ensemble_retriever = EnsembleRetriever(
                 retrievers=[semantic_retriever, bm25_retriever],
                 weights=[0.7, 0.3],
            )

            groq = Groq(model=model_name)
            llm = await groq.get_llm()
    
            extract_page_content_runnable = RunnableLambda(Prompts.extract_page_content)
    
            chat_prompt = await Prompts.get_chat_prompt()
            summary_prompt = await Prompts.get_summary_prompt()
            # validator_prompt = await Prompts.get_validator_prompt()

            if cohere_flag == 1:
                # long context reordering
                reranker = CohereRerank(top_n=5, model="rerank-english-v3.0")

                compression_retriever = ContextualCompressionRetriever(
                    base_retriever=ensemble_retriever,
                    base_compressor=reranker
                )

                if summary_flag == 1:
                    summary_chain = (
                        {"context": compression_retriever | extract_page_content_runnable}
                        | summary_prompt
                        | llm
                        | StrOutputParser()
                    )

                    hybrid_rag_chain = (
                        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                        | chat_prompt
                        | llm
                        # | {"answer": RunnablePassthrough(), "question": RunnablePassthrough()}
                        # | validator_prompt
                        # | llm
                        | StrOutputParser()
                    )

                    summary = await summary_chain.ainvoke(query)
                    response = await hybrid_rag_chain.ainvoke({"context": summary, "question": query})
                    result = await Evaluations.evaluate(context=summary, answer=response, question=query)
                    return result
                
                else:
                    capture_context_runnable = RunnableLambda(capture_context)

                    hybrid_rag_chain = (
                        {"context": compression_retriever | extract_page_content_runnable, "question": RunnablePassthrough()}
                        | capture_context_runnable
                        | chat_prompt
                        | llm
                        # | {"answer": RunnablePassthrough(), "question": RunnablePassthrough()}
                        # | validator_prompt
                        # | llm
                        | StrOutputParser()
                    )
            
                    response = await hybrid_rag_chain.ainvoke(query)
                    context = captured_context.get("context", [])
                    result = await Evaluations.evaluate(context=context, answer=response, question=query)
                    return result
            
            else:
                if summary_flag == 1:
                    summary_chain = (
                        {"context": ensemble_retriever | extract_page_content_runnable}
                        | summary_prompt
                        | llm
                        | StrOutputParser()
                    )

                    hybrid_rag_chain = (
                        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                        | chat_prompt
                        | llm
                        # | {"answer": RunnablePassthrough(), "question": RunnablePassthrough()}
                        # | validator_prompt
                        # | llm
                        | StrOutputParser()
                    )

                    summary = await summary_chain.ainvoke(query)
                    response = await hybrid_rag_chain.ainvoke({"context": summary, "question": query})
                    result = await Evaluations.evaluate(context=summary, answer=response, question=query)
                    return result
                
                else:
                    capture_context_runnable = RunnableLambda(capture_context)

                    hybrid_rag_chain = (
                        {"context": ensemble_retriever | extract_page_content_runnable, "question": RunnablePassthrough()}
                        | capture_context_runnable
                        | chat_prompt
                        | llm
                        # | {"answer": RunnablePassthrough(), "question": RunnablePassthrough()}
                        # | validator_prompt
                        # | llm
                        | StrOutputParser()
                    )
            
                    response = await hybrid_rag_chain.ainvoke(query)
                    context = captured_context.get("context", [])
                    result = await Evaluations.evaluate(context=context, answer=response, question=query)
                    return result
                

class AdvanceRag:
    
    # --- REPLACE YOUR CURRENT FUNCTION WITH THIS ONE ---
    async def self_query_executor(query):
        self_query_prompt = await Prompts.get_self_query_prompt()
        
        groq = Groq(model="llama-3.3-70b-versatile")
        llm = await groq.get_llm()

        self_query_chain = (
            {"question": RunnablePassthrough()}
            | self_query_prompt
            | llm
            | StrOutputParser()
        )

        response = await self_query_chain.ainvoke({"question": query})

        # IMPROVEMENT 1: Strip markdown and extra text
        cleaned = re.sub(r"^```(?:json)?|```$", "", response.strip(), flags=re.MULTILINE).strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            # IMPROVEMENT 2: Don't crash! Just print the error and fall back to the original query
            print(f"⚠️ JSON Parsing failed. Using original query. Error: {e}")
            return {"original query": query}

        return parsed


    async def executor(query: str, model_name: str="llama-3.3-70b-versatile", collection_name: str="gdpr_euAI_complainces_basic_preprocess", cohere_flag: int=1, summary_flag: int=0):

        chromadb = ChromadDBConfig(collection_name=collection_name)

        # llm generates 5 queries from the original user query
        self_queries_json = await AdvanceRag.self_query_executor(query=query)

        retriever = await chromadb.get_retriever()

        # putting all the sub-queries in a list
        sub_queries = [v for k, v in self_queries_json.items() if "query" in k.lower()]

        # fetching top 5 documents from retriever per sub-query
        all_docs = []
        for sub_q in sub_queries:
            top_docs = await chromadb.retrieval_for_advance_rag(retriever=retriever, query=query)
            all_docs.extend(top_docs)

        # reranking all the sub-queries generated
        reranker = CohereRerank(top_n=5, model="rerank-english-v3.0")
        reranked_info = reranker.rerank(documents=all_docs, query=query)
        reranked_docs = [all_docs[item['index']] for item in reranked_info]
        reranked_docs_context = Prompts.extract_page_content(reranked_docs)
        
        groq = Groq(model=model_name)
        llm = await groq.get_llm()

        chat_prompt = await Prompts.get_chat_prompt()
        summary_prompt = await Prompts.get_summary_prompt()


        if summary_flag == 1:
            summary_chain = (
                {"context": RunnablePassthrough()}
                | summary_prompt
                | llm
                | StrOutputParser()
            )

            advance_rag_chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | chat_prompt
                | llm
                | StrOutputParser()
            )

            summary = await summary_chain.ainvoke(reranked_docs_context)
            response = await advance_rag_chain.ainvoke({"context": summary, "question": query})
            result = await Evaluations.evaluate(context=summary, answer=response, question=query)
            return result
        
        else:
            advance_rag_chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | chat_prompt
                | llm
                | StrOutputParser()
            )

            response = await advance_rag_chain.ainvoke({"context": reranked_docs_context, "question": query})
            result = await Evaluations.evaluate(context=reranked_docs, answer=response, question=query)
            return result