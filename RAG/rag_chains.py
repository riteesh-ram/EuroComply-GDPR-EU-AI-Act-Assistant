from config.chroma_client import ChromadDBConfig
from config.groq_client import Groq
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
import json
import re
import asyncio

captured_context = {}

def capture_context(inputs):
    captured_context['context'] = inputs['context']
    return inputs

class Prompts:

    async def get_chat_prompt():
        chat_template = """
        You are EuroComply, an AI assistant specialising in GDPR and EU AI Act compliance.

        Context retrieved from the knowledge base:
        {context}

        Question: {question}

        Instructions:
        - If the question is a greeting (e.g. "hi", "hello", "hey") or small talk, respond in a friendly, brief way and mention what you can help with (GDPR and EU AI Act compliance questions).
        - If the question asks what you are, what you do, or what you are good at, introduce yourself as EuroComply and explain your purpose.
        - If the context clearly answers the question, provide a concise and accurate response using that context.
        - If the context partially answers the question, reflect that partial information.
        - If the question is about compliance but the context does not contain relevant information, say so honestly.
        - Do not invent facts about GDPR or EU AI Act that are not in the context.
        - Do not give unnecessary reasoning or preamble.

        Answer:
        """
        chat_prompt = ChatPromptTemplate.from_template(chat_template)
        return chat_prompt

    async def get_validator_prompt():
        validator_template = """
        You are a validator. Determine if the given answer is valid based on the user's question.
        - If the answer is a greeting, introduction, or general response about the assistant's purpose, output it as-is — it is valid.
        - If the answer has enough relevancy to the question, output the answer itself.
        - If the answer is clearly off-topic or nonsensical for a compliance assistant, output: "Sorry, I cannot answer that at the moment."
        - Do not give any reasoning.

        Question: {question}
        Answer: {answer}
        """
        validator_prompt = ChatPromptTemplate.from_template(validator_template)
        return validator_prompt

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

    async def evaluate(context, answer, question, model_name: str = "llama-3.1-8b-instant"):
        relevance_prompt = await Prompts.get_relevance_prompt()
        faithfulness_prompt = await Prompts.get_faithfulness_prompt()

        groq = Groq(model=model_name)
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


class HybridRag:

    @traceable
    async def executor(query: str, model_name: str = "llama-3.3-70b-versatile", collection_name: str = "gdpr_euAI_complainces_custom_preprocess"):
        chromadb = ChromadDBConfig(collection_name=collection_name)

        policy = await Evaluations.get_metadata(query=query)

        use_filter = (policy == "GDPR" or policy == "EU AI Act") and "custom" in collection_name
        if use_filter:
            semantic_retriever = await chromadb.get_retriever_with_metadata_filter(k=10, policy=policy)
            chroma_retriever_BM25 = await chromadb.get_retriever_with_metadata_filter(k=20, policy=policy)
        else:
            semantic_retriever = await chromadb.get_retriever(k=10)
            chroma_retriever_BM25 = await chromadb.get_retriever(k=20)

        semantic_docs_for_BM25 = await chromadb.retrieval_for_BM25(retriever=chroma_retriever_BM25, query=query)

        if not semantic_docs_for_BM25:
            return {"response": "Sorry, I cannot answer that at the moment.", "relevance score": "N/A", "faithfulness score": "N/A"}

        bm25_retriever = BM25Retriever.from_documents(documents=semantic_docs_for_BM25, k=10)

        ensemble_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=[0.7, 0.3],
        )

        reranker = CohereRerank(top_n=5, model="rerank-english-v3.0")
        compression_retriever = ContextualCompressionRetriever(
            base_retriever=ensemble_retriever,
            base_compressor=reranker
        )

        groq = Groq(model=model_name)
        llm = await groq.get_llm()

        extract_page_content_runnable = RunnableLambda(Prompts.extract_page_content)
        capture_context_runnable = RunnableLambda(capture_context)
        chat_prompt = await Prompts.get_chat_prompt()

        hybrid_rag_chain = (
            {"context": compression_retriever | extract_page_content_runnable, "question": RunnablePassthrough()}
            | capture_context_runnable
            | chat_prompt
            | llm
            | StrOutputParser()
        )

        response = await hybrid_rag_chain.ainvoke(query)
        context = captured_context.get("context", [])
        return await Evaluations.evaluate(context=context, answer=response, question=query, model_name=model_name)


class AdvanceRag:

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

        response = await self_query_chain.ainvoke(query)

        cleaned = re.sub(r"^```(?:json)?|```$", "", response.strip(), flags=re.MULTILINE).strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON Parsing failed. Using original query. Error: {e}")
            return {"original query": query}

        return parsed

    async def executor(query: str, model_name: str = "llama-3.3-70b-versatile", collection_name: str = "gdpr_euAI_complainces_custom_preprocess"):
        chromadb = ChromadDBConfig(collection_name=collection_name)

        self_queries_json = await AdvanceRag.self_query_executor(query=query)

        retriever = await chromadb.get_retriever()

        sub_queries = [v for k, v in self_queries_json.items() if "query" in k.lower()]

        results = await asyncio.gather(
            *[chromadb.retrieval_for_advance_rag(retriever=retriever, query=sub_q) for sub_q in sub_queries]
        )
        all_docs = [doc for docs in results for doc in docs]

        reranker = CohereRerank(top_n=5, model="rerank-english-v3.0")
        reranked_info = reranker.rerank(documents=all_docs, query=query)
        reranked_docs = [all_docs[item['index']] for item in reranked_info]
        reranked_docs_context = Prompts.extract_page_content(reranked_docs)

        groq = Groq(model=model_name)
        llm = await groq.get_llm()

        chat_prompt = await Prompts.get_chat_prompt()

        advance_rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | chat_prompt
            | llm
            | StrOutputParser()
        )

        response = await advance_rag_chain.ainvoke({"context": reranked_docs_context, "question": query})
        return await Evaluations.evaluate(context=reranked_docs, answer=response, question=query, model_name=model_name)
