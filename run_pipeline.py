from pipeline.preprocess_pdfs import Preprocessing, CustomPreprocessingGDPR, CustomPreprocessingEUAI
from config.chroma_client import ChromadDBConfig
import asyncio

async def run_basic_preprocessing(pdfs: list[str]):

    all_chunks = []

    for pdf_path in pdfs:
        chunks = await Preprocessing.preprocess_pdfs_chroma(pdf_path)
        all_chunks.extend(chunks)

    collection_name = "gdpr_euAI_complainces_basic_preprocess"
    chroma_client = ChromadDBConfig(collection_name=collection_name)
    await chroma_client.add_documents(all_chunks)
    
    print("Documents added to ChromaDB successfully.")

async def run_custom_preprocessing_gdpr(jsonl: list[str]):
    
    all_chunks = []

    for jsonl_path in jsonl:
        chunks = await CustomPreprocessingGDPR.preprocess_gdpr_to_chunks(jsonl_path)
        all_chunks.extend(chunks)

    collection_name = "gdpr_euAI_complainces_custom_preprocess"
    chroma_client = ChromadDBConfig(collection_name=collection_name)
    await chroma_client.add_documents(all_chunks)

    print("Documents added to ChromaDB successfully.")

async def run_custom_preprocessing_EUAI(csv: list[str]):
    all_chunks = []

    for csv_path in csv:
        chunks = await CustomPreprocessingEUAI.process_csv_to_chunks(csv_path)
        all_chunks.extend(chunks)

    collection_name = "gdpr_euAI_complainces_custom_preprocess"
    chroma_client = ChromadDBConfig(collection_name=collection_name)
    await chroma_client.add_documents(all_chunks)
    
    print("Documents added to ChromaDB successfully.")

async def main():
    pdfs = [
        "pipeline/pdfs/GDPR_policies.pdf",
        "pipeline/pdfs/EU AI Act.pdf"
    ]

    jsonl = [
         "pipeline/pdfs/gdpr_articles_kaggle.jsonl"
    ]
    
    csv = [
        "pipeline/pdfs/eu_ai_act_2024_from_pdf.csv"
    ]

    await run_custom_preprocessing_EUAI(csv)
    await run_custom_preprocessing_gdpr(jsonl)
    await run_basic_preprocessing(pdfs)


if __name__ == "__main__":
    asyncio.run(main())




