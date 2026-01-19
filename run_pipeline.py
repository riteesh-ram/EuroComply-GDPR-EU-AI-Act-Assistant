from pipeline.preprocess_pdfs import Preprocessing, CustomPreprocessingGDPR, CustomPreprocessingEUAI
from config.chroma_client import ChromadDBConfig
import asyncio

async def run_basic_preprocessing(pdfs: list[str]):
    print("--- Starting Basic Preprocessing ---")
    all_chunks = []

    for pdf_path in pdfs:
        try:
            chunks = await Preprocessing.preprocess_pdfs_chroma(pdf_path)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")

    if all_chunks:
        collection_name = "gdpr_euAI_complainces_basic_preprocess"
        chroma_client = ChromadDBConfig(collection_name=collection_name)
        
        # FIX: Initialize the vectorstore before adding data
        await chroma_client.get_vectorstore()
        
        await chroma_client.add_documents(all_chunks)
        print(f"✅ Successfully added {len(all_chunks)} basic chunks to ChromaDB.")
    else:
        print("⚠️ No basic chunks found to add.")


async def run_custom_preprocessing_gdpr(jsonl: list[str]):
    print("\n--- Starting Custom GDPR Preprocessing ---")
    all_chunks = []

    for jsonl_path in jsonl:
        try:
            chunks = await CustomPreprocessingGDPR.preprocess_gdpr_to_chunks(jsonl_path)
            all_chunks.extend(chunks)
        except Exception as e:
             print(f"Error processing JSONL {jsonl_path}: {e}")

    if all_chunks:
        collection_name = "gdpr_euAI_complainces_custom_preprocess"
        chroma_client = ChromadDBConfig(collection_name=collection_name)
        
        # FIX: Initialize the vectorstore before adding data
        await chroma_client.get_vectorstore()
        
        await chroma_client.add_documents(all_chunks)
        print(f"✅ Successfully added {len(all_chunks)} GDPR chunks to ChromaDB.")
    else:
        print("⚠️ No GDPR chunks found to add.")


async def run_custom_preprocessing_EUAI(csv: list[str]):
    print("\n--- Starting Custom EU AI Act Preprocessing ---")
    all_chunks = []

    for csv_path in csv:
        try:
            chunks = await CustomPreprocessingEUAI.process_csv_to_chunks(csv_path)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Error processing CSV {csv_path}: {e}")

    if all_chunks:
        collection_name = "gdpr_euAI_complainces_custom_preprocess"
        chroma_client = ChromadDBConfig(collection_name=collection_name)
        
        # FIX: Initialize the vectorstore before adding data
        await chroma_client.get_vectorstore()
        
        await chroma_client.add_documents(all_chunks)
        print(f"✅ Successfully added {len(all_chunks)} EU AI chunks to ChromaDB.")
    else:
         print("⚠️ No EU AI chunks found to add.")


async def main():
    # Define your file paths
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

    # Run all pipelines
    # We verify the files exist by running the functions (logic inside handles missing files gracefully)
    await run_custom_preprocessing_EUAI(csv)
    await run_custom_preprocessing_gdpr(jsonl)
    await run_basic_preprocessing(pdfs)


if __name__ == "__main__":
    asyncio.run(main())