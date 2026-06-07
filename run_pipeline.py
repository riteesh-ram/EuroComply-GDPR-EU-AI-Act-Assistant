from pipeline.preprocess_pdfs import Preprocessing, CustomPreprocessingGDPR, CustomPreprocessingEUAI
from config.chroma_client import ChromadDBConfig
from langchain.schema import Document
import asyncio
import json
import os

async def run_basic_preprocessing(pdfs: list[str]):
    print("\n--- 🚀 START: Basic Preprocessing ---")
    all_chunks = []

    for pdf_path in pdfs:
        if os.path.exists(pdf_path):
            print(f"📄 Processing: {pdf_path}")
            try:
                chunks = await Preprocessing.preprocess_pdfs_chroma(pdf_path)
                all_chunks.extend(chunks)
                print(f"✅ Got {len(chunks)} chunks from {pdf_path}")
            except Exception as e:
                print(f"⚠️ Warning: Could not process {pdf_path}: {e}")
        else:
            print(f"⚠️ Warning: File not found: {pdf_path}")

    # Fallback: if PDFs failed (e.g. LFS pointer files), use preprocessed JSONL chunks
    if not all_chunks:
        print("⚠️ PDF processing yielded no chunks — falling back to preprocessed JSONL sources.")
        fallback_files = [
            "pipeline/preprocessed_docs/gdpr_chunks.jsonl",
            "pipeline/preprocessed_docs/eu_ai_chunks.jsonl"
        ]
        for fpath in fallback_files:
            if os.path.exists(fpath):
                print(f"📄 Loading fallback chunks from: {fpath}")
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            content = obj.get("page_content") or obj.get("content") or obj.get("text", "")
                            metadata = obj.get("metadata", {})
                            if content:
                                all_chunks.append(Document(page_content=content, metadata=metadata))
                        except Exception:
                            pass
                print(f"✅ Loaded {len(all_chunks)} fallback chunks so far")
            else:
                print(f"⚠️ Fallback file not found: {fpath}")

    if all_chunks:
        print(f"📊 Total Basic chunks to ingest: {len(all_chunks)}")
        collection_name = "gdpr_euAI_complainces_basic_preprocess"
        chroma_client = ChromadDBConfig(collection_name=collection_name)
        await chroma_client.get_vectorstore()
        await chroma_client.add_documents(all_chunks)
    else:
        print("⚠️ No chunks available for Basic collection — skipping.")
    print("--- 🏁 END: Basic Preprocessing ---\n")

async def run_custom_preprocessing_gdpr(jsonl: list[str]):
    print("\n--- 🚀 START: GDPR Preprocessing ---")
    all_chunks = []

    for jsonl_path in jsonl:
        if os.path.exists(jsonl_path):
            print(f"📄 Processing: {jsonl_path}")
            chunks = await CustomPreprocessingGDPR.preprocess_gdpr_to_chunks(jsonl_path)
            all_chunks.extend(chunks)
        else:
            print(f"⚠️ Warning: File not found: {jsonl_path}")

    if all_chunks:
        print(f"📊 Total GDPR chunks generated: {len(all_chunks)}")
        collection_name = "gdpr_euAI_complainces_custom_preprocess"
        chroma_client = ChromadDBConfig(collection_name=collection_name)
        await chroma_client.get_vectorstore()
        await chroma_client.add_documents(all_chunks)
    else:
        print("⚠️ No chunks to add for GDPR.")
    print("--- 🏁 END: GDPR Preprocessing ---\n")

async def run_custom_preprocessing_EUAI(csv: list[str]):
    print("\n--- 🚀 START: EU AI Act Preprocessing ---")
    all_chunks = []

    for csv_path in csv:
        if os.path.exists(csv_path):
            print(f"📄 Processing: {csv_path}")
            chunks = await CustomPreprocessingEUAI.process_csv_to_chunks(csv_path)
            all_chunks.extend(chunks)
        else:
            print(f"⚠️ Warning: File not found: {csv_path}")

    if all_chunks:
        print(f"📊 Total EU AI chunks generated: {len(all_chunks)}")
        collection_name = "gdpr_euAI_complainces_custom_preprocess"
        chroma_client = ChromadDBConfig(collection_name=collection_name)
        await chroma_client.get_vectorstore()
        await chroma_client.add_documents(all_chunks)
    else:
        print("⚠️ No chunks to add for EU AI Act.")
    print("--- 🏁 END: EU AI Act Preprocessing ---\n")

async def main():
    print("🛠️ DEBUG: Pipeline Script Started")

    print(f"📂 Current Working Directory: {os.getcwd()}")
    if os.path.exists("pipeline/pdfs"):
        print(f"📂 Files in pipeline/pdfs: {os.listdir('pipeline/pdfs')}")
    else:
        print("❌ ERROR: 'pipeline/pdfs' directory does not exist!")

    pdfs = ["pipeline/pdfs/GDPR_policies.pdf", "pipeline/pdfs/EU AI Act.pdf"]
    jsonl = ["pipeline/pdfs/gdpr_articles_kaggle.jsonl"]
    csv = ["pipeline/pdfs/eu_ai_act_2024_from_pdf.csv"]

    # Custom preprocessing always runs first (JSONL/CSV — not affected by LFS)
    await run_custom_preprocessing_EUAI(csv)
    await run_custom_preprocessing_gdpr(jsonl)

    # Basic preprocessing tries PDFs; falls back to preprocessed JSONL if PDFs fail
    await run_basic_preprocessing(pdfs)

    print("✅ DEBUG: Pipeline Script Finished Successfully")

if __name__ == "__main__":
    asyncio.run(main())
