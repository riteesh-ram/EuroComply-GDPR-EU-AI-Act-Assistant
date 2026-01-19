from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import asyncio
import re
import os
import json
import csv
import ast

class Preprocessing:

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'-\n', '', text)  # Fix hyphenated line breaks
        text = re.sub(r'\n+', '\n', text)  # Collapse multiple newlines
        text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces/tabs
        text = text.strip()
        return text
    
    async def preprocess_pdfs_chroma(pdf_path: str) -> list[Document]:

        loop = asyncio.get_event_loop()
        loader = PyPDFLoader(pdf_path)
        pages = await loop.run_in_executor(None, loader.load)

        cleaned_pages = [
            Document(
                page_content=Preprocessing.clean_text(doc.page_content),
                metadata=doc.metadata
            )
            for doc in pages
        ]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=200,
        )

        chunks = splitter.split_documents(cleaned_pages)

        return chunks
    
class CustomPreprocessingGDPR:

    chapter_mapping = {
        1: range(1, 5),
        2: range(5, 12),
        3: range(12, 24),
        4: range(24, 44),
        5: range(44, 51),
        6: range(51, 60),
        7: range(60, 77),
        8: range(77, 85),
        9: range(85, 92),
        10: range(92, 94),
        11: range(94, 100),
    }
    
    async def find_chapter(article_number):
        for chapter, article_range in CustomPreprocessingGDPR.chapter_mapping.items():
            if article_number in article_range:
                return chapter
        return ""

    async def improved_extract_number_title(input_text):
        if input_text.startswith("Article "):
            parts = input_text.split(" ", 2)
            return {
                "type": "article",
                "number": int(parts[1]),
                "title": parts[2] if len(parts) > 2 else ""
            }
        else:
            parts = input_text.split("-", 2)
            try:
                number = int(parts[1])
            except (IndexError, ValueError):
                number = None
            return {
                "type": "recital",
                "number": number,
                "title": ""
            }

    async def clean_text(text):
        text = re.sub(r'-\n', '', text)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    async def preprocess_gdpr_to_chunks(jsonl_path: str) -> list[Document]:
        chunks = []

        # Load JSONL from file path
        with open(jsonl_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        for entry in data:
            parsed = await CustomPreprocessingGDPR.improved_extract_number_title(entry["input-text"])
            chapter_number = await CustomPreprocessingGDPR.find_chapter(parsed["number"]) if parsed["type"] == "article" else ""

            metadata = {
                "number": parsed["number"],
                "title": parsed["title"] if parsed["type"] == "article" else "",
                "chapter_number": chapter_number if parsed["type"] == "article" else "",
                "policy": "GDPR",
                "type": parsed["type"]
            }

            full_text = await CustomPreprocessingGDPR.clean_text(
                f"{entry['output-text']} (number: {parsed['number']}, title: {parsed['title']}, chapter: {chapter_number})"
            )

            chunks.append(Document(page_content=full_text, metadata=metadata))
        
        # Save to JSONL
        os.makedirs("pipeline/preprocessed_docs", exist_ok=True)
        with open("pipeline/preprocessed_docs/gdpr_chunks.jsonl", "w", encoding="utf-8") as f:
            for doc in chunks:
                json.dump({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }, f, ensure_ascii=False)
                f.write("\n")

        return chunks



class CustomPreprocessingEUAI:

    async def extract_number(source, prefix):
        if source is None:
            return None
        number_str = str(source).replace(prefix, '').strip()
        try:
            return int(number_str)
        except ValueError:
            return number_str

    async def process_meta_item(meta_item):
        metadata = {
            "number": None,
            "title": "",
            "chapter_number": "",
            "policy": "EU AI Act",
            "type": ""
        }

        if 'Preamble' in meta_item and meta_item['Preamble']:
            recital = meta_item['Preamble']
            metadata.update({
                "number": await CustomPreprocessingEUAI.extract_number(recital, 'Recital '),
                "type": "recital"
            })
        elif 'chapter' in meta_item and meta_item['chapter']:
            chapter = meta_item['chapter']
            chapter_number = await CustomPreprocessingEUAI.extract_number(chapter, 'Chapter ')
            
            if 'article' in meta_item and meta_item['article']:
                article = meta_item['article']
                metadata.update({
                    "number": await CustomPreprocessingEUAI.extract_number(article, 'Article '),
                    "title": meta_item.get('article_title', ''),
                    "chapter_number": chapter_number,
                    "type": "article"
                })
            else:
                metadata.update({
                    "number": chapter_number,
                    "title": meta_item.get('chapter_title', ''),
                    "chapter_number": chapter_number,
                    "type": "chapter"
                })
        elif 'annex' in meta_item and meta_item['annex']:
            annex = meta_item['annex']
            metadata.update({
                "number": await CustomPreprocessingEUAI.extract_number(annex, 'ANNEX '),
                "title": meta_item.get('title', ''),
                "type": "annex"
            })

        return metadata

    async def process_csv_to_chunks(csv_path: str) -> list[Document]:
        output_path = "pipeline/preprocessed_docs/eu_ai_chunks.jsonl"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        chunks = []

        with open(csv_path, 'r', encoding='utf-8') as file, open(output_path, 'w', encoding='utf-8') as outfile:
            reader = csv.DictReader(file)

            for row in reader:
                page_content = row['text']
                first_metadata = None

                try:
                    meta_list = ast.literal_eval(row['meta_data'])
                except:
                    meta_list = []

                for meta_item in meta_list:
                    try:
                        processed = await CustomPreprocessingEUAI.process_meta_item(meta_item)
                        if processed["number"] is not None:
                            first_metadata = processed
                            break
                    except Exception as e:
                        print(f"Error processing item: {meta_item}")
                        print(f"Error details: {str(e)}")
                        continue

                if first_metadata:
                    doc = Document(page_content=page_content, metadata=first_metadata)
                    chunks.append(doc)

                    # Save to JSONL
                    json.dump({"page_content": doc.page_content, "metadata": doc.metadata}, outfile, ensure_ascii=False)
                    outfile.write("\n")

        return chunks