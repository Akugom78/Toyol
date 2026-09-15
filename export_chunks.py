import os
import json
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_FOLDER = "./documents"
OUTPUT_FILE = "./data/chunks.json"

print("📚 Loading and chunking documents...")
loader = DirectoryLoader(PDF_FOLDER, glob="**/*.pdf", loader_cls=PyMuPDFLoader, use_multithreading=True)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
chunks = text_splitter.split_documents(documents)

# Format for lightweight JSON storage
chunk_data = []
for doc in chunks:
    chunk_data.append({
        "text": doc.page_content,
        "source": os.path.basename(doc.metadata.get("source", "Unknown")),
        "page": doc.metadata.get("page", "?")
    })

# Ensure data folder exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(chunk_data, f, ensure_ascii=False, indent=2)

print(f"✅ Successfully exported {len(chunk_data)} chunks to {OUTPUT_FILE}")