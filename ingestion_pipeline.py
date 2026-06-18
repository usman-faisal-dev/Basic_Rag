from dotenv import load_dotenv
import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from uuid import uuid4
import shutil

load_dotenv()

print("Initializing Ollama Embeddings...")
embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
)

db_location = os.getenv("DATABASE_LOCATION")

if os.path.exists(db_location):
    print(f"Clearing existing vector database at: {db_location}")
    shutil.rmtree(db_location)

print("Connecting to Chroma DB (This is where it hangs if Ollama is unresponsive)...")
vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=db_location, 
)
print("Connected to Chroma DB successfully!")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

knowledge_base_dir = Path("rag_knowledge_base")
documents_to_process = []

# Verify directory exists
if not knowledge_base_dir.exists():
    print(f"Error: The directory '{knowledge_base_dir}' does not exist. Run your data_scraping.py script first.")
    exit()

# Loop through all markdown files saved by Crawl4AI
for file_path in knowledge_base_dir.glob("*"):
    if file_path.is_file():
        print(f"Reading document: {file_path.name}")
        
        # Read the plain markdown text
        markdown_content = file_path.read_text(encoding="utf-8")
        
        # Construct a LangChain Document with metadata tracking the origin
        doc = Document(
            page_content=markdown_content,
            metadata={
                "source": file_path.name,
                "title": file_path.stem.replace("_", " ")  # Reconstructs title from filename
            }
        )
        documents_to_process.append(doc)

print(f"\nSplitting and Ingesting {len(documents_to_process)} documents into Chroma...")

for doc in documents_to_process:
    print(f"Processing chunking for: {doc.metadata['title']}")
    
    # Pass the actual Document object to split_documents
    chunks = text_splitter.split_documents([doc])
    
    total_chunks = len(chunks)
    print(f"Created {total_chunks} chunks from this document.")
    print(f"Generating embeddings via Ollama and saving to Chroma DB...")

    # Process in smaller batches of 5 chunks so you can see live terminal updates
    batch_size = 5
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        uuids = [str(uuid4()) for _ in range(len(batch))]
        
        # Ingest the small batch
        vector_store.add_documents(documents=batch, ids=uuids)
        
        # Print progress update
        processed_count = min(i + batch_size, total_chunks)
        print(f"Progress: Ingested {processed_count}/{total_chunks} chunks...")

print("\nIngestion complete! Your local RAG knowledge base is fully indexed and ready.")