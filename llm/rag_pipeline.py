import os
import hashlib
import faiss
import numpy as np
import requests
from PyPDF2 import PdfReader
from tqdm import tqdm
from logger import get_logger

logger = get_logger(__name__)

PDF_DIR = "data/docs"
INDEX_DIR = "data/faiss_index"
HASH_FILE = os.path.join(INDEX_DIR, ".doc_hash")
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"

def hash_docs(pdf_paths):
    hash_md5 = hashlib.md5()
    for path in sorted(pdf_paths):
        with open(path, 'rb') as f:
            hash_md5.update(f.read())
    return hash_md5.hexdigest()

def load_documents():
    all_docs = []
    logger.info("Loading PDF documents from '%s'...", PDF_DIR)
    for filename in os.listdir(PDF_DIR):
        if filename.endswith(".pdf"):
            path = os.path.join(PDF_DIR, filename)
            try:
                reader = PdfReader(path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                all_docs.append({"content": text, "filename": filename})
                logger.info("  - Loaded %s (%d pages)", filename, len(reader.pages))
            except Exception as e:
                logger.error(f"Failed to load %s: %s", filename, e)
                continue
    return all_docs

def split_text(text, chunk_size=500, chunk_overlap=100):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def split_documents(documents):
    all_chunks = []
    for doc in documents:
        chunks = split_text(doc["content"])
        for chunk in chunks:
            all_chunks.append({"content": chunk, "filename": doc["filename"]})
    return all_chunks

def get_embedding(text):
    try:
        payload = {"model": "nomic-embed-text:latest", "prompt": text}
        response = requests.post(OLLAMA_EMBED_URL, json=payload)
        response.raise_for_status()
        return response.json().get("embedding")
    except Exception as e:
        logger.error(f"Embedding failed for text: {e}")
        return None

def build_or_load_vectorstore():
    os.makedirs(INDEX_DIR, exist_ok=True)
    pdf_paths = [os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    current_hash = hash_docs(pdf_paths)

    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            previous_hash = f.read().strip()
        logger.info("Checking existing FAISS index...")
        if previous_hash == current_hash and os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
            logger.info("Loading existing FAISS index from '%s'...", INDEX_DIR)
            try:
                index = faiss.read_index(os.path.join(INDEX_DIR, "index.faiss"))
                with open(os.path.join(INDEX_DIR, "chunks.txt"), "r") as f:
                    chunks = [line.strip() for line in f]
                return index, chunks
            except Exception as e:
                logger.error(f"Failed to load FAISS index: %s", e)
                logger.info("Rebuilding FAISS index...")

    logger.info("Rebuilding FAISS index due to PDF changes or loading failure...")
    documents = load_documents()
    if not documents:
        logger.error("No valid PDF documents found in '%s'. Cannot build vector store.", PDF_DIR)
        return None, None
    logger.info(f"Loaded {len(documents)} documents for processing.")
    chunks = split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks.")

    # Create FAISS index
    dimension = 768  # Assuming nomic-embed-text dimension (adjust if needed)
    index = faiss.IndexFlatL2(dimension)
    embeddings = []
    chunk_texts = []

    logger.info("Embedding documents")
    for chunk in tqdm(chunks, desc="🔍 Embedding chunks"):
        embedding = get_embedding(chunk["content"])
        if embedding:
            embeddings.append(embedding)
            chunk_texts.append(chunk["content"])
    
    if embeddings:
        embeddings = np.array(embeddings, dtype=np.float32)
        index.add(embeddings)
        logger.info("Saving FAISS index")
        try:
            faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))
            with open(os.path.join(INDEX_DIR, "chunks.txt"), "w") as f:
                for chunk in chunk_texts:
                    f.write(f"{chunk}\n")
            with open(HASH_FILE, "w") as f:
                f.write(current_hash)
            logger.info("Saved vectorstore successfully.")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: %s", e)
            raise
    return index, chunk_texts

def retrieve_context(query, k=10):
    logger.info(f"Retrieving RAG context for query: {query}")
    try:
        index, chunks = build_or_load_vectorstore()
        if index is None or chunks is None:
            logger.warning("No vector store available due to missing documents.")
            return "No relevant agricultural knowledge found. Please add PDF documents to 'data/docs/'."
        
        query_embedding = get_embedding(query)
        if not query_embedding:
            return "Error: Could not embed query."
        
        query_embedding = np.array([query_embedding], dtype=np.float32)
        distances, indices = index.search(query_embedding, k)
        relevant_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks.")
        return "\n\n".join(relevant_chunks)
    except Exception as e:
        logger.error(f"Context retrieval failed: %s", e)
        return "No relevant agricultural knowledge found. Please add PDF documents to 'data/docs/'."