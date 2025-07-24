import requests
import json

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"

def get_embedding_dimension():
    try:
        # Sample text for embedding
        sample_text = "Test sentence for embedding dimension."
        payload = {
            "model": "nomic-embed-text:latest",
            "prompt": sample_text
        }
        response = requests.post(OLLAMA_EMBED_URL, json=payload)
        response.raise_for_status()
        embedding = response.json().get("embedding")
        if embedding:
            dimension = len(embedding)
            print(f"Embedding dimension for nomic-embed-text:latest: {dimension}")
            return dimension
        else:
            print("Error: No embedding returned.")
            return None
    except Exception as e:
        print(f"Error querying embedding API: {e}")
        return None

if __name__ == "__main__":
    dimension = get_embedding_dimension()