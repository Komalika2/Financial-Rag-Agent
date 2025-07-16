import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# --- Configuration ---
CHUNKS_DIR = os.path.join(os.path.dirname(__file__), '../chunks')
EMBEDDINGS_DIR = os.path.join(os.path.dirname(__file__), '../embeddings') # Make sure this line is present
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, 'faiss_index.bin') # Make sure this line is updated
CHUNK_METADATA_PATH = os.path.join(CHUNKS_DIR, 'processed_chunks_metadata.json')

# Ensure the embeddings directory exists
os.makedirs(EMBEDDINGS_DIR, exist_ok=True) # Make sure this line is present

class RAGPipeline:
    def __init__(self):
        self.model = None
        self.index = None
        self.chunks_metadata = []
        self._initialize_model()

    def _initialize_model(self):
        """Initializes the Sentence Transformer model."""
        try:
            print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
            self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print("Embedding model loaded successfully.")
        except Exception as e:
            print(f"Error loading SentenceTransformer model: {e}")
            print("Please ensure you have 'sentence-transformers' installed: pip install sentence-transformers")
            self.model = None

    def _load_chunks_metadata(self):
        """Loads chunk metadata from the JSON file."""
        if not os.path.exists(CHUNK_METADATA_PATH):
            print(f"Error: Chunk metadata file not found at {CHUNK_METADATA_PATH}")
            print("Please run 'python scripts/parse_and_chunk.py' first.")
            return False
        try:
            with open(CHUNK_METADATA_PATH, 'r', encoding='utf-8') as f:
                self.chunks_metadata = json.load(f)
            print(f"Loaded {len(self.chunks_metadata)} chunks from {CHUNK_METADATA_PATH}")
            return True
        except Exception as e:
            print(f"Error loading chunk metadata: {e}")
            return False

    def _create_and_save_embeddings_and_index(self):
        """Generates embeddings and creates a FAISS index, then saves it."""
        if not self.model:
            print("Embedding model not loaded. Cannot create embeddings.")
            return False
        
        if not self._load_chunks_metadata():
            return False

        if not self.chunks_metadata:
            print("No chunks found to embed.")
            return False

        print("Generating embeddings for chunks...")
        texts = [chunk['text'] for chunk in self.chunks_metadata]
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype('float32'))
            print(f"FAISS index created with {self.index.ntotal} embeddings.")

            faiss.write_index(self.index, FAISS_INDEX_PATH)
            print(f"FAISS index saved to {FAISS_INDEX_PATH}")
            return True
        except Exception as e:
            print(f"Error creating/saving embeddings or FAISS index: {e}")
            print("Please ensure 'faiss-cpu' is installed: pip install faiss-cpu")
            return False

    def load_or_create_index(self):
        """Loads FAISS index if it exists, otherwise creates it."""
        # Check if both index file and metadata file exist
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNK_METADATA_PATH):
            try:
                print(f"Attempting to load FAISS index from {FAISS_INDEX_PATH}...")
                self.index = faiss.read_index(FAISS_INDEX_PATH)
                print("FAISS index loaded successfully.")
                self._load_chunks_metadata() # Load metadata to match the index
                return True
            except Exception as e:
                print(f"Error loading FAISS index: {e}. Attempting to re-create index.")
                return self._create_and_save_embeddings_and_index()
        else:
            print("FAISS index or chunk metadata not found. Creating new index.")
            return self._create_and_save_embeddings_and_index()

    def retrieve_chunks(self, query: str, k: int = 5):
        """
        Retrieves top-k relevant chunks for a given query.
        """
        if not self.model or not self.index or not self.chunks_metadata:
            print("RAG pipeline not fully initialized. Please ensure load_or_create_index() ran successfully.")
            return []

        try:
            query_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')
            distances, indices = self.index.search(query_embedding, k)
            
            retrieved_chunks = []
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.chunks_metadata): # Ensure index is valid and within bounds
                    chunk = self.chunks_metadata[idx]
                    retrieved_chunks.append({
                        "text": chunk['text'],
                        "company": chunk['company'],
                        "year": chunk['year'],
                        "source_doc": chunk['source_doc'],
                        "chunk_id": chunk['chunk_id'],
                        "page": chunk['page'],
                        "distance": distances[0][i] # Optional: similarity score/distance
                    })
                else:
                    print(f"Warning: Invalid index {idx} encountered during retrieval.")
            return retrieved_chunks
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

# Example usage (for testing this module directly)
if __name__ == "__main__":
    print("--- Running embed_and_store.py directly for testing ---")
    rag_pipeline = RAGPipeline()
    if rag_pipeline.load_or_create_index():
        print("\n--- Testing Retrieval ---")
        test_query = "What was Microsoft's total revenue in 2023?"
        print(f"Query: '{test_query}'")
        results = rag_pipeline.retrieve_chunks(test_query, k=3)
        if results:
            for i, result in enumerate(results):
                print(f"\n--- Retrieved Chunk {i+1} (Distance: {result.get('distance', 'N/A'):.4f}) ---")
                print(f"Source: {result['company']} {result['year']} - {result['source_doc']} (Page: {result['page']})")
                print(result['text'][:500] + "...") # Print first 500 characters
        else:
            print("No chunks retrieved.")
    else:
        print("\nFailed to initialize RAG pipeline.")