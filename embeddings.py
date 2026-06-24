from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

def build_index():
    # Load scraped chunks
    print("Loading chunks...")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    
    print(f"Loaded {len(chunks)} chunks!")
    
    # Extract just the text content
    texts = [chunk["content"] for chunk in chunks]
    
    # Load embedding model
    # This will download automatically (~90MB, only once)
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Convert all chunks to vectors
    print("Creating embeddings... (this takes 1-2 mins)")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Build FAISS index
    print("Building search index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save index and chunks
    faiss.write_index(index, "docs.index")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    
    print("Index saved to docs.index!")
    return index, chunks


def search(query, index, chunks, top_k=3):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Convert question to vector
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding, dtype="float32")
    
    # Search for similar chunks
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "title": chunks[idx]["title"],
            "content": chunks[idx]["content"],
            "score": float(distances[0][i])
        })
    
    return results


if __name__ == "__main__":
    # Build the index
    index, chunks = build_index()
    
    # Test a search
    print("\nTesting search...")
    results = search("how to create a ticket", index, chunks)
    
    print("\n--- Search Results ---")
    for i, r in enumerate(results):
        title = r['title'].encode('ascii', errors='ignore').decode()
        content = r['content'][:200].encode('ascii', errors='ignore').decode()
        print(f"\n[{i+1}] Title: {title}")
        print(f"     Score: {r['score']:.2f}")
        print(f"     Content: {content}")