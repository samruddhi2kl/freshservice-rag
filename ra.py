import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def load_data():
    print("Loading index and chunks...")
    index = faiss.read_index("docs.index")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    print(f"Loaded {len(chunks)} chunks!")
    return index, chunks

def search(query, index, chunks, top_k=3):
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding, dtype="float32")
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "title": chunks[idx]["title"],
            "content": chunks[idx]["content"],
            "score": float(distances[0][i])
        })
    return results

def answer_question(user_question, index, chunks):
    results = search(user_question, index, chunks)
    
    # Build answer from retrieved chunks directly
    answer = f"Based on Freshservice API documentation:\n\n"
    for i, r in enumerate(results):
        answer += f"--- Section {i+1}: {r['title']} ---\n"
        answer += r['content'] + "\n\n"
    
    return answer, results

if __name__ == "__main__":
    index, chunks = load_data()
    
    while True:
        print("\n" + "="*50)
        question = input("Ask a question (or type 'quit' to exit): ")
        
        if question.lower() == 'quit':
            break
            
        print("\nSearching documentation...")
        answer, sources = answer_question(question, index, chunks)
        
        print("\n--- Answer ---")
        print(answer[:2000])
        
        print("--- Sources ---")
        for i, s in enumerate(sources):
            title = s['title'].encode('ascii', errors='ignore').decode()
            print(f"[{i+1}] {title} (score: {s['score']:.2f})")