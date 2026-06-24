from groq import Groq
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

GROQ_API_KEY = "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"

client = Groq(api_key=GROQ_API_KEY)
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_data():
    print("Loading index and chunks...")
    index = faiss.read_index("docs.index")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    print(f"Loaded {len(chunks)} chunks!")
    return index, chunks

def search(query, index, chunks, top_k=3):
    query_embedding = model.encode([query])
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
    context = "\n\n".join([r["content"] for r in results])
    prompt = f"""You are an expert on the Freshservice API.
Use the following documentation to answer the question.
Always include relevant parameters, curl commands, or 
examples if available.

Documentation:
{context}

Question: {user_question}

Answer:"""
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    return answer, results

if __name__ == "__main__":
    index, chunks = load_data()
    question = "Give me the curl command to create a ticket"
    print(f"\nQuestion: {question}")
    print("\nSearching documentation...")
    answer, sources = answer_question(question, index, chunks)
    print("\n--- Answer ---")
    print(answer)
    print("\n--- Sources Used ---")
    for i, s in enumerate(sources):
        title = s['title'].encode('ascii', errors='ignore').decode()
        print(f"[{i+1}] {title} (score: {s['score']:.2f})")