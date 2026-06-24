import streamlit as st
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Freshservice API Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Freshservice API Assistant")
st.write("Ask anything about the Freshservice API!")

@st.cache_resource
def load_data():
    index = faiss.read_index("docs.index")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, chunks, model

index, chunks, model = load_data()

def search(query, top_k=3):
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

def answer_question(question):
    results = search(question)
    answer = "**Based on Freshservice API Documentation:**\n\n"
    for i, r in enumerate(results):
        answer += f"### Section {i+1}: {r['title']}\n"
        answer += r['content'] + "\n\n"
    return answer, results

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box
if question := st.chat_input("Ask about Freshservice API..."):
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documentation..."):
            answer, sources = answer_question(question)
        st.markdown(answer)
        
        st.subheader("📚 Sources Used")
        for i, s in enumerate(sources):
            with st.expander(f"Source {i+1}: {s['title']}"):
                st.write(s['content'])
                st.caption(f"Relevance score: {s['score']:.2f}")

    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer
    })