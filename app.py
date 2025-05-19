
import gradio as gr
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load FAISS index + docs
index = faiss.read_index("vector_store/faiss.index")
docs = np.load("vector_store/docs.npy", allow_pickle=True)

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Ollama endpoint + model
OLLAMA_URL = "http://172.29.16.1:8080/api/generate" #:http://127.0.0.1:11434
# OLLAMA_URL = "http://127.0.0.1:11434/"

OLLAMA_MODEL = "yoda"  # <- Change this to your actual Yoda model name

def chat_with_yoda(user_input):
    try:
        # Embed query
        query_vector = embedder.encode([user_input]).astype("float32")
        
        # Search vector DB
        D, I = index.search(query_vector, k=2)
        retrieved_context = "\n".join([docs[i] for i in I[0]])
        print(retrieved_context)

        # Build prompt
        prompt = f"""
You are a wise Jedi master who speaks like Yoda. Use the trivia context to guide your answer.

Read this as your knowledge and trivia information:

{retrieved_context}

Read this as the the USER prompt:
{user_input}

Answer:
"""
        
        # Send to Ollama
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        })
        
        data = response.json()
        return data.get("response", "⚠️ Unexpected structure.")
    except Exception as e:
        return f"❌ Exception: {e}"


css = """
.gradio-container {
    background-color: black !important;
    color: white !important;
}

.gradio-container h1 {
    color: green !important; /* Change this to your preferred color */
}

"""


# Gradio UI
gr.Interface(
    fn=chat_with_yoda,
    # inputs="text",
    # outputs="text",
    inputs=gr.Textbox(label="Your Question"),
    outputs=gr.Textbox(label="Yoda's Response"),
    title="⭐ Yoda Trivia Chat",
    theme=gr.themes.Glass(primary_hue="green", secondary_hue=gr.themes.colors.gray),
    css=css
).launch(share = True)

# return f"""
# You are a wise Jedi master who speaks like Yoda.

# Use the following trivia to answer the question. 
# Do not make up facts. If unsure, say: "Hmm. Know, I do not."

# === TRIVIA ===
# {context}

# === QUESTION ===
# {question}

# === ANSWER ===
# """