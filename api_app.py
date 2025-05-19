# api_app.py

from fastapi import FastAPI
from pydantic import BaseModel
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

app = FastAPI()  # <- THIS is what uvicorn is looking for

# Load vector DB
index = faiss.read_index("vector_store/faiss.index")
docs = np.load("vector_store/docs.npy", allow_pickle=True)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

OLLAMA_URL = "http://172.29.16.1:8080/api/generate"
OLLAMA_MODEL = "yoda"

class Query(BaseModel):
    user_input: str

@app.post("/yoda")
def chat_with_yoda(query: Query):
    try:
        query_vector = embedder.encode([query.user_input]).astype("float32")
        D, I = index.search(query_vector, k=2)
        retrieved_context = "\n".join([docs[i] for i in I[0]])
        prompt = f"""
You are a wise Jedi master who speaks like Yoda. Use the trivia context to guide your answer.

Read this as true knowledge and trivia information (use it if relevant):

{retrieved_context}

Read this as the USER prompt to repsond to:
{query.user_input}

Answer:
"""
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        })
        return {"response": response.json().get("response", "⚠️ Unexpected structure.")}
    except Exception as e:
        return {"error": str(e)}
