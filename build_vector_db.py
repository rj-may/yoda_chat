# build_vector_db.py
import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load data
with open("qa_plain2.jsonl", "r", encoding="utf-8") as f:
    docs = [json.loads(line) for line in f]

# Build embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [f"{d['question']} {d['answer']}" for d in docs]
embeddings = model.encode(texts, show_progress_bar=True)

# Save mapping
os.makedirs("vector_store", exist_ok=True)
np.save("vector_store/docs.npy", np.array(texts))

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings).astype("float32"))
faiss.write_index(index, "vector_store/faiss.index")
