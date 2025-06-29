# build_myntra_index.py
"""
Script to embed a Myntra catalog (CSV with img, title, etc.) using BGE-large and dump FAISS index.
Only generates:
    - myntra_bge.index
    - myntra_meta.json

Minimal and deterministic. No CLI. Edit CATALOG_PATH as needed.
"""

import json
import pathlib

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# --- Config ---
CATALOG_PATH = pathlib.Path("/Users/arjunrao/Downloads/myntra202305041052.csv")
INDEX_FILE = "myntra_bge.index"
META_FILE = "myntra_meta.json"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBED_BATCH = 64

# --- Load and preprocess catalog ---
print("[+] Loading catalog…")
df = pd.read_csv(CATALOG_PATH).sample(3000)

# If 'name' or 'brand' missing, fall back
if "name" not in df.columns:
    raise ValueError("Catalog must contain a 'name' column.")

text_cols = ["name", "brand", "price", "rating", "discount"]
existing = [col for col in text_cols if col in df.columns]
df["__blob__"] = df[existing].fillna("").astype(str).agg(" ".join, axis=1)

# --- Embed ---
print("[+] Loading embedding model…")
model = SentenceTransformer(MODEL_NAME)

print("[+] Embedding products…")
vectors = model.encode(
    df["__blob__"].tolist(),
    batch_size=EMBED_BATCH,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
).astype("float32")

# --- Build FAISS index ---
print("[+] Building FAISS index…")
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)

# --- Save ---
print("[+] Saving index and metadata…")
faiss.write_index(index, INDEX_FILE)
with open(META_FILE, "w") as f:
    json.dump(df.to_dict(orient="records"), f)

print(f"✅ Done. Saved index → {INDEX_FILE}, metadata → {META_FILE}")
