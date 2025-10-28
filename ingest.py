# ingest.py
import os, uuid, time
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Prefer the new package; fall back to community if needed
try:
    from langchain_ollama import OllamaEmbeddings
    OLLAMA_MSG = None
except Exception:
    from langchain_community.embeddings import OllamaEmbeddings
    OLLAMA_MSG = ("NOTE: using deprecated langchain_community.OllamaEmbeddings. "
                  "pip install -U langchain-ollama and switch imports to silence this.")

from pinecone_utils import get_pc_index
from tqdm import tqdm

# ----------------- config -----------------
DATA_PATH = "data/crime_and_punishment.txt"
EMBED_MODEL = "nomic-embed-text"   # 768-dim
EMBED_BATCH = 128                  # how many chunks to embed per call
UPSERT_BATCH = 200                 # how many vectors per Pinecone upsert call
NAMESPACE = "c_and_p"              # keep project data separate
# -----------------------------------------

def chunked(seq, n):
    """Yield (start_index, slice) pairs of size n from seq."""
    for i in range(0, len(seq), n):
        yield i, seq[i:i + n]

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

if __name__ == "__main__":
    load_dotenv()

    if OLLAMA_MSG:
        print(OLLAMA_MSG)

    # 1) Load and chunk the text
    text = load_text(DATA_PATH)
    print(f"Loaded file: {DATA_PATH}  ({len(text):,} chars)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    print(f"Chunks: {len(chunks)}  (≈1000 chars each, overlap=150)")

    # 2) Warm up the embedder (fast fail if Ollama/model not ready)
    emb = OllamaEmbeddings(model=EMBED_MODEL)
    try:
        dim = len(emb.embed_query("ping"))
        print(f"Embedding model '{EMBED_MODEL}' OK (dim={dim})")
    except Exception as e:
        print(
            "Embedding warm-up FAILED.\n"
            "- Ensure Ollama is running:   ollama serve\n"
            "- Pull the model once:       ollama pull nomic-embed-text\n"
            f"- Error: {e}"
        )
        raise

    # 3) Embed with progress
    vectors = []
    t0 = time.perf_counter()
    total_batches = (len(chunks) + EMBED_BATCH - 1) // EMBED_BATCH
    for _, part in tqdm(chunked(chunks, EMBED_BATCH), total=total_batches, desc="Embedding"):
        vectors.extend(emb.embed_documents(part))
    t1 = time.perf_counter()
    rate = len(chunks) / (t1 - t0) if t1 > t0 else 0.0
    print(f"Embedded {len(chunks)} chunks in {t1 - t0:.1f}s  ({rate:.1f} chunks/s)")
    assert len(vectors) == len(chunks), "vector count != chunk count"

    # 4) Build IDs + metadata
    ids = [f"candp-{i}-{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
    metas = [{"text": chunk, "chunk": i, "source": "crime_and_punishment"} for i, chunk in enumerate(chunks)]

    # 5) Upsert to Pinecone with progress
    index = get_pc_index()
    print("Upserting to Pinecone…")
    payload = [{"id": ids[i], "values": vectors[i], "metadata": metas[i]} for i in range(len(ids))]
    total_upserts = (len(payload) + UPSERT_BATCH - 1) // UPSERT_BATCH
    for _, batch in tqdm(chunked(payload, UPSERT_BATCH), total=total_upserts, desc="Upserting"):
        index.upsert(vectors=batch, namespace=NAMESPACE)
    print("Upsert complete.")

    # 6) Show index stats
    try:
        stats = index.describe_index_stats()
        print("Index stats:", stats)
    except Exception as e:
        print("Could not fetch index stats (non-fatal):", e)

    print("Ingestion complete.")
