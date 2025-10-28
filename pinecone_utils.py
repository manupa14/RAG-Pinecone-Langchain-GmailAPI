import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "candp")
CLOUD = os.getenv("PINECONE_CLOUD", "aws")
REGION = os.getenv("PINECONE_REGION", "us-east-1")

# Embedding dimension for 'nomic-embed-text'
EMBED_DIM = 768
NAMESPACE = "c_and_p"

def get_pc_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    # create index if missing
    existing = [ix.name for ix in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
    index = pc.Index(INDEX_NAME)
    return index

def upsert_chunks(index, ids: List[str], vectors: List[List[float]], metadatas: List[Dict[str, Any]]):
    payload = [{"id": _id, "values": vec, "metadata": md} for _id, vec, md in zip(ids, vectors, metadatas)]
    # Upsert in batches
    B = 200
    for i in range(0, len(payload), B):
        index.upsert(vectors=payload[i:i+B], namespace=NAMESPACE)

def query_topk(index, vector: List[float], k: int = 5):
    res = index.query(
        vector=vector,
        top_k=k,
        include_metadata=True,
        namespace=NAMESPACE
    )
    # handle both dict/object shapes
    matches = res.get("matches") if isinstance(res, dict) else getattr(res, "matches", [])
    return matches or []
