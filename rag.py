from typing import Tuple, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from pinecone_utils import get_pc_index, query_topk

SYSTEM = """You answer questions ONLY from the provided context. 
If the answer isn't clearly in the context, say you don't know.
Cite short excerpts from the context when helpful."""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer concisely, then list brief citations.")
])

def build_context_snippets(matches, max_chars: int = 1800) -> Tuple[str, List[str]]:
    # Turn matches into a context block and capture citation snippets
    snippets, cites = [], []
    for m in matches:
        meta = m["metadata"] if isinstance(m, dict) else getattr(m, "metadata", {})
        txt = meta.get("text", "")
        score = m.get("score") if isinstance(m, dict) else getattr(m, "score", None)
        if not txt:
            continue
        excerpt = txt.strip().replace("\n", " ")
        if len(" ".join(snippets)) + len(excerpt) > max_chars:
            break
        snippets.append(excerpt)
        chunk = meta.get("chunk", "?")
        cites.append(f"(chunk {chunk}, score={score:.3f}) {excerpt[:160]}...")
    return "\n\n---\n\n".join(snippets), cites

def answer(question: str) -> dict:
    # embed query, fetch top-k
    emb = OllamaEmbeddings(model="nomic-embed-text")
    qvec = emb.embed_query(question)
    matches = query_topk(get_pc_index(), qvec, k=5)

    context, citations = build_context_snippets(matches)

    llm = ChatOllama(model="qwen2:7b", temperature=0.2)
    msgs = PROMPT.format_messages(question=question, context=context if context else "(no matches)")
    resp = llm.invoke(msgs)
    content = resp.content if hasattr(resp, "content") else str(resp)

    return {
        "answer": content.strip(),
        "citations": citations,
        "num_matches": len(matches),
    }
