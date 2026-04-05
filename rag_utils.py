"""RAG pipeline utility functions."""
from typing import List, Dict, Optional
import hashlib

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:min(start+chunk_size, len(text))])
        start += chunk_size - overlap
    return chunks

def generate_chunk_id(text: str, source: str) -> str:
    return hashlib.sha256(f"{source}:{text}".encode()).hexdigest()[:16]

def format_context(docs: List[Dict], max_tokens: int = 3000) -> str:
    parts, total = [], 0
    for d in docs:
        f = f"[{d.get('source','?')} | {d.get('score',0):.2f}]\n{d.get('content','')}\n"
        if total + len(f) > max_tokens * 4: break
        parts.append(f); total += len(f)
    return "\n---\n".join(parts)

def build_prompt(query: str, context: str, system: Optional[str] = None) -> str:
    s = system or "Answer based on the provided context."
    return f"{s}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

def evaluate_retrieval(retrieved: List[str], relevant: List[str]) -> Dict[str, float]:
    r_set, rel_set = set(retrieved), set(relevant)
    if not rel_set: return {"precision":0.0,"recall":0.0,"f1":0.0}
    tp = len(r_set & rel_set)
    p = tp/len(r_set) if r_set else 0.0
    r = tp/len(rel_set)
    f1 = 2*p*r/(p+r) if (p+r)>0 else 0.0
    return {"precision":round(p,4),"recall":round(r,4),"f1":round(f1,4)}
