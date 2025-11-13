from __future__ import annotations
import json, asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

# Optional dependencies; keep imports local to avoid hard failures
def _get_lightrag():
    from lightrag import LightRAG, QueryParam  # pip install lightrag-hku
    return LightRAG, QueryParam

async def build_index(case_id: str, rag_dir: Path,
                      embedding_func=None, llm_model_func=None) -> object:
    """Build a LightRAG index from rag/chunks.jsonl for one case."""
    chunks_path = rag_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(chunks_path)

    LightRAG, _ = _get_lightrag()
    rag = LightRAG(working_dir=str(rag_dir),
                   embedding_func=embedding_func,
                   llm_model_func=llm_model_func)
    await rag.initialize_storages()

    texts: List[str] = []
    meta:  List[Dict[str, Any]] = []
    with chunks_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            texts.append(rec["text"])
            rec["_chunk_id"] = i
            meta.append(rec)
    # Insert with ids+metadata so we can point back to chunks
    rag.insert(texts=texts, ids=[f"{case_id}__{i}" for i in range(len(texts))], metadatas=meta)
    return rag

async def aquery(rag: object, question: str, topk: int = 12) -> Dict[str, Any]:
    """Hybrid query; return text snippets + metadatas for downstream prompts."""
    _, QueryParam = _get_lightrag()
    res = await rag.aquery(question, param=QueryParam(mode="hybrid", top_k=topk))
    # Expect res like dict with 'nodes'/'chunks' depending on version; normalize:
    hits = []
    for hit in res.get("chunks", []):
        hits.append({"text": hit.get("text", ""), "meta": hit.get("metadata", {})})
    if not hits and isinstance(res, dict):  # older variants
        for hit in res.get("nodes", []):
            hits.append({"text": hit.get("text", ""), "meta": hit.get("metadata", {})})
    return {"hits": hits}
