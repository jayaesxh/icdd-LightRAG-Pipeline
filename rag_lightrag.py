# rag_lightrag.py
# Minimal LightRAG ↔ your pipeline adapter.
# - indexes /output/<CASE_ID>/rag/chunks.jsonl
# - provides a simple query() function
# - prints obvious progress so you can see it “do something”

from __future__ import annotations
import os, json, asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

from lightrag import LightRAG, QueryParam, EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

# OPTIONAL HF embedder (fast + local; if unavailable, we fall back to "naive" mode)
try:
    from transformers import AutoModel, AutoTokenizer
    import torch
    _HAS_HF_EMBED = True
except Exception:
    _HAS_HF_EMBED = False

def _hf_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Returns an EmbeddingFunc that LightRAG accepts.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    dim = model.config.hidden_size if hasattr(model.config, "hidden_size") else 384

    @torch.no_grad()
    def _embed(texts: List[str]):
        # mean-pooling CLS-free embedding
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state  # [B, T, H]
        mask = inputs.attention_mask.unsqueeze(-1)  # [B, T, 1]
        sum_ = (last_hidden * mask).sum(dim=1)
        den_ = mask.sum(dim=1).clamp(min=1)
        emb = (sum_ / den_).cpu().numpy()  # [B, H]
        return emb

    return EmbeddingFunc(embedding_dim=dim, func=_embed)

async def _async_build(store_dir: Path, chunks_path: Path,
                       llm_callable, llm_model_name: str,
                       use_hf_embed: bool = True):
    setup_logger("lightrag", level="INFO")

    if use_hf_embed and _HAS_HF_EMBED:
        embed = _hf_embedder(os.getenv("ICDD_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    else:
        embed = None  # LightRAG will still work in naive mode

    rag = LightRAG(
        working_dir=str(store_dir),
        llm_model_func=llm_callable,   # your HF local LLM wrapper
        llm_model_name=llm_model_name, # for metadata/logging
        embedding_func=embed,          # optional (falls back to naive if None)
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()

    # ingest chunks
    total = 0
    with chunks_path.open("r", encoding="utf-8") as f:
        batch_texts, batch_ids = [], []
        for line in f:
            obj = json.loads(line)
            txt = obj.get("text", "").strip()
            cid = obj.get("id") or f"chunk-{total}"
            if not txt:
                continue
            batch_texts.append(txt)
            batch_ids.append(cid)
            total += 1
            if len(batch_texts) >= 128:  # batch insert to show progress
                await rag.insert(batch_texts, ids=batch_ids)
                print(f"[LightRAG] inserted {len(batch_texts)} chunks… (running total {total})")
                batch_texts, batch_ids = [], []
        if batch_texts:
            await rag.insert(batch_texts, ids=batch_ids)
            print(f"[LightRAG] inserted {len(batch_texts)} chunks… (final total {total})")

    # simple stats file
    stats = {"chunks_indexed": total}
    (store_dir / "index_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[LightRAG] indexing complete. stats → {store_dir/'index_stats.json'}")

    return rag

def ensure_index(run_dir: Path, llm_callable, llm_model_name: str,
                 prefer_hf_embed: bool = True) -> Path:
    """
    Public entry: build (or reuse) a LightRAG index under run_dir/rag/lightrag_store
    """
    run_dir = Path(run_dir)
    chunks = run_dir / "rag" / "chunks.jsonl"
    if not chunks.exists():
        raise FileNotFoundError(f"Missing chunks.jsonl: {chunks}")
    store = run_dir / "rag" / "lightrag_store"
    store.mkdir(parents=True, exist_ok=True)

    # Idempotent: if stats say we’ve already indexed >0 chunks, skip rebuild.
    stats_path = store / "index_stats.json"
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text())
            if int(stats.get("chunks_indexed", 0)) > 0:
                print(f"[LightRAG] found existing index ({stats['chunks_indexed']} chunks). Reusing at {store}")
                return store
        except Exception:
            pass

    asyncio.run(_async_build(store, chunks, llm_callable, llm_model_name, use_hf_embed=prefer_hf_embed))
    return store

def query(run_dir: Path, question: str, top_k: int = 10, mode: str = "hybrid") -> Dict[str, Any]:
    """
    Convenience query with visible output. Returns dict with 'answer' and 'raw'.
    """
    async def _q():
        store = Path(run_dir) / "rag" / "lightrag_store"
        rag = LightRAG(working_dir=str(store))
        await rag.initialize_storages()
        param = QueryParam(top_k=top_k, mode=mode)   # 'hybrid' | 'local' | 'global' | 'naive'
        ans = await rag.query(question, param=param)
        return ans

    res = asyncio.run(_q())
    # LightRAG returns a string with reasoning + citations or a dict, depending on version.
    print("\n[LightRAG][query] --------")
    print(question)
    print("--------------------------")
    print(res if isinstance(res, str) else json.dumps(res, indent=2)[:2000])
    print("[LightRAG][query] --------\n")
    return {"answer": res, "raw": res}
