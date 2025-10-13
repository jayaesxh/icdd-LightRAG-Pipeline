# /workspace/icdd-rag-pipeline/src/lightrag_client.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple
import json

# ---------- Public API ----------

def build_fallback_index(
    corpus_dir: str,
    index_dir: str = "/workspace/icdd-rag-pipeline/.lightrag_index",
    chunk_size: int = 800,
    overlap: int = 200,
    embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    cdir = Path(corpus_dir)
    idir = Path(index_dir)
    idir.mkdir(parents=True, exist_ok=True)

    txt_files = list(cdir.rglob("*.txt"))
    if not txt_files:
        print(f"[LightRAG-lite] No .txt files in {cdir}.")
        return

    # Read files
    files: List[Tuple[str, str]] = []
    for p in txt_files:
        try:
            files.append((str(p), p.read_text(encoding="utf-8", errors="ignore")))
        except Exception:
            pass

    # Chunk files (low-level)
    chunks: List[Dict] = []
    for path, txt in files:
        for t in _split_with_overlap(txt, chunk_size, overlap):
            if t.strip():
                chunks.append({"path": path, "text": t})

    # Embed chunks
    emb_chunks = _embed([c["text"] for c in chunks], embed_model_id)
    for rec, vec in zip(chunks, emb_chunks):
        rec["embedding"] = vec

    # Summaries (high-level)
    summaries: List[Dict] = []
    for path, txt in files:
        s = _first_n_sentences(txt, n=2)
        summaries.append({"path": path, "summary": s})

    emb_summ = _embed([s["summary"] for s in summaries], embed_model_id)
    for rec, vec in zip(summaries, emb_summ):
        rec["embedding"] = vec

    # Write index
    with (idir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    with (idir / "summaries.jsonl").open("w", encoding="utf-8") as f:
        for s in summaries:
            f.write(json.dumps(s) + "\n")

    print(f"[LightRAG-lite] Indexed {len(files)} files → {len(chunks)} chunks, {len(summaries)} summaries.")
    print(f"[LightRAG-lite] Index saved under: {idir}")

def dual_level_retrieve(
    query: str,
    index_dir: str = "/workspace/icdd-rag-pipeline/.lightrag_index",
    top_k_low: int = 5,
    top_k_high: int = 5,
    embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> str:
    # Try official LightRAG first (library optional)
    try:
        import lightrag  # noqa: F401
        # TODO: integrate real API here if/when available.
    except Exception:
        pass

    # Fallback index
    idir = Path(index_dir)
    chunks_path = idir / "chunks.jsonl"
    summ_path   = idir / "summaries.jsonl"
    if not chunks_path.exists() or not summ_path.exists():
        return ""  # main.py handles empty context gracefully

    q_emb = _embed([query], embed_model_id)[0]

    # High-level
    highs = []
    with summ_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rec["score"] = _cosine(q_emb, rec["embedding"])
            highs.append(rec)
    highs.sort(key=lambda r: r["score"], reverse=True)
    high_txt = "\n".join(f"[High {i+1}] {r['summary']}" for i, r in enumerate(highs[:top_k_high]))

    # Low-level
    lows = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rec["score"] = _cosine(q_emb, rec["embedding"])
            lows.append(rec)
    lows.sort(key=lambda r: r["score"], reverse=True)
    low_txt = "\n".join(f"[Low {i+1}] {r['text'][:500]}" for i, r in enumerate(lows[:top_k_low]))

    return f"--- LightRAG dual-level context ---\n{high_txt}\n---\n{low_txt}\n--- End LightRAG context ---"


# ---------- Helpers (HF Transformers mean pooling) ----------

_EMBED_MODEL = None
_TOKENIZER = None
_EMBED_MODEL_ID = None

def _load_embedder(model_id: str):
    global _EMBED_MODEL, _TOKENIZER, _EMBED_MODEL_ID
    if _EMBED_MODEL is None or _EMBED_MODEL_ID != model_id:
        from transformers import AutoModel, AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(model_id)
        _EMBED_MODEL = Aut
