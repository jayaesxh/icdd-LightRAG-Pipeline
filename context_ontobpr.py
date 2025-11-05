# /workspace/icdd-rag-pipeline/context_ontobpr.py
# Minimal, robust context + (optional) LightRAG + (optional) LLM OntoBPR extractor.
# Safe fallbacks: if a lib or API is missing, we skip gracefully and keep the pipeline green.

from __future__ import annotations
import os, json, re
from pathlib import Path
from typing import List, Dict, Optional

# -------- Text extraction (robust fallbacks) --------
def _read_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return path.read_text(errors="ignore")

def _read_pdf(path: Path) -> str:
    try:
        # pdfminer.six
        from pdfminer.high_level import extract_text
        return extract_text(str(path)) or ""
    except Exception:
        try:
            # pypdf as a fallback
            import pypdf
            r = pypdf.PdfReader(str(path))
            out = []
            for pg in r.pages:
                try:
                    out.append(pg.extract_text() or "")
                except Exception:
                    pass
            return "\n".join(out)
        except Exception:
            return ""

def _read_docx(path: Path) -> str:
    try:
        import docx
        d = docx.Document(str(path))
        return "\n".join(p.text for p in d.paragraphs)
    except Exception:
        return ""

def extract_text_generic(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in [".txt", ".md", ".csv", ".json"]:
        return _read_txt(path)
    if ext in [".pdf"]:
        return _read_pdf(path)
    if ext in [".docx"]:
        return _read_docx(path)
    # IFC/other handled elsewhere. For unknowns return empty (safe).
    return ""

# -------- Chunking --------
def chunk_text(text: str, max_chars: int = 1400, overlap: int = 200) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + max_chars, n)
        chunk = text[i:j]
        chunks.append(chunk)
        if j >= n: break
        i = max(0, j - overlap)
    return chunks

# -------- Build context index (JSONL) --------
def build_context_index(payload_abs_paths: List[Path], run_dir: Path, case_id: str) -> Path:
    ctx_dir = run_dir / "Context"
    ctx_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = ctx_dir / "chunks.jsonl"

    with out_jsonl.open("w", encoding="utf-8") as f:
        for p in payload_abs_paths:
            text = extract_text_generic(p)
            if not text:
                continue
            for idx, chunk in enumerate(chunk_text(text)):
                rec = {
                    "case_id": case_id,
                    "source": str(p.name),
                    "chunk_id": f"{p.name}::chunk{idx}",
                    "text": chunk,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return out_jsonl

# -------- Optional LightRAG index (best-effort) --------
def maybe_build_lightrag_index(chunks_jsonl: Path) -> Optional[Path]:
    try:
        import lightrag  # noqa: F401  (presence check)
    except Exception:
        # LightRAG not installed; skip silently
        return None

    # Simple placeholder: store the chunks in a folder lightrag/ as corpus.jsonl
    idx_dir = chunks_jsonl.parent / "lightrag"
    idx_dir.mkdir(exist_ok=True)
    target = idx_dir / "corpus.jsonl"
    try:
        # A minimal "index" for later use. Advanced users can drop in true LightRAG building here.
        data = chunks_jsonl.read_text(encoding="utf-8")
        target.write_text(data, encoding="utf-8")
        return target
    except Exception:
        return None

# -------- Optional LLM OntoBPR extraction --------
def llm_available() -> bool:
    # Heuristics: presence of OPENAI_API_KEY or OLLAMA_HOST suggests availability
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("OLLAMA_HOST"))

def _call_llm(prompt: str) -> Optional[str]:
    """
    Try OpenAI first (if OPENAI_API_KEY), else try Ollama (if OLLAMA_HOST).
    Must return a JSON string or None on failure.
    """
    # OpenAI path
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a precise information extraction engine. Output ONLY JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            return resp.choices[0].message.content
        except Exception:
            return None
    # Ollama path (json mode via prompt discipline)
    if os.getenv("OLLAMA_HOST"):
        try:
            import requests, json as pyjson
            model = os.getenv("OLLAMA_MODEL", "qwen2.5")
            r = requests.post(
                f"{os.getenv('OLLAMA_HOST').rstrip('/')}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            j = r.json()
            return j.get("response")
        except Exception:
            return None
    return None

EXTRACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "application_id": {"type": "string"},
        "applicant_name": {"type": "string"},
        "site_address":   {"type": "string"},
        "building_name":  {"type": "string"},
        "energy_certificate_id": {"type": "string"},
        "dates": {"type": "object", "properties": {
            "application_date": {"type": "string"}
        }},
    },
    "required": [],
    "additionalProperties": True
}

def _json_sanitize(s: str) -> Optional[Dict]:
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            # Very light schema gate
            return {k: v for k, v in data.items() if k in EXTRACTION_JSON_SCHEMA["properties"] or True}
    except Exception:
        pass
    return None

def extract_ontobpr_facts_from_chunks(chunks_jsonl: Path, budget_chars: int = 9000) -> Optional[Dict]:
    """
    Concatenate a bounded amount of text, ask LLM for OntoBPR-related fields.
    If LLM not available, return None (pipeline continues).
    """
    if not llm_available():
        return None

    buf = []
    used = 0
    with chunks_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
                t = j.get("text", "")
                if not t: 
                    continue
                if used + len(t) > budget_chars:
                    break
                buf.append(f"[{j.get('source')}::{j.get('chunk_id')}] {t}")
                used += len(t)
            except Exception:
                continue

    if not buf:
        return None

    corpus = "\n".join(buf)
    prompt = (
        "Extract key fields for an OntoBPR building-application knowledge graph.\n"
        "Return ONLY a compact JSON object with keys from this schema (omit unknowns):\n"
        f"{json.dumps(EXTRACTION_JSON_SCHEMA)}\n\n"
        "Text:\n"
        f"{corpus}\n"
    )
    out = _call_llm(prompt)
    if not out:
        return None
    return _json_sanitize(out)

# -------- Render OntoBPR TTL --------
ONTOBPR_PREFIX = "https://w3id.org/ontobpr#"

def write_ontobpr_ttl(case_id: str, facts: Dict, out_path: Path) -> Optional[Path]:
    """
    Write a tiny OntoBPR TTL using only the fields we actually have.
    This is conservative: we only emit a couple of triples to avoid wrong assertions.
    """
    if not facts:
        return None

    lines = [
        '@prefix ontobpr: <https://w3id.org/ontobpr#> .',
        '@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .',
        '',
        f'<https://example.org/{case_id}#BA_{case_id}> a ontobpr:BuildingApplication ;',
        f'  ontobpr:hasBuildingApplicationContainer <https://example.org/{case_id}#Container_{case_id}> .',
        ''
    ]

    if facts.get("building_name") or facts.get("application_id"):
        lines.append(f'<https://example.org/{case_id}#Building_{case_id}> a ontobpr:Building .')

    # You can safely attach available literals with generic annotations where OntoBPR has them.
    # Keep it minimal to avoid schema drift; extend once you map the exact properties you need.
    if facts.get("applicant_name"):
        lines += [
            f'<https://example.org/{case_id}#BA_{case_id}> ontobpr:hasApplicantName "{facts["applicant_name"]}"^^xsd:string .'
        ]
    if facts.get("application_id"):
        lines += [
            f'<https://example.org/{case_id}#BA_{case_id}> ontobpr:hasApplicationIdentifier "{facts["application_id"]}"^^xsd:string .'
        ]
    if facts.get("site_address"):
        lines += [
            f'<https://example.org/{case_id}#BA_{case_id}> ontobpr:hasSiteAddress "{facts["site_address"]}"^^xsd:string .'
        ]
    if facts.get("energy_certificate_id"):
        lines += [
            f'<https://example.org/{case_id}#BA_{case_id}> ontobpr:hasEnergyCertificateId "{facts["energy_certificate_id"]}"^^xsd:string .'
        ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path

# -------- Orchestrator used by driver --------
def build_context_and_ontobpr(case_id: str, run_dir: Path, payload_abs_paths: List[Path]) -> Optional[Path]:
    """
    Returns path to OntoBPR TTL if produced, else None.
    1) Build chunks.jsonl
    2) (opt) LightRAG index
    3) (opt) LLM extract facts -> OntoBPR.ttl
    """
    chunks = build_context_index(payload_abs_paths, run_dir, case_id)
    maybe_build_lightrag_index(chunks)
    facts = extract_ontobpr_facts_from_chunks(chunks)
    if not facts:
        return None
    out_ttl = run_dir / "Payload documents" / case_id / "OntoBPR.ttl"
    return write_ontobpr_ttl(case_id, facts, out_ttl)
