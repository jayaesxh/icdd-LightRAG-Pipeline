# /workspace/icdd-rag-pipeline/ontobpr_extracter.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional, List, Dict, Any
import json

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF

# --- Existing namespaces ------------------------------------------------------

ONTOBPR = Namespace("https://w3id.org/ontobpr#")
CT      = Namespace("https://standards.iso.org/iso/21597/-1/ed-1/en/Container#")


# --- Existing DocStub ---------------------------------------------------------

@dataclass
class DocStub:
    iri: URIRef
    rel_filename: str
    name: Optional[str] = None


def _hash_base(case_id: str) -> str:
    # Use hash-delimited base to satisfy platform import preference
    return f"https://example.org/{case_id}#"


# === NEW: text chunk dataclass ===============================================

@dataclass
class TextChunk:
    case_id: str
    doc_iri: str
    rel_filename: str
    name: str
    page: int
    chunk_id: int
    text: str


# === NEW: low-level PDF text extraction (uses pdfminer.six) ===================

def _extract_pdf_chunks(
    pdf_path: Path,
    case_id: str,
    doc_iri: URIRef,
    rel_filename: str,
    name: str,
    max_chars: int = 1000,
) -> List[TextChunk]:
    """
    Extract text from a PDF file page-by-page and split into chunks of up to
    max_chars characters. Requires pdfminer.six to be installed.
    """
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextContainer
    except ImportError as e:
        raise RuntimeError(
            "pdfminer.six is required for PDF text extraction. "
            "Install it in your environment with 'pip install pdfminer.six'."
        ) from e

    chunks: List[TextChunk] = []

    for page_no, page_layout in enumerate(extract_pages(str(pdf_path)), start=1):
        page_parts: List[str] = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                page_parts.append(element.get_text())
        page_text = "\n".join(page_parts).strip()
        if not page_text:
            continue

        # Simple fixed-length chunking; can be refined later (sentence-based, etc.)
        start = 0
        chunk_idx = 0
        while start < len(page_text):
            end = min(len(page_text), start + max_chars)
            text_slice = page_text[start:end].strip()
            if text_slice:
                chunks.append(
                    TextChunk(
                        case_id=case_id,
                        doc_iri=str(doc_iri),
                        rel_filename=rel_filename,
                        name=name,
                        page=page_no,
                        chunk_id=chunk_idx,
                        text=text_slice,
                    )
                )
                chunk_idx += 1
            start = end

    return chunks


# === NEW: generic preprocessing over all docs =================================

def preprocess_documents_for_case(
    case_id: str,
    run_dir: Path,
    doc_index: Iterable[DocStub],
    max_chars: int = 1000,
) -> Path:
    """
    Robust preprocessing step:

      - Locates each staged document for the case under:
            run_dir / 'Payload documents' / case_id / <filename>
      - For PDFs: extracts text with pdfminer.six and splits into TextChunks.
      - For non-PDFs: reads as UTF-8 text (best-effort) into a single chunk.
      - Writes one JSONL file per document into run_dir / 'rag' / 'docs'.
      - Writes a manifest.json summarizing all docs and their chunk counts.

    Returns the 'rag' directory path.
    """
    rag_dir = run_dir / "rag"
    docs_dir = rag_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {"case_id": case_id, "documents": []}

    for stub in doc_index:
        rel = Path(stub.rel_filename)
        # Your rel_filename currently looks like just the file name
        # so we resolve as: <run_dir>/Payload documents/<CASE>/<filename>
        staged_path = run_dir / "Payload documents" / case_id / rel.name

        if not staged_path.exists():
            # non-fatal; just log and continue
            print(f"[preprocess] WARNING: staged file not found: {staged_path}")
            continue

        doc_name = stub.name or rel.stem
        all_chunks: List[TextChunk] = []

        if staged_path.suffix.lower() == ".pdf":
            all_chunks = _extract_pdf_chunks(
                pdf_path=staged_path,
                case_id=case_id,
                doc_iri=stub.iri,
                rel_filename=stub.rel_filename,
                name=doc_name,
                max_chars=max_chars,
            )
        else:
            # Best-effort plain-text read for non-PDF files
            try:
                txt = staged_path.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                txt = ""
            if txt:
                all_chunks = [
                    TextChunk(
                        case_id=case_id,
                        doc_iri=str(stub.iri),
                        rel_filename=stub.rel_filename,
                        name=doc_name,
                        page=1,
                        chunk_id=0,
                        text=txt,
                    )
                ]

        # Write per-document JSONL file
        out_jsonl = docs_dir / f"{rel.stem}.jsonl"
        with out_jsonl.open("w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(json.dumps(asdict(chunk), ensure_ascii=False))
                f.write("\n")

        manifest["documents"].append(
            {
                "doc_iri": str(stub.iri),
                "rel_filename": stub.rel_filename,
                "name": doc_name,
                "num_chunks": len(all_chunks),
                "staged_path": str(staged_path),
                "jsonl_path": str(out_jsonl),
            }
        )

    # Write manifest once at the end
    rag_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = rag_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")

    print(f"[preprocess] Wrote manifest to {manifest_path}")
    return rag_dir


# === Existing OntoBPR attachment (kept, now calling preprocessing) ============

def index_and_extract_for_ontobpr(
    case_id: str,
    run_dir: Path,
    doc_index: Iterable[DocStub],
) -> Path:
    """
    Minimal, safe OntoBPR attachment + preprocessing:

      1. Run robust preprocessing to create rag/manifest + JSONL chunks.
      2. Create a tiny OntoBPR graph:

         - ontobpr:BuildingApplication BA_<CASE>
             ontobpr:hasBuildingApplicationContainer -> ct:ContainerDescription
         - ontobpr:Building Building_<CASE>
             ontobpr:hasBuildingDocument -> chosen application doc

    Produces: Payload triples/OntoBPR.ttl
    """
    # Step 1: preprocessing (text extraction). This is the core of your thesis'
    # "robust preprocessing" and is LightRAG-ready.
    preprocess_documents_for_case(case_id, run_dir, doc_index)

    # Step 2: same minimal OntoBPR graph as before
    base = _hash_base(case_id)
    g = Graph()
    g.bind("ontobpr", ONTOBPR)
    g.bind("ct", CT)

    container = URIRef(f"{base}Container_{case_id}")
    ba        = URIRef(f"{base}BA_{case_id}")
    bldg      = URIRef(f"{base}Building_{case_id}")

    docs = list(doc_index)
    app_doc = None
    for d in docs:
        fn = d.rel_filename.lower()
        if "application" in fn or "antrag" in fn:
            app_doc = d
            break
    if app_doc is None and docs:
        app_doc = docs[0]

    # BA triples
    g.add((ba, RDF.type, ONTOBPR.BuildingApplication))
    g.add((ba, ONTOBPR.hasBuildingApplicationContainer, container))

    # Building triples (optional, but useful to show link to a doc)
    if app_doc:
        g.add((bldg, RDF.type, ONTOBPR.Building))
        g.add((bldg, ONTOBPR.hasBuildingDocument, app_doc.iri))

    out = run_dir / "Payload triples" / "OntoBPR.ttl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(g.serialize(format="turtle"), encoding="utf-8")
    print(f"[ontobpr] Wrote OntoBPR.ttl to {out}")
    return out
