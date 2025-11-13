# preprocess_for_rag.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional
import json

from extractors import read_preview  # your existing lightweight preview
from ontobpr_extracter import DocStub  # or ontobpr_extractor, depending on your rename


@dataclass
class TextChunk:
    case_id: str
    doc_iri: str
    rel_filename: str
    name: str
    role: Optional[str]  # can be None for now
    filetype: str
    page: int
    chunk_id: int
    text: str


def _read_text_fallback(path: Path, max_chars: int = 6000) -> str:
    """
    Use your existing read_preview as a fallback for non-PDF or
    non-critical cases (IFC, TXT, etc.).
    """
    return read_preview(path, max_chars=max_chars)


def _extract_pdf_chunks(
    pdf_path: Path,
    max_chars_per_chunk: int = 1000,
) -> List[str]:
    """
    Return a list of text chunks from a PDF, roughly max_chars_per_chunk each.
    Requires pdfminer.six, but falls back to filename if unavailable.
    """
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextContainer
    except ImportError:
        # If pdfminer isn't installed, just return a single placeholder chunk
        return [_read_text_fallback(pdf_path, max_chars=max_chars_per_chunk)]

    chunks: List[str] = []
    for page_layout in extract_pages(str(pdf_path)):
        page_text_parts = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                page_text_parts.append(element.get_text())
        page_text = "\n".join(page_text_parts).strip()
        if not page_text:
            continue

        # split into fixed-length chunks
        start = 0
        while start < len(page_text):
            end = min(len(page_text), start + max_chars_per_chunk)
            slice_ = page_text[start:end].strip()
            if slice_:
                chunks.append(slice_)
            start = end
    if not chunks:
        chunks.append(_read_text_fallback(pdf_path, max_chars=max_chars_per_chunk))
    return chunks


def preprocess_for_rag(
    case_id: str,
    run_dir: Path,
    doc_index: Iterable[DocStub],
    roles: Optional[Dict[str, str]] = None,
    max_chars_per_chunk: int = 1000,
) -> Path:
    """
    Robust preprocessing for all staged documents of a case.

    - Handles multiple file types:
        * PDF   -> pdfminer-based chunking (if available)
        * IFC   -> read as text (IFC is text-based)
        * TXT/CSV/JSON/XML/YAML/MD -> read as text
        * XLSX  -> for now: preview via read_preview (can be upgraded to openpyxl)
        * Others (DWG, images, etc.) -> preview via read_preview (usually just filename)

    - Saves output as:
        <run_dir>/rag/chunks.jsonl  (one TextChunk per line)
        <run_dir>/rag/manifest.json (per-document metadata)

    Returns the rag/ directory path.
    """
    rag_dir = run_dir / "rag"
    rag_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = rag_dir / "chunks.jsonl"
    manifest_path = rag_dir / "manifest.json"

    manifest: Dict[str, Any] = {
        "case_id": case_id,
        "documents": [],
    }

    def guess_role(stub: DocStub) -> Optional[str]:
        # Placeholder: use filename heuristics for now
        fn = stub.rel_filename.lower()
        if "application" in fn or "antrag" in fn:
            return "ApplicationForm"
        if "site" in fn and "plan" in fn:
            return "SitePlan"
        if "elevation" in fn:
            return "ElevationDrawing"
        return None

    with chunks_path.open("w", encoding="utf-8") as fout:
        for stub in doc_index:
            rel = Path(stub.rel_filename)
            ext = rel.suffix.lower()
            staged = run_dir / "Payload documents" / case_id / rel.name

            if not staged.exists():
                print(f"[preprocess] WARNING: file missing: {staged}")
                continue

            role = (roles or {}).get(stub.rel_filename) or guess_role(stub)
            base_name = stub.name or rel.stem
            text_chunks: List[str] = []

            if ext == ".pdf":
                text_chunks = _extract_pdf_chunks(staged, max_chars_per_chunk)
            elif ext in {".txt", ".csv", ".xml", ".json", ".md", ".yaml", ".yml", ".ifc"}:
                text_chunks = [_read_text_fallback(staged, max_chars=max_chars_per_chunk)]
            elif ext in {".xlsx", ".xls"}:
                # Excel: for now, just use preview; can be upgraded to openpyxl later
                text_chunks = [_read_text_fallback(staged, max_chars=max_chars_per_chunk)]
            else:
                # DWG, images, etc. – we can’t read text easily; fallback preview
                text_chunks = [_read_text_fallback(staged, max_chars=max_chars_per_chunk)]

            # Write TextChunk records
            for i, txt in enumerate(text_chunks):
                tc = TextChunk(
                    case_id=case_id,
                    doc_iri=str(stub.iri),
                    rel_filename=stub.rel_filename,
                    name=base_name,
                    role=role,
                    filetype=ext.lstrip("."),
                    page=1 + i,  # crude; for PDFs you could track real page numbers
                    chunk_id=i,
                    text=txt,
                )
                fout.write(json.dumps(asdict(tc), ensure_ascii=False) + "\n")

            manifest["documents"].append(
                {
                    "doc_iri": str(stub.iri),
                    "rel_filename": stub.rel_filename,
                    "name": base_name,
                    "role": role,
                    "filetype": ext.lstrip("."),
                    "num_chunks": len(text_chunks),
                }
            )

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[preprocess] Wrote chunks to {chunks_path}")
    print(f"[preprocess] Wrote manifest to {manifest_path}")
    return rag_dir
