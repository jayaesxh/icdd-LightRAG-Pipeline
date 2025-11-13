# /workspace/icdd-rag-pipeline/driver_upload.py
from __future__ import annotations
import os, io, json, zipfile, shutil, mimetypes, asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, XSD

# ---- Optional, guarded imports (don’t break baseline if missing) ----
try:
    # If you have this (your file was named ontobpr_extracter.py), import it
    from ontobpr_extracter import DocStub  # user-provided helper (typo in filename kept)
except Exception:
    # Fallback DocStub to avoid crashes if module missing or signature differs
    @dataclass
    class DocStub:
        iri: str
        rel_filename: str
        name: str

try:
    from pyshacl import validate as shacl_validate
except Exception:
    shacl_validate = None  # SHACL becomes optional

# ---- Namespaces (ISO 21597) ----
CT  = Namespace("https://standards.iso.org/iso/21597/-1/ed-1/en/Container#")
LS  = Namespace("https://standards.iso.org/iso/21597/-1/ed-1/en/Linkset#")
ELS = Namespace("https://standards.iso.org/iso/21597/-2/ed-1/en/ExtendedLinkset#")
ONTOBPR = Namespace("https://w3id.org/ontobpr#")

# ---- Constants / Paths ----
ROOT   = Path("/workspace/icdd-rag-pipeline")
UPLOAD = ROOT / "upload"
OUTPUT = ROOT / "output"
STATIC_RES = ROOT / "static_resources" / "ontology_resources"

REQUIRED_RESOURCES = [
    "Container.rdf",
    "Container.shapes.ttl",
    "Linkset.rdf",
    "ExtendedLinkset.rdf",
    "Part1ClassesCheck.shapes.rdf",
    "Part2ClassesCheck.shapes.ttl",
]

@dataclass
class DocSpec:
    iri: str
    rel_filename: str   # relative path inside Payload documents/<CASE_ID>/
    filetype: str       # "pdf" etc.
    format: str         # MIME, e.g. "application/pdf"
    name: str           # human label

# ----------------------------- utilities -----------------------------

def make_case_id(upload_item: Path) -> str:
    stem = upload_item.stem
    return stem

def ensure_run_dir(case_id: str) -> Path:
    run_dir = OUTPUT / case_id
    (run_dir / "Ontology resources").mkdir(parents=True, exist_ok=True)
    (run_dir / "Payload documents" / case_id).mkdir(parents=True, exist_ok=True)
    (run_dir / "Payload triples").mkdir(parents=True, exist_ok=True)
    (run_dir / "rag").mkdir(parents=True, exist_ok=True)
    return run_dir

def copy_ontology_resources(run_dir: Path) -> None:
    src = STATIC_RES
    dst = run_dir / "Ontology resources"
    for name in REQUIRED_RESOURCES:
        s = src / name
        d = dst / name
        if not s.exists():
            print(f"[warn] Missing static resource: {s}")
            continue
        shutil.copy2(s, d)

def stage_from_upload(upload_zip: Path, run_dir: Path, case_id: str) -> List[Path]:
    """Unzip into Payload documents/<CASE_ID>/ and return absolute filepaths."""
    target = run_dir / "Payload documents" / case_id
    staged: List[Path] = []
    if upload_zip.suffix.lower() == ".zip":
        with zipfile.ZipFile(upload_zip, "r") as z:
            for m in z.infolist():
                if m.is_dir():
                    continue
                # flatten one level; store bare filename under case folder
                out = target / Path(m.filename).name
                with z.open(m, "r") as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                staged.append(out)
    else:
        # If a folder/file was dropped directly
        if upload_zip.is_dir():
            for p in upload_zip.rglob("*"):
                if p.is_file():
                    out = target / p.name
                    shutil.copy2(p, out)
                    staged.append(out)
        else:
            out = target / upload_zip.name
            shutil.copy2(upload_zip, out)
            staged.append(out)
    return staged

def sanitize(stem: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)[:160]

def to_docspecs(staged_paths: List[Path], case_id: str) -> List[DocSpec]:
    out: List[DocSpec] = []
    for p in staged_paths:
        name_noext = sanitize(p.stem)
        ext = p.suffix.lower().lstrip(".") or "bin"
        mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        iri = f"https://example.org/{case_id}#Doc_{name_noext}_{case_id}"
        # rel path inside Payload documents/<CASE_ID>/
        rel = f"{case_id}/{p.name}"
        out.append(DocSpec(iri=iri, rel_filename=rel, filetype=ext, format=mime, name=name_noext))
    return out

# --------------------------- graph builders ---------------------------

def build_index_graph(case_id: str, docs: List[DocSpec]) -> Graph:
    g = Graph()
    g.bind("ct", CT)
    container = URIRef(f"https://example.org/{case_id}#Container_{case_id}")
    publisher = URIRef(f"https://example.org/{case_id}#Publisher_{case_id}")
    linkset   = URIRef(f"https://example.org/{case_id}#Linkset_{case_id}")

    # ContainerDescription
    g.add((container, RDF.type, CT.ContainerDescription))
    g.add((container, CT.conformanceIndicator, Literal("ICDD-Part1-Container")))
    g.add((container, CT.containsLinkset, linkset))

    # Publisher as ct:Party (not a string)
    g.add((publisher, RDF.type, CT.Party))
    g.add((publisher, CT.name, Literal("icdd-rag-pipeline")))
    g.add((container, CT.publisher, publisher))

    # Linkset node registration
    g.add((linkset, RDF.type, CT.Linkset))
    g.add((linkset, CT.filename, Literal("Doc_Application_Links.rdf")))

    # Documents
    for d in docs:
        d_iri = URIRef(d.iri)
        g.add((d_iri, RDF.type, CT.Document))
        g.add((d_iri, RDF.type, CT.InternalDocument))
        g.add((d_iri, CT.belongsToContainer, container))
        g.add((d_iri, CT.filename, Literal(Path(d.rel_filename).name)))
        g.add((d_iri, CT.filetype, Literal(d.filetype)))
        g.add((d_iri, CT["format"], Literal(d.format)))
        g.add((d_iri, CT.name, Literal(d.name)))
        g.add((container, CT.containsDocument, d_iri))
    return g

def build_payload_linkset(case_id: str, docs: List[DocSpec]) -> Graph:
    g = Graph()
    g.bind("ls", LS)
    g.bind("els", ELS)

    if len(docs) >= 2:
        a, b = docs[0], docs[1]
        link = URIRef(f"https://example.org/{case_id}#Link_1_{case_id}")
        from_el = URIRef(f"{link}#from")
        to_el   = URIRef(f"{link}#to")
        id_a    = URIRef(f"{from_el}#id")
        id_b    = URIRef(f"{to_el}#id")

        g.add((link, RDF.type, ELS.IsControlledBy))
        g.add((link, LS.hasFromLinkElement, from_el))
        g.add((link, LS.hasToLinkElement,   to_el))

        g.add((from_el, RDF.type, LS.LinkElement))
        g.add((from_el, LS.document, URIRef(a.iri)))
        g.add((from_el, LS.hasIdentifier, id_a))

        g.add((to_el, RDF.type, LS.LinkElement))
        g.add((to_el, LS.document, URIRef(b.iri)))
        g.add((to_el, LS.hasIdentifier, id_b))

        g.add((id_a, RDF.type, LS.StringBasedIdentifier))
        g.add((id_a, LS.identifier, Literal("whole-doc")))
        g.add((id_b, RDF.type, LS.StringBasedIdentifier))
        g.add((id_b, LS.identifier, Literal("whole-doc")))
    # It’s valid to have a linkset file even with 0 links; shapes won’t fail.
    return g

def write_ontobpr_minimal(run_dir: Path, case_id: str, docs: List[DocSpec]) -> None:
    out = run_dir / "Payload triples" / "OntoBPR.ttl"
    g = Graph()
    g.bind("ontobpr", ONTOBPR)

    ba = URIRef(f"https://example.org/{case_id}#BA_{case_id}")
    cont = URIRef(f"https://example.org/{case_id}#Container_{case_id}")
    g.add((ba, RDF.type, ONTOBPR.BuildingApplication))
    g.add((ba, ONTOBPR.hasBuildingApplicationContainer, cont))

    if docs:
        b = URIRef(f"https://example.org/{case_id}#Building_{case_id}")
        g.add((b, RDF.type, ONTOBPR.Building))
        g.add((b, ONTOBPR.hasBuildingDocument, URIRef(docs[0].iri)))

    out.write_text(g.serialize(format="turtle"))

# ------------------------- SHACL & checks -----------------------------

def structural_check(run_dir: Path, case_id: str) -> Tuple[bool, List[str]]:
    errs = []
    root_ok = all((run_dir / p).exists() for p in [
        f"Ontology resources",
        f"Payload documents/{case_id}",
        f"Payload triples",
        "index.rdf",
    ])
    if not root_ok:
        errs.append("Missing required folders or index.rdf.")
    if not (run_dir / "Payload triples" / "Doc_Application_Links.rdf").exists():
        errs.append("Missing Payload triples/Doc_Application_Links.rdf.")
    for name in REQUIRED_RESOURCES:
        if not (run_dir / "Ontology resources" / name).exists():
            errs.append(f"Missing Ontology resources/{name}")

    return (len(errs) == 0, errs)

def coherence_check(run_dir: Path, case_id: str) -> Tuple[bool, List[str]]:
    errs = []
    g = Graph()
    idx = run_dir / "index.rdf"
    linkset_file = run_dir / "Payload triples" / "Doc_Application_Links.rdf"
    try:
        g.parse(idx)
    except Exception as e:
        return (False, [f"index.rdf parse error: {e}"])

    container = URIRef(f"https://example.org/{case_id}#Container_{case_id}")
    linkset   = URIRef(f"https://example.org/{case_id}#Linkset_{case_id}")

    if (container, RDF.type, CT.ContainerDescription) not in g:
        errs.append("Container node not typed as ct:ContainerDescription.")
    if (container, CT.containsLinkset, linkset) not in g:
        errs.append("ct:containsLinkset missing.")
    if (linkset, RDF.type, CT.Linkset) not in g:
        errs.append("Linkset node not typed ct:Linkset.")
    if (linkset, CT.filename, Literal("Doc_Application_Links.rdf")) not in g:
        errs.append("Linkset ct:filename mismatch.")

    # Docs
    for _, _, doc in g.triples((container, CT.containsDocument, None)):
        if (doc, RDF.type, CT.Document) not in g or (doc, RDF.type, CT.InternalDocument) not in g:
            errs.append(f"Document {doc} missing required types.")

    return (len(errs) == 0, errs)

def load_shapes_graph(run_dir: Path) -> Graph:
    shapes = Graph()
    shapes_dir = run_dir / "Ontology resources"
    for p in shapes_dir.glob("*"):
        if p.suffix.lower() in (".ttl", ".rdf", ".xml"):
            try:
                shapes.parse(p)
            except Exception as e:
                print(f"[warn] failed to parse shapes {p}: {e}")
    return shapes

def run_shacl_validation(run_dir: Path, case_id: str) -> Tuple[bool, Optional[str]]:
    if shacl_validate is None:
        print("[SHACL] pyshACL not installed; skipping validation.")
        return (True, None)

    data = Graph()
    try:
        data.parse(run_dir / "index.rdf")
    except Exception as e:
        return (False, f"index.rdf parse error: {e}")
    # Linkset (RDF/XML)
    try:
        data.parse(run_dir / "Payload triples" / "Doc_Application_Links.rdf")
    except Exception as e:
        return (False, f"Doc_Application_Links.rdf parse error: {e}")
    # OntoBPR is optional; include if exists
    onto = run_dir / "Payload triples" / "OntoBPR.ttl"
    if onto.exists():
        try:
            data.parse(onto)
        except Exception as e:
            print(f"[warn] OntoBPR.ttl parse issue: {e}")

    shapes = load_shapes_graph(run_dir)
    if len(shapes) == 0:
        print("[SHACL] No shapes loaded; skipping.")
        return (True, "No shapes loaded")

    conforms, report_graph, report_text = shacl_validate(
        data_graph=data,
        shacl_graph=shapes,
        inference="rdfs",
        allow_infos=True,
        allow_warnings=True,
    )
    print("\n=== SHACL RESULT ===")
    print(f"Conforms: {conforms}")
    print(report_text)
    return (bool(conforms), report_text)

# ------------------------- Preprocessing (RAG) ------------------------

def extract_text_pdf(pdf_path: Path) -> str:
    try:
        from pdfminer.high_level import extract_text
        return extract_text(str(pdf_path)) or ""
    except Exception:
        try:
            return pdf_path.read_text(errors="ignore")
        except Exception:
            return ""

def preprocess_and_dump(run_dir: Path, case_id: str, docs: List[DocSpec]) -> Path:
    """Very simple page-less extractor to JSONL; OK as a baseline for LightRAG."""
    rag_dir = run_dir / "rag"
    chunks_path = rag_dir / "chunks.jsonl"
    manifest_path = rag_dir / "manifest.json"

    records = []
    for d in docs:
        abs_path = run_dir / "Payload documents" / d.rel_filename
        text = ""
        if d.filetype == "pdf":
            text = extract_text_pdf(abs_path)
        else:
            try:
                text = abs_path.read_text(errors="ignore")
            except Exception:
                text = ""
        records.append({
            "doc_iri": d.iri,
            "doc_name": d.name,
            "rel_path": d.rel_filename,
            "mime": d.format,
            "text": text[:5_000_000],  # keep it reasonable
        })

    with open(chunks_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {
        "case_id": case_id,
        "num_docs": len(docs),
        "items": [{"iri": d.iri, "rel": d.rel_filename, "name": d.name, "mime": d.format} for d in docs],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[preprocess] Wrote chunks to {chunks_path}")
    print(f"[preprocess] Wrote manifest to {manifest_path}")
    return chunks_path

# ------------------------------ writer --------------------------------

def write_files_and_zip(run_dir: Path, case_id: str, g_index: Graph, g_link: Graph) -> Path:
    # Write graphs
    (run_dir / "index.rdf").write_bytes(g_index.serialize(format="application/rdf+xml").encode("utf-8"))
    (run_dir / "Payload triples" / "Doc_Application_Links.rdf").write_bytes(
        g_link.serialize(format="application/rdf+xml").encode("utf-8")
    )

    # Zip with <CASE_ID>/ as the archive root
    out_zip = OUTPUT / f"ICDD_{case_id}.icdd"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in run_dir.rglob("*"):
            if p.is_file():
                arcname = f"{case_id}/{p.relative_to(run_dir)}"
                z.write(p, arcname=arcname)
    print(f"✅ ICDD written: {out_zip}")
    return out_zip

# ------------------------------- main ---------------------------------

def discover_upload_items(upload_root: Path) -> List[Path]:
    if not upload_root.exists():
        return []
    items = [p for p in upload_root.iterdir() if p.is_file() or p.is_dir()]
    zips_first = sorted([p for p in items if p.suffix.lower() == ".zip"])
    others = sorted([p for p in items if p.suffix.lower() != ".zip"])
    return zips_first + others

def main():
    items = discover_upload_items(UPLOAD)
    if not items:
        print("No submissions in upload/.")
        return

    print("Found submissions:")
    for it in items:
        print(" -", it)

    for item in items:
        case_id = make_case_id(item)
        print(f"\n=== Processing {item.name} → CASE_ID={case_id} ===")
        run_dir = ensure_run_dir(case_id)
        copy_ontology_resources(run_dir)

        staged = stage_from_upload(item, run_dir, case_id)
        docs = to_docspecs(staged, case_id)

        # Preprocess for RAG
        preprocess_and_dump(run_dir, case_id, docs)

        # Build graphs
        g_index = build_index_graph(case_id, docs)
        g_link  = build_payload_linkset(case_id, docs)

        # Minimal OntoBPR seed (we will extend with LLM+LightRAG next)
        write_ontobpr_minimal(run_dir, case_id, docs)

        # Write & zip
        write_files_and_zip(run_dir, case_id, g_index, g_link)

        # Structural / coherence checks
        ok, errs = structural_check(run_dir, case_id)
        print("\n=== Structural check:", "OK" if ok else "FAIL", "===")
        if not ok:
            for e in errs: print(" -", e)

        cok, cerrs = coherence_check(run_dir, case_id)
        print("=== Coherence check:", "OK" if cok else "FAIL", "===")
        if not cok:
            for e in cerrs: print(" -", e)

        # SHACL
        run_shacl_validation(run_dir, case_id)

if __name__ == "__main__":
    main()
