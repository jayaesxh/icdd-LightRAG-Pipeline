# driver_upload.py  (final, copy-paste)
from __future__ import annotations
import os, zipfile, shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, XSD

from ontobpr_extractor import index_and_extract_for_ontobpr, DocStub

# ==== Constants / Paths ====
ROOT       = Path("/workspace/icdd-rag-pipeline")
UPLOAD     = ROOT / "upload"
OUTPUT     = ROOT / "output"
STATIC_OR  = ROOT / "static_resources" / "ontology_resources"  # your known-good copies

CT  = Namespace("https://standards.iso.org/iso/21597/-1/ed-1/en/Container#")
LS  = Namespace("https://standards.iso.org/iso/21597/-1/ed-1/en/Linkset#")
ELS = Namespace("https://standards.iso.org/iso/21597/-2/ed-1/en/ExtendedLinkset#")

# ==== Data structures ====
@dataclass
class DocSpec:
    iri: URIRef
    rel_filename: str  # relative to 'Payload documents/<case_id>/'
    filetype: str      # 'pdf', 'jpg', ...
    format: str        # MIME type
    name: str          # short name for ct:name

# ==== Helpers ====
def make_case_id(p: Path) -> str:
    stem = p.stem
    return stem.replace(" ", "_")

def ensure_run_dir(case_id: str) -> Path:
    run_dir = OUTPUT / case_id
    (run_dir / "Payload documents" / case_id).mkdir(parents=True, exist_ok=True)
    (run_dir / "Payload triples").mkdir(parents=True, exist_ok=True)
    (run_dir / "Ontology resources").mkdir(parents=True, exist_ok=True)

    # Copy official ontology + shapes into the container
    if STATIC_OR.exists():
        for src in STATIC_OR.rglob("*"):
            if src.is_file():
                rel = src.relative_to(STATIC_OR)
                dst = run_dir / "Ontology resources" / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    else:
        print("WARNING: static ontology resources not found at", STATIC_OR)

    return run_dir

def unzip_to(src_zip: Path, dest_dir: Path) -> List[Path]:
    out = []
    with zipfile.ZipFile(src_zip, "r") as zf:
        for m in zf.infolist():
            if m.is_dir():
                continue
            target = dest_dir / Path(m.filename).name
            with zf.open(m) as inp, open(target, "wb") as outfp:
                outfp.write(inp.read())
            out.append(target)
    return out

def stage_upload_to_payload(item: Path, run_dir: Path, case_id: str) -> List[Path]:
    dest = run_dir / "Payload documents" / case_id
    dest.mkdir(parents=True, exist_ok=True)
    staged = []
    if item.is_dir():
        for p in item.rglob("*"):
            if p.is_file():
                tgt = dest / p.name
                shutil.copy2(p, tgt)
                staged.append(tgt)
    elif item.suffix.lower() == ".zip":
        staged.extend(unzip_to(item, dest))
    else:
        tgt = dest / item.name
        shutil.copy2(item, tgt)
        staged.append(tgt)
    return staged

def slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "-" for ch in s)

def to_docspecs(staged_abs_paths: List[Path], case_id: str) -> List[DocSpec]:
    out: List[DocSpec] = []
    base = f"https://example.org/{case_id}#"
    for p in staged_abs_paths:
        ext = p.suffix.lower().lstrip(".") or "bin"
        mime = "application/pdf" if ext == "pdf" else "application/octet-stream"
        rel_filename = f"{case_id}/{p.name}"
        name = slug(p.stem)
        iri  = URIRef(f"{base}Doc_{name}_{case_id}")
        out.append(DocSpec(iri=iri, rel_filename=rel_filename, filetype=ext, format=mime, name=name))
    return out

# ==== Graph builders ====
def build_index_graph(case_id: str, docs: List[DocSpec]) -> Graph:
    g = Graph()
    g.bind("ct", CT); g.bind("ls", LS); g.bind("els", ELS)

    base = f"https://example.org/{case_id}#"
    container = URIRef(f"{base}Container_{case_id}")
    linkset   = URIRef(f"{base}Linkset_{case_id}")

    # Required typing for SHACL (official shapes expect these classes)
    g.add((container, RDF.type, CT.ContainerDescription))
    g.add((linkset,   RDF.type, CT.Linkset))

    # Minimal required properties
    g.add((container, CT.conformanceIndicator, Literal("ICDD-Part1-Container", datatype=XSD.string)))
    g.add((container, CT.containsLinkset, linkset))

    # Publisher as ct:Party with ct:name (fixes earlier ClassConstraint issues)
    publisher = URIRef(f"{base}Publisher_{case_id}")
    g.add((publisher, RDF.type, CT.Party))
    g.add((publisher, CT.name, Literal("icdd-rag-pipeline", datatype=XSD.string)))
    g.add((container, CT.publisher, publisher))

    # Documents + membership + metadata
    for d in docs:
        g.add((d.iri, RDF.type, CT.Document))
        g.add((d.iri, RDF.type, CT.InternalDocument))
        g.add((d.iri, CT.belongsToContainer, container))
        g.add((d.iri, CT.filename, Literal(Path(d.rel_filename).name, datatype=XSD.string)))
        g.add((d.iri, CT.filetype, Literal(d.filetype, datatype=XSD.string)))
        g.add((d.iri, CT[ "format"], Literal(d.format, datatype=XSD.string)))  # avoid .format attr
        g.add((d.iri, CT.name,     Literal(d.name, datatype=XSD.string)))
        g.add((container, CT.containsDocument, d.iri))

    # Linkset entry points back to payload triples file name
    g.add((linkset, CT.filename, Literal("Doc_Application_Links.rdf", datatype=XSD.string)))
    return g

def build_payload_linkset(case_id: str, docs: List[DocSpec]) -> Graph:
    """
    Create a tiny linkset with one IsControlledBy between first two docs if >=2.
    """
    g = Graph()
    g.bind("ls", LS); g.bind("els", ELS); g.bind("ct", CT)
    base = f"https://example.org/{case_id}#"

    if len(docs) >= 2:
        d1, d2 = docs[0], docs[1]
        link  = URIRef(f"{base}Link_1_{case_id}")
        fromE = URIRef(f"{base}Link_1_{case_id}#from")
        toE   = URIRef(f"{base}Link_1_{case_id}#to")
        idF   = URIRef(f"{base}Link_1_{case_id}#id_from")
        idT   = URIRef(f"{base}Link_1_{case_id}#id_to")

        g.add((link, RDF.type, ELS.IsControlledBy))
        g.add((fromE, RDF.type, LS.LinkElement))
        g.add((toE,   RDF.type, LS.LinkElement))
        g.add((idF,   RDF.type, LS.StringBasedIdentifier))
        g.add((idT,   RDF.type, LS.StringBasedIdentifier))

        g.add((fromE, LS.document, d1.iri))
        g.add((fromE, LS.hasIdentifier, idF))
        g.add((toE,   LS.document, d2.iri))
        g.add((toE,   LS.hasIdentifier, idT))
        g.add((idF,   LS.identifier, Literal("whole-doc", datatype=XSD.string)))
        g.add((idT,   LS.identifier, Literal("whole-doc", datatype=XSD.string)))

        g.add((link,  LS.hasFromLinkElement, fromE))
        g.add((link,  LS.hasToLinkElement,   toE))

    return g

# ==== Writers / Validators ====
def write_files_and_zip(case_id: str, run_dir: Path, g_index: Graph, g_link: Graph) -> Path:
    (run_dir / "index.rdf").write_bytes(g_index.serialize(format="application/rdf+xml").encode("utf-8"))
    (run_dir / "Payload triples" / "Doc_Application_Links.rdf").write_bytes(
        g_link.serialize(format="application/rdf+xml").encode("utf-8")
    )
    out_zip = OUTPUT / f"ICDD_{case_id}.icdd"
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in run_dir.rglob("*"):
            z.write(p, arcname=str(p.relative_to(run_dir)))
    return out_zip

def structural_check(run_dir: Path, case_id: str) -> Tuple[bool, List[str]]:
    errs = []
    # required payload triples file
    if not (run_dir / "Payload triples" / "Doc_Application_Links.rdf").exists():
        errs.append("Payload file not found: Doc_Application_Links.rdf")
    # ensure Ontology resources exist (Container.rdf, Linkset.rdf, ExtendedLinkset.rdf, and shapes)
    need = ["Container.rdf", "Linkset.rdf", "ExtendedLinkset.rdf"]
    for n in need:
        if not (run_dir / "Ontology resources" / n).exists():
            errs.append(f"Missing ontology resource: {n}")
    return (len(errs) == 0, errs)

def coherence_check(run_dir: Path, case_id: str) -> Tuple[bool, List[str]]:
    """
    Sanity: container contains docs + linkset typed.
    """
    g = Graph()
    g.parse(str(run_dir / "index.rdf"))
    base = f"https://example.org/{case_id}#"
    cont = URIRef(f"{base}Container_{case_id}")
    lset = URIRef(f"{base}Linkset_{case_id}")
    errs = []
    if not list(g.triples((cont, CT.containsLinkset, lset))):
        errs.append("Container does not ct:containsLinkset the Linkset node.")
    if not list(g.triples((lset, RDF.type, CT.Linkset))):
        errs.append("Linkset node is not typed ct:Linkset.")
    if not list(g.triples((None, CT.belongsToContainer, cont))):
        errs.append("No document belongsToContainer the Container.")
    return (len(errs) == 0, errs)

def run_shacl_validation(run_dir: Path) -> Tuple[bool, str]:
    """
    Load data (index + payload links) + shapes from:
    - run_dir/Ontology resources (+ subfolders)
    - static_resources/ontology_resources
    """
    from pyshacl import validate

    data_graph = Graph()
    data_graph.parse(str(run_dir / "index.rdf"))
    links = run_dir / "Payload triples" / "Doc_Application_Links.rdf"
    if links.exists():
        data_graph.parse(str(links))

    shapes_graph = Graph()
    candidates = [
        run_dir / "Ontology resources",
        run_dir / "Ontology resources" / "shapes",
        STATIC_OR,
    ]
    def _fmt(p: Path):
        s = p.suffix.lower()
        if s == ".ttl": return "turtle"
        if s in (".rdf", ".owl", ".xml"): return "xml"
        return None
    loaded = 0
    for base in candidates:
        if not base or not base.exists(): continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".ttl", ".rdf", ".owl", ".xml"):
                try:
                    shapes_graph.parse(str(p), format=_fmt(p))
                    loaded += 1
                except Exception:
                    pass

    conforms, rep_g, rep_t = validate(
        data_graph,
        shacl_graph=shapes_graph,
        inference="rdfs",
        advanced=True,
        debug=False,
    )
    return (bool(conforms), rep_t)

# ==== MAIN ====
def main():
    items = sorted([str(p) for p in UPLOAD.iterdir() if not p.name.startswith(".")])
    if not items:
        print("No submissions found in", UPLOAD)
        return

    print("Found submissions:")
    for s in items:
        print(" -", s)

    for item in items:
        p = Path(item)
        case_id = make_case_id(p)
        print(f"\n=== Processing {p.name} → CASE_ID={case_id} ===")

        run_dir = ensure_run_dir(case_id)
        staged_abs = stage_upload_to_payload(p, run_dir, case_id)
        if not staged_abs:
            print("No files discovered; skipping.")
            continue

        # 1) build graphs
        docs = to_docspecs(staged_abs, case_id)
        g_index = build_index_graph(case_id, docs)
        g_link  = build_payload_linkset(case_id, docs)

        # 2) OntoBPR via LLM + LightRAG (best-effort, never fatal)
        try:
            rag_dir = run_dir / "rag"
            doc_stubs = [DocStub(iri=str(d.iri), abs_path=Path(run_dir / "Payload documents" / d.rel_filename), name=d.name) for d in docs]
            ttl_text, fields = index_and_extract_for_ontobpr(case_id, run_dir / "Payload documents" / case_id, rag_dir, doc_stubs)
            (run_dir / "Payload triples" / "OntoBPR.ttl").write_text(ttl_text, encoding="utf-8")
        except Exception as e:
            print("(OntoBPR extraction skipped:", e, ")")

        # 3) write files + zip
        icdd_path = write_files_and_zip(case_id, run_dir, g_index, g_link)
        print("✅ ICDD written:", icdd_path)

        # 4) structural + coherence checks
        ok, errs = structural_check(run_dir, case_id)
        if not ok:
            print("=== Structural check: FAIL ===")
            print(" -", ok)
            print(" -", errs)
        else:
            print("=== Structural check: OK ===")

        ok2, errs2 = coherence_check(run_dir, case_id)
        if not ok2:
            print("=== Coherence check: FAIL ===")
            for e in errs2:
                print(" -", e)
        else:
            print("=== Coherence check: OK ===")

        # 5) SHACL
        try:
            conf, report = run_shacl_validation(run_dir)
            print("\n=== SHACL RESULT ===")
            print("Conforms:", conf)
            print(report)
        except Exception as e:
            print("(SHACL validation skipped:", e, ")")

if __name__ == "__main__":
    main()
