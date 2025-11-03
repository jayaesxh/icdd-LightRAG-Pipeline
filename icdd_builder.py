from __future__ import annotations
import os, shutil, zipfile, mimetypes
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, XSD

# ISO namespaces
CT  = Namespace("https://standards.iso.org/iso/21597/-1/ed-1/en/Container#")
LS  = Namespace("https://standards.iso.org/iso/21597/-1/ed-1/en/Linkset#")
ELS = Namespace("https://standards.iso.org/iso/21597/-2/ed-1/en/ExtendedLinkset#")

def ex_ns(case_id: str) -> Namespace:
    return Namespace(f"https://example.org/{case_id}/")

def guess_filetype_and_format(filename: str) -> tuple[str, str]:
    name = filename.lower()
    if name.endswith(".pdf"):  return ("pdf", "application/pdf")
    if name.endswith(".json"): return ("json", "application/json")
    if name.endswith(".txt"):  return ("txt", "text/plain")
    if name.endswith(".ttl"):  return ("ttl", "text/turtle")
    if name.endswith(".rdf") or name.endswith(".owl"):
        return ("rdf", "application/rdf+xml")
    if name.endswith(".xml"):  return ("xml", "application/xml")
    if name.endswith(".csv"):  return ("csv", "text/csv")
    if name.endswith(".ifc"):  return ("ifc", "application/octet-stream")
    if name.endswith(".dwg"):  return ("dwg", "application/octet-stream")
    mt = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    ext = (Path(filename).suffix or ".bin").lstrip(".")
    return (ext, mt)

@dataclass
class DocDef:
    iri: URIRef
    rel_path: str     # e.g. "BOCHUM-2025-0017/BuildingApplication.pdf"
    filetype: str     # e.g. "pdf"
    format: str       # e.g. "application/pdf"
    name: str         # human readable name for ct:name

def discover_and_stage_documents(case_id: str, incoming: Path, run_dir: Path) -> List[DocDef]:
    """
    Copies all files from 'incoming' into:
        run_dir / "Payload documents" / case_id / <same relative paths>
    Returns a DocDef for each copied file.
    """
    dest_root = run_dir / "Payload documents" / case_id
    dest_root.mkdir(parents=True, exist_ok=True)

    docs: List[DocDef] = []
    ex = ex_ns(case_id)

    for p in incoming.rglob("*"):
        if not p.is_file():
            continue

        rel_under_incoming = p.relative_to(incoming).as_posix()
        dest = dest_root / rel_under_incoming
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)

        filetype, fmt = guess_filetype_and_format(rel_under_incoming)
        # stable IRI: Doc_<relative path with / replaced by _>
        doc_id = rel_under_incoming.replace("/", "_")
        iri = ex[f"Doc_{doc_id}"]
        docs.append(DocDef(
            iri=iri,
            rel_path=f"{case_id}/{rel_under_incoming}",
            filetype=filetype,
            format=fmt,
            name=Path(rel_under_incoming).name
        ))
    return docs

def build_index_graph(case_id: str,
                      docs: List[DocDef],
                      linkset_filename: str,
                      publisher_text: str = "icdd-rag-pipeline") -> Graph:

    g = Graph()
    g.bind("ct", CT)
    g.bind("ls", LS)
    g.bind("els", ELS)
    g.bind("xsd", XSD)
    ex = ex_ns(case_id)
    g.bind("ex", ex)

    container = ex[f"Container_{case_id}"]

    # container description
    g.add((container, RDF.type, CT.ContainerDescription))
    g.add((container, CT.conformanceIndicator,
           Literal("ICDD-Part1-Container", datatype=XSD.string)))
    g.add((container, CT.publisher,
           Literal(publisher_text, datatype=XSD.string)))

    # documents
    for d in docs:
        g.add((d.iri, RDF.type, CT.Document))
        g.add((d.iri, RDF.type, CT.InternalDocument))
        g.add((d.iri, CT.belongsToContainer, container))
        g.add((d.iri, CT.filename, Literal(d.rel_path, datatype=XSD.string)))
        g.add((d.iri, CT.filetype, Literal(d.filetype, datatype=XSD.string)))
        g.add((d.iri, CT["format"], Literal(d.format, datatype=XSD.string)))
        g.add((d.iri, CT.name, Literal(d.name, datatype=XSD.string)))
        g.add((container, CT.containsDocument, d.iri))

    # register linkset (minimal)
    linkset = ex[f"Linkset_{case_id}"]
    g.add((linkset, RDF.type, CT.Linkset))
    g.add((linkset, CT.filename,
           Literal(linkset_filename, datatype=XSD.string)))
    g.add((container, CT.containsLinkset, linkset))

    return g

@dataclass
class LinkSpec:
    link_type: URIRef        # e.g. ELS.IsControlledBy
    from_doc: URIRef
    from_ident: str
    to_doc: URIRef
    to_ident: str
    link_id: Optional[str] = None

def build_linkset_graph(case_id: str, links: List[LinkSpec]) -> Graph:
    g = Graph()
    g.bind("ls", LS)
    g.bind("els", ELS)
    g.bind("xsd", XSD)

    ex = ex_ns(case_id)
    g.bind("ex", ex)

    for i, L in enumerate(links, start=1):
        link_iri = ex[f"Link_{L.link_id or i}_{case_id}"]
        g.add((link_iri, RDF.type, L.link_type))

        from_elem = URIRef(f"{link_iri}#from")
        to_elem   = URIRef(f"{link_iri}#to")
        id_from   = URIRef(f"{link_iri}#id_from")
        id_to     = URIRef(f"{link_iri}#id_to")

        g.add((id_from, RDF.type, LS.StringBasedIdentifier))
        g.add((id_from, LS.identifier,
               Literal(L.from_ident, datatype=XSD.string)))

        g.add((id_to, RDF.type, LS.StringBasedIdentifier))
        g.add((id_to, LS.identifier,
               Literal(L.to_ident, datatype=XSD.string)))

        for elem, doc_iri, id_iri in (
            (from_elem, L.from_doc, id_from),
            (to_elem,   L.to_doc,   id_to),
        ):
            g.add((elem, RDF.type, LS.LinkElement))
            g.add((elem, LS.document, doc_iri))
            g.add((elem, LS.hasIdentifier, id_iri))

        g.add((link_iri, LS.hasFromLinkElement, from_elem))
        g.add((link_iri, LS.hasToLinkElement,   to_elem))

    return g

def write_icdd_files(case_id: str,
                     run_dir: Path,
                     g_index: Graph,
                     g_link: Graph,
                     linkset_filename: str = "Doc_Application_Links.rdf",
                     copy_ontologies_from: Optional[Path] = None) -> Path:

    p_docs = run_dir / "Payload documents"
    p_trps = run_dir / "Payload triples"
    p_onts = run_dir / "Ontology resources"
    p_docs.mkdir(parents=True, exist_ok=True)
    p_trps.mkdir(parents=True, exist_ok=True)
    p_onts.mkdir(parents=True, exist_ok=True)

    # Write core graphs
    (run_dir / "index.rdf").write_text(g_index.serialize(format="xml"))
    (p_trps / linkset_filename).write_text(g_link.serialize(format="xml"))

    # Human-readable TTL for debugging only
    (p_docs / "BuildingApplicationIndex.ttl").write_text(
        g_index.serialize(format="turtle"))

    # Copy just the ISO ontologies
    if copy_ontologies_from and copy_ontologies_from.exists():
        for name in ("Container.rdf", "Linkset.rdf", "ExtendedLinkset.rdf"):
            src = copy_ontologies_from / name
            if src.exists():
                shutil.copy2(src, p_onts / name)

    # Zip to output/ICDD_<CASE_ID>.icdd
    zip_path = run_dir.parent / f"ICDD_{case_id}.icdd"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in run_dir.rglob("*"):
            z.write(p, p.relative_to(run_dir))
    return zip_path

def quick_structural_check(zip_path: Path, run_dir: Path) -> None:
    problems = []
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        print("ZIP entries:")
        for n in sorted(names):
            print(" ", n)
        if "index.rdf" not in names:
            problems.append("Missing lowercase index.rdf at root.")
        for folder in ("Ontology resources/", "Payload documents/", "Payload triples/"):
            if folder not in names:
                problems.append(f"Missing folder in ZIP: {folder}")

    idx = run_dir / "index.rdf"
    lnk = run_dir / "Payload triples" / "Doc_Application_Links.rdf"
    if not idx.exists(): problems.append("index.rdf missing on disk")
    if not lnk.exists(): problems.append("Doc_Application_Links.rdf missing on disk")

    if problems:
        print("\n❌ Problems:")
        for p in problems:
            print(" -", p)
    else:
        print("\n✅ Structural checks passed.")
