# icdd_builder.py
from __future__ import annotations

import os, io, zipfile, shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Iterable

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, XSD

# --- ISO 21597 namespaces (ed-1) ---
CT  = Namespace("https://standards.iso.org/iso/21597/-1/ed-1/en/Container#")
LS  = Namespace("https://standards.iso.org/iso/21597/-1/ed-1/en/Linkset#")
ELS = Namespace("https://standards.iso.org/iso/21597/-2/ed-1/en/ExtendedLinkset#")

# ---------- simple specs ----------
@dataclass
class DocSpec:
    iri: URIRef
    rel_filename: str  # relative to "Payload documents/<CASE_ID>/"
    filetype: str      # e.g. 'pdf', 'ifc'
    format: str        # MIME, e.g. 'application/pdf'
    name: str          # label
    description: str   # free text

@dataclass
class LinkSpec:
    link_type: URIRef     # e.g., ELS.IsControlledBy
    from_doc: URIRef      # a ct:Document IRI (from index)
    from_ident: str       # "whole-doc" or any string identifier
    to_doc: URIRef
    to_ident: str

# ---------- helpers ----------
def _base(case_id: str) -> str:
    # supervisor asked to use fragment IRIs (#)
    return f"https://example.org/{case_id}#"

def copy_ontology_resources(src_dir: Path, dst_dir: Path, also_write_fixed_shapes: bool = False) -> None:
    """
    Copy everything from static_resources/ontology_resources into run_dir/"Ontology resources".
    This makes the .icdd consumable by external platforms (RUB converter, etc.).
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src_dir.exists():
        for p in src_dir.iterdir():
            if p.is_file():
                shutil.copy2(p, dst_dir / p.name)
    # Do not rewrite shapes unless explicitly requested. Default keeps the official ones.
    if also_write_fixed_shapes:
        # (Optional patch hook; not used by default)
        pass

# ---------- graph builders ----------
def build_index_graph(
    case_id: str,
    docs: Iterable[DocSpec],
    linkset_filename: str = "Doc_Application_Links.rdf",
) -> Graph:
    """
    Build a minimal but ISO-21597-compliant index graph:
      - ct:ContainerDescription node
      - ct:containsDocument per payload doc
      - ct:Linkset node registered via ct:containsLinkset with ct:filename
      - ct:publisher as a ct:Party with ct:name (to satisfy common SHACL)
    """
    base = _base(case_id)
    g = Graph()
    g.bind("ct", CT)
    g.bind("ls", LS)
    g.bind("els", ELS)

    container = URIRef(base + f"Container_{case_id}")
    linkset   = URIRef(base + f"Linkset_{case_id}")
    publisher = URIRef(base + f"Publisher_{case_id}")

    # ContainerDescription
    g.add((container, RDF.type, CT.ContainerDescription))
    g.add((container, CT.conformanceIndicator, Literal("ICDD-Part1-Container", datatype=XSD.string)))

    # Publisher as ct:Party with ct:name (matches typical Container.shapes.ttl)
    g.add((publisher, RDF.type, CT.Party))
    g.add((publisher, CT.name, Literal("icdd-rag-pipeline", datatype=XSD.string)))
    g.add((container, CT.publisher, publisher))

    # Documents
    for d in docs:
        g.add((d.iri, RDF.type, CT.Document))
        g.add((d.iri, RDF.type, CT.InternalDocument))
        g.add((d.iri, CT.belongsToContainer, container))
        g.add((d.iri, CT.filename, Literal(Path(d.rel_filename).name, datatype=XSD.string)))
        g.add((d.iri, CT.filetype, Literal(d.filetype, datatype=XSD.string)))
        g.add((d.iri, CT["format"], Literal(d.format, datatype=XSD.string)))
        g.add((d.iri, CT.name, Literal(d.name, datatype=XSD.string)))
        g.add((container, CT.containsDocument, d.iri))

    # Linkset registration
    g.add((linkset, RDF.type, CT.Linkset))
    g.add((linkset, CT.filename, Literal(linkset_filename, datatype=XSD.string)))
    g.add((container, CT.containsLinkset, linkset))

    return g

def build_payload_linkset_graph(case_id: str, links: Iterable[LinkSpec]) -> Graph:
    """
    Build a payload linkset graph written under 'Payload triples/<filename>'.
    """
    base = _base(case_id)
    g = Graph()
    g.bind("ls", LS)
    g.bind("els", ELS)

    for i, Lk in enumerate(links, start=1):
        link_iri = URIRef(base + f"Link_{i}_{case_id}")
        from_elem = URIRef(str(link_iri) + "#from")
        to_elem   = URIRef(str(link_iri) + "#to")
        id_from   = URIRef(str(link_iri) + "#id_from")
        id_to     = URIRef(str(link_iri) + "#id_to")

        g.add((link_iri, RDF.type, Lk.link_type))

        for elem, doc_iri, id_iri in ((from_elem, Lk.from_doc, id_from),
                                      (to_elem,   Lk.to_doc,   id_to)):
            g.add((elem, RDF.type, LS.LinkElement))
            g.add((elem, LS.document, doc_iri))
            g.add((elem, LS.hasIdentifier, id_iri))

        g.add((id_from, RDF.type, LS.StringBasedIdentifier))
        g.add((id_from, LS.identifier, Literal(Lk.from_ident, datatype=XSD.string)))

        g.add((id_to, RDF.type, LS.StringBasedIdentifier))
        g.add((id_to, LS.identifier, Literal(Lk.to_ident, datatype=XSD.string)))

        g.add((link_iri, LS.hasFromLinkElement, from_elem))
        g.add((link_iri, LS.hasToLinkElement,   to_elem))

    return g

# ---------- writers & checks ----------
def write_icdd_files(
    case_id: str,
    g_index: Graph,
    g_link: Graph,
    run_dir: Path,
    out_root: Path,
    linkset_filename: str = "Doc_Application_Links.rdf",
) -> Path:
    """
    Writes:
      index.rdf
      Payload triples/<linkset_filename>
      (zip) output/ICDD_<CASE_ID>.icdd with <CASE_ID>/[...] inside
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "Payload triples").mkdir(parents=True, exist_ok=True)

    # Write graphs (use destination=... so rdflib handles encoding)
    g_index.serialize(destination=str(run_dir / "index.rdf"), format="xml")
    g_link.serialize(destination=str(run_dir / "Payload triples" / linkset_filename), format="xml")

    # Make .icdd with root folder named <CASE_ID>/
    out_root.mkdir(parents=True, exist_ok=True)
    icdd_path = out_root / f"ICDD_{case_id}.icdd"
    with zipfile.ZipFile(icdd_path, "w", zipfile.ZIP_DEFLATED) as zf:
        def _add_dir(d: Path, arc_prefix: str):
            for p in d.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=str(Path(case_id) / arc_prefix / p.relative_to(d)))
        # Ontology resources, Payload documents, Payload triples, index.rdf
        _add_dir(run_dir / "Ontology resources", "Ontology resources")
        _add_dir(run_dir / "Payload documents", "Payload documents")
        _add_dir(run_dir / "Payload triples",   "Payload triples")
        zf.write(run_dir / "index.rdf", arcname=str(Path(case_id) / "index.rdf"))

    return icdd_path

def quick_structural_check(run_dir: Path) -> List[str]:
    """
    Basic ZIP/layout sanity (no SHACL here).
    """
    issues = []
    must = [
        run_dir / "Ontology resources",
        run_dir / "Payload documents",
        run_dir / "Payload triples",
        run_dir / "index.rdf",
    ]
    for p in must:
        if not p.exists():
            issues.append(f"Missing required path: {p}")
    # must have linkset file under Payload triples
    if not list((run_dir / "Payload triples").glob("*.rdf")):
        issues.append("No RDF linkset found under 'Payload triples'.")
    return issues

def coherence_check(run_dir: Path, case_id: str) -> List[str]:
    """
    Ensures that index.rdf registers the Linkset correctly and that at least one
    ct:containsDocument exists.
    """
    issues = []
    g = Graph()
    idx = run_dir / "index.rdf"
    if not idx.exists():
        return ["index.rdf is missing"]
    g.parse(str(idx))
    container = URIRef(_base(case_id) + f"Container_{case_id}")
    linkset   = URIRef(_base(case_id) + f"Linkset_{case_id}")

    if not list(g.triples((container, CT.containsLinkset, linkset))):
        issues.append("Container does not ct:containsLinkset the Linkset node.")
    if not list(g.triples((linkset, RDF.type, CT.Linkset))):
        issues.append("Linkset node is not typed ct:Linkset.")
    if not list(g.triples((container, CT.containsDocument, None))):
        issues.append("Container has no ct:containsDocument triples.")
    return issues

def run_shacl_validation(run_dir: Path, shapes_dirs: Optional[List[Path]] = None):
    """
    Convenience wrapper around pySHACL.
    """
    try:
        from pyshacl import validate
    except Exception:
        return False, "pySHACL not installed", ""

    data = Graph()
    data.parse(str(run_dir / "index.rdf"))
    for p in (run_dir / "Payload triples").glob("*.rdf"):
        data.parse(str(p))

    # collect shapes
    shapes = Graph()
    candidates = shapes_dirs or [
        run_dir / "Ontology resources",
        run_dir / "Ontology resources" / "shapes",
        Path("/workspace/icdd-rag-pipeline/static_resources/ontology_resources"),
    ]
    def _fmt(pp: Path):
        s = pp.suffix.lower()
        if s == ".ttl": return "turtle"
        if s in (".rdf", ".owl", ".xml"): return "xml"
        return None

    for base in candidates:
        if not base.exists(): continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".ttl", ".rdf", ".owl", ".xml"):
                try:
                    shapes.parse(str(p), format=_fmt(p))
                except Exception:
                    pass

    conforms, rep_g, rep_t = validate(
        data,
        shacl_graph=shapes,
        inference="rdfs",
        advanced=True,
        debug=False
    )
    return bool(conforms), rep_t, rep_g
