# /workspace/icdd-rag-pipeline/ontobpr_extractor.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF

ONTOBPR = Namespace("https://w3id.org/ontobpr#")
CT      = Namespace("https://standards.iso.org/iso/21597/-1/ed-1/en/Container#")

@dataclass
class DocStub:
    iri: URIRef
    rel_filename: str
    name: Optional[str] = None

def _hash_base(case_id: str) -> str:
    # Use hash-delimited base to satisfy platform import preference
    return f"https://example.org/{case_id}#"

def index_and_extract_for_ontobpr(
    case_id: str,
    run_dir: Path,
    doc_index: Iterable[DocStub],
) -> Path:
    """
    Minimal, safe OntoBPR attachment:
      - ontobpr:BuildingApplication BA_<CASE>
          ontobpr:hasBuildingApplicationContainer -> ct:ContainerDescription node
      - ontobpr:Building Building_<CASE>
          ontobpr:hasBuildingDocument -> chosen application doc
    Produces: Payload triples/OntoBPR.ttl
    """
    base = _hash_base(case_id)
    g = Graph()
    g.bind("ontobpr", ONTOBPR)
    g.bind("ct", CT)

    container = URIRef(f"{base}Container_{case_id}")
    ba        = URIRef(f"{base}BA_{case_id}")
    bldg      = URIRef(f"{base}Building_{case_id}")

    # Pick the application form if present, else first doc
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
    return out
