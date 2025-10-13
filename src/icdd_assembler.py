# /workspace/icdd-rag-pipeline/src/icdd_assembler.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Dict, Any, List, Optional
import zipfile, shutil
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD

CT  = Namespace("https://standards.iso.org/iso/21597/-1/ed-1/en/Container#")
LS  = Namespace("https://standards.iso.org/iso/21597/-1/ed-1/en/Linkset#")
ELS = Namespace("https://standards.iso.org/iso/21597/-2/ed-1/en/ExtendedLinkset#")

def write_index_rdf(
    out_root: Path,
    container_iri: str,
    documents: Iterable[Dict[str, Any]],
    description: Optional[str] = None,
    publisher_iri: Optional[str] = None,
) -> Path:
    g = Graph()
    g.bind("ct", CT); g.bind("ls", LS); g.bind("els", ELS)
    g.bind("owl", OWL); g.bind("xsd", XSD); g.bind("rdfs", RDFS)

    C = URIRef(container_iri)
    g.add((C, RDF.type, CT.ContainerDescription))
    g.add((C, CT.conformanceIndicator, Literal("ICDD-Part1-Container", datatype=XSD.string)))
    g.add((C, OWL.imports, URIRef("https://standards.iso.org/iso/21597/-1/ed-1/en/Container.rdf")))
    if publisher_iri:
        g.add((C, CT.publishedBy, URIRef(publisher_iri)))
    if description:
        g.add((C, CT.description, Literal(description, datatype=XSD.string)))

    for d in documents:
        D = URIRef(d["iri"])
        g.add((D, RDF.type, CT.InternalDocument))
        g.add((C, CT.containsDocument, D))
        # ct:filename must be relative to "Payload documents/"
        g.add((D, CT.filename, Literal(d["filename"], datatype=XSD.string)))
        if ft := d.get("filetype"):
            g.add((D, CT.filetype, Literal(ft, datatype=XSD.string)))
        if fm := d.get("format"):
            g.add((D, CT.format, Literal(fm, datatype=XSD.string)))
        if nm := d.get("name"):
            g.add((D, CT.name, Literal(nm, datatype=XSD.string)))

    out = out_root / "Index.rdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(out), format="xml")
    return out

def write_linkset_rdf(
    out_root: Path,
    linkset_relpath: str,
    container_iri: str,
    links: Iterable[Dict[str, Any]],
) -> Path:
    g = Graph()
    g.bind("ct", CT); g.bind("ls", LS); g.bind("els", ELS)
    g.bind("owl", OWL); g.bind("xsd", XSD)

    # import Index + Linkset ontology
    g.add((URIRef(container_iri), OWL.imports, URIRef("Index.rdf")))
    g.add((URIRef(container_iri), OWL.imports, URIRef("https://standards.iso.org/iso/21597/-1/ed-1/en/Linkset.rdf")))

    for L in links:
        link_iri = URIRef(L["id"])
        ltype = L.get("type", "ls:Link")
        if ltype.startswith("els:"):
            g.add((link_iri, RDF.type, ELS[ltype.split(":",1)[1]]))
        else:
            g.add((link_iri, RDF.type, LS.Link))

        fr = L["from"]; to = L["to"]
        le_from = URIRef(f"{L['id']}#from")
        le_to   = URIRef(f"{L['id']}#to")
        g.add((le_from, RDF.type, LS.LinkElement))
        g.add((le_to,   RDF.type, LS.LinkElement))
        g.add((link_iri, LS.hasFromLinkElement, le_from))
        g.add((link_iri, LS.hasToLinkElement,   le_to))

        g.add((le_from, LS.document, URIRef(fr["doc_iri"])))
        g.add((le_to,   LS.document, URIRef(to["doc_iri"])))

        def add_identifier(le_node, ident: Dict[str, str]):
            kind = ident.get("kind")
            val  = ident.get("value","")
            idn  = URIRef(f"{le_node}#id")
            if kind == "uri":
                g.add((idn, RDF.type, LS.URIBasedIdentifier))
                g.add((idn, LS.uri, Literal(val, datatype=XSD.anyURI)))
            elif kind == "query":
                g.add((idn, RDF.type, LS.QueryBasedIdentifier))
                g.add((idn, LS.queryExpression, Literal(val, datatype=XSD.string)))
                g.add((idn, LS.queryLanguage,   Literal(ident.get("language","SQL"), datatype=XSD.string)))
            else:
                g.add((idn, RDF.type, LS.StringBasedIdentifier))
                g.add((idn, LS.identifier, Literal(val, datatype=XSD.string)))
            g.add((le_node, LS.hasIdentifier, idn))

        add_identifier(le_from, fr["identifier"])
        add_identifier(le_to,   to["identifier"])

    out = Path(out_root, "Payload triples", linkset_relpath)
    out.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(out), format="xml")
    return out

def copy_into_container(
    out_root: Path,
    ontology_dir: Path,
    payload_docs_dir: Path
) -> None:
    (out_root / "Ontology resources").mkdir(parents=True, exist_ok=True)
    (out_root / "Payload documents").mkdir(parents=True, exist_ok=True)
    (out_root / "Payload triples").mkdir(parents=True, exist_ok=True)

    # Ontologies (optional but recommended)
    for name in ["Container.rdf","Linkset.rdf","ExtendedLinkset.rdf"]:
        src = ontology_dir / name
        if src.exists():
            dst = out_root / "Ontology resources" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    # Payload docs
    if payload_docs_dir.exists():
        for p in payload_docs_dir.rglob("*"):
            if p.is_file():
                dst = out_root / "Payload documents" / p.relative_to(payload_docs_dir)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)

def zip_as_icdd(out_root: Path, icdd_path: Path) -> Path:
    with zipfile.ZipFile(icdd_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for fp in out_root.rglob("*"):
            if fp.is_file():
                z.write(fp, arcname=str(fp.relative_to(out_root)))
    return icdd_path
