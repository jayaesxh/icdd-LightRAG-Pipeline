# ontobpr_llm.py
from __future__ import annotations
import json, asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from llm_backend import call_llm

ONTOBPR = Namespace("https://w3id.org/ontobpr#")
EX      = "https://example.org/{case}#"

SYSTEM = """You extract structured building-permit facts for OntoBPR.
Return ONLY compact JSON with keys:
{
  "application_reference": "string or null",
  "application_type": "string or null",
  "submission_date": "YYYY-MM-DD or null",
  "site_address": {"street": "...", "city": "...", "postcode": "...", "country": "..."},
  "applicant": {"name": "...", "organization": "..."},
  "building": {"label": "...", "type": "...", "storeys": "int or null"},
  "supporting_doc_chunk_ids": ["chunk-id-1", "chunk-id-2", "chunk-id-3"]
}
Rules:
- Only fill values that are explicitly evidenced in the provided context.
- Prefer exact values as printed (reference IDs, address, dates).
- If uncertain, set null or omit the subfield.
"""

def _case_iri(case_id: str, tag: str) -> str:
    return EX.format(case=case_id) + tag

async def _rag_extract(run_dir: Path, case_id: str, model_id: str) -> Dict[str, Any]:
    """
    Ask LightRAG for focused evidence, then one LLM pass to emit normalized JSON.
    """
    store = run_dir / "rag" / "lightrag_store"
    rag = LightRAG(working_dir=str(store))
    await rag.initialize_storages()

    # 1) Pull top context chunks
    queries = {
      "application_reference": "Provide the application reference number/id exactly as written.",
      "site_address": "Provide the site address (street, city, postcode, country if present).",
      "applicant": "Who is the applicant (person/organization)?",
      "application_type": "What is the application type (e.g., Full, Outline, etc.)?",
      "submission_date": "What is the submission/received date (YYYY-MM-DD if possible)?",
      "building": "What is the building label/type and storeys (if present)?"
    }

    param = QueryParam(top_k=10, mode="hybrid")
    gathered = {}
    for k, q in queries.items():
        ans = await rag.query(q, param=param)
        gathered[k] = ans

    # 2) Compose a compact context block for the LLM (raw LightRAG returns + IDs)
    # Some LightRAG versions return strings; others return dicts. Keep it simple:
    context_blob = json.dumps(gathered)[:100000]

    # 3) Ask local model for normalized JSON (function-free, pure text JSON)
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"CASE_ID={case_id}\nContext:\n{context_blob}\nReturn ONLY JSON."}
    ]
    out = call_llm(msgs, model_id_or_path=model_id, max_new_tokens=768, temperature=0.0)
    # be defensive
    try:
        j = json.loads(out[out.find("{"): out.rfind("}")+1])
    except Exception:
        # fallback: very strict “empty”
        j = {
          "application_reference": None, "application_type": None, "submission_date": None,
          "site_address": {"street": None, "city": None, "postcode": None, "country": None},
          "applicant": {"name": None, "organization": None},
          "building": {"label": None, "type": None, "storeys": None},
          "supporting_doc_chunk_ids": []
        }
    return j

def _add_if(g: Graph, s: URIRef, p: URIRef, val: Optional[str], dtype=None):
    if val is None or str(val).strip() == "":
        return
    if dtype == "int":
        g.add((s, p, Literal(int(val), datatype=XSD.integer)))
    elif dtype == "date":
        g.add((s, p, Literal(val, datatype=XSD.date)))
    else:
        g.add((s, p, Literal(str(val))))

def build_ontobpr_from_json(run_dir: Path, case_id: str, j: Dict[str, Any]) -> Graph:
    """
    Map normalized JSON → OntoBPR triples.
    """
    g = Graph()
    g.bind("ontobpr", ONTOBPR)

    # IRIs
    app = URIRef(_case_iri(case_id, f"BA_{case_id}"))
    cont = URIRef(_case_iri(case_id, f"Container_{case_id}"))
    bld = URIRef(_case_iri(case_id, f"Building_{case_id}"))

    g.add((app, RDF.type, ONTOBPR.BuildingApplication))
    g.add((bld, RDF.type, ONTOBPR.Building))
    # link app → container
    g.add((app, ONTOBPR.hasBuildingApplicationContainer, cont))

    # Application-level
    _add_if(g, app, ONTOBPR.hasApplicationReference, j.get("application_reference"))
    _add_if(g, app, ONTOBPR.hasApplicationType, j.get("application_type"))
    _add_if(g, app, ONTOBPR.hasSubmissionDate, j.get("submission_date"), dtype="date")

    # Site address (flatten to literals; you can later model a dedicated Address node if OntoBPR has one)
    site = j.get("site_address") or {}
    for k in ["street", "city", "postcode", "country"]:
        _add_if(g, app, URIRef(ONTOBPR + f"site_{k.capitalize()}"), site.get(k))

    # Applicant
    appl = j.get("applicant") or {}
    if appl.get("name") or appl.get("organization"):
        actor = URIRef(_case_iri(case_id, f"Applicant_{case_id}"))
        g.add((actor, RDF.type, ONTOBPR.Actor))
        _add_if(g, actor, ONTOBPR.actorName, appl.get("name"))
        _add_if(g, actor, ONTOBPR.actorOrganization, appl.get("organization"))
        g.add((app, ONTOBPR.hasApplicant, actor))

    # Building
    binfo = j.get("building") or {}
    _add_if(g, bld, ONTOBPR.buildingLabel, binfo.get("label"))
    _add_if(g, bld, ONTOBPR.buildingType, binfo.get("type"))
    _add_if(g, bld, ONTOBPR.storeys, binfo.get("storeys"), dtype="int")

    # (Optional) link supporting docs by IRI if you later map chunk-ids → doc IRIs
    # left as TODO; depends on your chunking metadata

    return g

def run_ontobpr_extraction(run_dir: str, case_id: str, model_id: str) -> Path:
    """
    Full pipeline: RAG → JSON → TTL file in Payload triples/OntoBPR.ttl
    """
    run_dir = Path(run_dir)
    j = asyncio.run(_rag_extract(run_dir, case_id, model_id))
    g = build_ontobpr_from_json(run_dir, case_id, j)
    out = run_dir / "Payload triples" / "OntoBPR.ttl"
    out.write_text(g.serialize(format="turtle"), encoding="utf-8")
    print(f"[OntoBPR-LLM] wrote → {out}")
    return out
