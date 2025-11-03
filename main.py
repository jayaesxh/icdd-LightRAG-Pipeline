# main.py — Transformers-only (no Ollama, no APOC)
import os, re, json, zipfile, shutil
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, DCTERMS, XSD, OWL

from src.retriever import Neo4jRetriever
from src.lightrag_client import dual_level_retrieve  # optional, safe if index missing

# Keep transformers logs quiet
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# ---- Env & output ----
if not load_dotenv():
    load_dotenv("/workspace/.env")

OUT_ROOT = Path("/workspace/icdd-rag-pipeline/output")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ---- Schema ----
class Triple(BaseModel):
    subject: str
    predicate: str
    object: str
    is_literal: bool = False
    datatype: Optional[str] = None

class ICDDOutput(BaseModel):
    index_triples: List[Triple]
    linkset_triples: List[Triple]
    provenance: dict

# ---- Prompt ----
SYSTEM = """You are an expert in OntoBPR and ISO 21597 (ICDD).
Return ONLY one valid JSON object with keys: index_triples, linkset_triples, provenance.
Each triple: {subject, predicate, object, is_literal, datatype?}. No markdown, no prose."""

USER_TPL = """You get two kinds of context.

[KG_CONTEXT]
---
{kg_context}
---

[TEXT_CONTEXT]
---
{text_context}
---

Use EXACTLY these document identifiers and filenames in your triples:
- ex:Doc_Application_{app_id} -> ct:filename "{app_id}/Application.json" (xsd:string), ct:filetype "application/json"
- ex:Doc_Regulations_{app_id} -> ct:filename "{app_id}/Regulations.txt" (xsd:string), ct:filetype "text/plain"
- ex:Doc_Index_{app_id}       -> ct:filename "BuildingApplicationIndex.ttl" (xsd:string), ct:filetype "text/turtle"

Rules:
- INDEX (index.rdf content): Create ct:ContainerDescription ex:Container_{app_id}
  * Add ct:InternalDocument for the three docs above with EXACT filenames shown.
  * Link ex:Container_{app_id} -> each doc via ct:containsDocument
  * Use rdf:type for classes
- LINKS (Doc_Application_Links.rdf):
  * For each obpr:mustComplyWith edge from application to a regulation sentence,
    create an ls:Link (type els:IsControlledBy) connecting two ls:LinkElement:
    - From: ex:Doc_Application_{app_id} with a StringBasedIdentifier (e.g., "whole-doc" or a section id)
    - To:   ex:Doc_Regulations_{app_id} with a StringBasedIdentifier (e.g., a line/anchor id)
  * Keep identifiers as short strings.

Application ID: {app_id}
Return STRICT JSON only: {{index_triples, linkset_triples, provenance}} with triples shaped
{{subject, predicate, object, is_literal, datatype?}}. No prose."""

# ---- Namespaces & helpers ----
def ns_map(app_id: str) -> dict:
    return {
        "ct":  Namespace("https://standards.iso.org/iso/21597/-1/ed-1/en/Container#"),
        "ls":  Namespace("https://standards.iso.org/iso/21597/-1/ed-1/en/Linkset#"),
        "els": Namespace("https://standards.iso.org/iso/21597/-2/ed-1/en/ExtendedLinkset#"),
        "obpr":Namespace("https://w3id.org/ontobpr#"),
        "ex":  Namespace(f"https://example.org/{app_id}/"),
        "rdf": RDF, "rdfs": RDFS, "dcterms": DCTERMS, "xsd": XSD, "owl": OWL
    }

def sanitize_curie(term: Optional[str]) -> Optional[str]:
    if not isinstance(term, str):
        return term
    t = term.strip()
    while t.startswith(":"):
        t = t[1:]
    t = re.sub(r"([A-Za-z0-9_]+)::", r"\1:", t)
    return t

def expand(term: Optional[str], ns: dict) -> URIRef:
    if not term:
        return URIRef("")
    term = sanitize_curie(term)
    if term.startswith(("http://", "https://")):
        return URIRef(term)
    if ":" in term:
        pfx, local = term.split(":", 1)
        if pfx in ns:
            return ns[pfx][local]
    return ns["ex"][term]

def _sanitize_ct_filename(value: str) -> str:
    v = (value or "").replace("\\", "/")
    return v[len("Payload documents/"):] if v.startswith("Payload documents/") else v

def normalize_subjects_to_instances(parsed: ICDDOutput, app_id: str) -> ICDDOutput:
    inst = f"ex:Container_{app_id}"
    fixed_index = []
    for t in parsed.index_triples:
        s = t.subject
        if s in ("ct:ContainerDescription", "ct:Container"):
            t = t.model_copy(update={"subject": inst})
        if t.predicate == "ct:filename" and t.is_literal:
            t = t.model_copy(update={"object": _sanitize_ct_filename(t.object)})
        fixed_index.append(t)
    return ICDDOutput(index_triples=fixed_index,
                      linkset_triples=parsed.linkset_triples,
                      provenance=parsed.provenance)

def create_minimal_docs(app_id: str, run_dir: Path, kg_context: str) -> dict:
    """
    Create minimal internal documents so ct:filename points to real files.
    Returns a dict with relative payload paths used later in Index.rdf.
    """
    docs_dir = run_dir / "Payload documents" / app_id
    docs_dir.mkdir(parents=True, exist_ok=True)

    # 1) Application.json
    app_json = docs_dir / "Application.json"
    if not app_json.exists():
        app_json.write_text(json.dumps({"applicationId": app_id}, indent=2), encoding="utf-8")

    # 2) Regulations.txt (seeded with KG context for now)
    regs_txt = docs_dir / "Regulations.txt"
    if not regs_txt.exists():
        regs_txt.write_text("Derived regulation context (raw):\n\n" + kg_context[:10000], encoding="utf-8")

    # 3) Index.ttl file is written later; we just return the name
    return {
        "application": f"{app_id}/Application.json",
        "regulations": f"{app_id}/Regulations.txt",
        "index_ttl": "BuildingApplicationIndex.ttl",
    }

def _coerce_triple_row(row):
    """
    Accept dict or list triple and return a dict:
    {subject, predicate, object, is_literal, datatype?}
    """
    if isinstance(row, dict):
        dt = row.get("datatype")
        if isinstance(dt, str) and dt.lower() == "string":
            row["datatype"] = "xsd:string"
        return row

    if isinstance(row, (list, tuple)) and len(row) >= 3:
        s, p, o = row[0], row[1], row[2]
        is_lit = bool(row[3]) if len(row) >= 4 else False
        dt = row[4] if len(row) >= 5 else None
        if isinstance(dt, str) and dt.lower() == "string":
            dt = "xsd:string"
        return {"subject": s, "predicate": p, "object": o, "is_literal": is_lit, "datatype": dt}
    return None

def extract_json_only(txt: str) -> str:
    """Return the substring between the first '{' and last '}' (after stripping code fences)."""
    t = txt.strip().replace("```json", "```")
    if t.startswith("```") and t.endswith("```"):
        t = t[3:-3]
    i, j = t.find("{"), t.rfind("}")
    return t[i:j+1] if (i != -1 and j != -1 and j > i) else t

def repair_and_normalize_json_payload(js_text: str, app_id: str) -> str:
    """
    Make the model's JSON strict and schema-like:
      - strip fences / prose
      - quote bare keys, fix quotes, remove/add commas
      - normalize booleans/null, fix key typos (linkset_triple -> linkset_triples)
      - coerce list-style triples to dicts
      - ensure provenance dict
    If parsing still fails, return a minimal valid JSON payload so the pipeline can continue.
    """
    import re, json

    s = js_text.strip().replace("```json", "```")
    if s.startswith("```") and s.endswith("```"):
        s = s[3:-3]

    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        s = s[i:j+1]

    s = (s.replace("\r", "")
          .replace("“", '"').replace("”", '"').replace("’", "'"))

    # Quote bare keys at start-of-line positions
    s = re.sub(r'(?m)^(\s*)([A-Za-z_][A-Za-z0-9_\-]*)\s*:', r'\1"\2":', s)

    # Single quotes -> double quotes (strings only)
    def _sq_to_dq(m):
        inner = m.group(1)
        return '"' + inner.replace('"', '\\"') + '"'
    s = re.sub(r"'([^'\\]*?)'", _sq_to_dq, s)

    # Remove trailing commas and add commas between adjacent objects
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    s = re.sub(r"}\s*{", r"},{", s)

    # Python-isms to JSON
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    s = s.replace('"linkset_triple"', '"linkset_triples"')

    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        # 🔑 Non-fatal fallback: let the pipeline proceed without LLM output
        return json.dumps(
            {"index_triples": [], "linkset_triples": [], "provenance": {"app_id": app_id, "sources": ["fallback-json"]}},
            ensure_ascii=False
        )

    # ensure keys and coerce
    if "index_triples" not in data or not isinstance(data["index_triples"], list):
        data["index_triples"] = []
    if "linkset_triples" not in data or not isinstance(data["linkset_triples"], list):
        data["linkset_triples"] = []

    def _coerce_triple_row(row):
        if isinstance(row, dict):
            dt = row.get("datatype")
            if isinstance(dt, str) and dt.lower() == "string":
                row["datatype"] = "xsd:string"
            return row
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            s_, p_, o_ = row[0], row[1], row[2]
            is_lit = bool(row[3]) if len(row) >= 4 else False
            dt = row[4] if len(row) >= 5 else None
            if isinstance(dt, str) and dt.lower() == "string":
                dt = "xsd:string"
            return {"subject": s_, "predicate": p_, "object": o_, "is_literal": is_lit, "datatype": dt}
        return None

    idx = []
    for t in data["index_triples"]:
        ct = _coerce_triple_row(t)
        if ct: idx.append(ct)
    lks = []
    for t in data["linkset_triples"]:
        ct = _coerce_triple_row(t)
        if ct: lks.append(ct)

    def sanitize_curie(term: str) -> str:
        if not isinstance(term, str): return term
        t = term.strip()
        while t.startswith(":"): t = t[1:]
        t = re.sub(r"([A-Za-z0-9_]+)::", r"\1:", t)
        return t

    def _clean_row(r):
        r["subject"]   = sanitize_curie(r.get("subject"))
        r["predicate"] = sanitize_curie(r.get("predicate"))
        if not r.get("is_literal", False):
            r["object"] = sanitize_curie(r.get("object"))
        if r.get("is_literal", False) and (not r.get("datatype")):
            r["datatype"] = "xsd:string"
        return r

    data["index_triples"]   = [_clean_row(t) for t in idx]
    data["linkset_triples"] = [_clean_row(t) for t in lks]

    prv = data.get("provenance")
    if not isinstance(prv, dict):
        data["provenance"] = {"app_id": app_id, "sources": [str(prv)] if prv else []}
    else:
        prv.setdefault("app_id", app_id)
        prv.setdefault("sources", [])

    return json.dumps(data, ensure_ascii=False)


def canonicalize_iris(parsed: ICDDOutput, app_id: str) -> ICDDOutput:
    CAN = {
        "container": f"ex:Container_{app_id}",
        "doc_app":   f"ex:Doc_Application_{app_id}",
        "doc_reg":   f"ex:Doc_Regulations_{app_id}",
        "doc_idx":   f"ex:Doc_Index_{app_id}",
    }
    def fix_term(t: str) -> str:
        if not isinstance(t, str): return t
        t = re.sub(r"ex:Container_[^,\s\"}]+", CAN["container"], t)
        t = re.sub(r"ex:Doc_Application_[^,\s\"}]+", CAN["doc_app"], t)
        t = re.sub(r"ex:Doc_Regulations_[^,\s\"}]+", CAN["doc_reg"], t)
        t = re.sub(r"ex:Doc_Index_[^,\s\"}]+", CAN["doc_idx"], t)
        return t

    idx2, lnk2 = [], []
    for tr in parsed.index_triples:
        idx2.append(tr.model_copy(update={"subject": fix_term(tr.subject),
                                          "predicate": tr.predicate,
                                          "object": fix_term(tr.object)}))
    for tr in parsed.linkset_triples:
        lnk2.append(tr.model_copy(update={"subject": fix_term(tr.subject),
                                          "predicate": tr.predicate,
                                          "object": fix_term(tr.object)}))
    return ICDDOutput(index_triples=idx2, linkset_triples=lnk2, provenance=parsed.provenance)

def coerce_linkset_shape(parsed: ICDDOutput, app_id: str) -> ICDDOutput:
    has_link_elements = any(t.predicate in ("ls:hasFromLinkElement", "ls:hasToLinkElement")
                            for t in parsed.linkset_triples)
    if has_link_elements:
        return parsed

    edges = [(t.subject, t.object) for t in parsed.linkset_triples
             if t.predicate in ("ls:controlledBy", "els:IsControlledBy")]
    if not edges:
        edges = [(f"ex:Doc_Application_{app_id}", f"ex:Doc_Regulations_{app_id}")]

    new_links: List[Triple] = []
    for i, (frm, to) in enumerate(edges, 1):
        link = f"ex:Link_{i}_{app_id}"
        le_from = f"{link}#from"
        le_to   = f"{link}#to"
        id_from = f"{link}#id_from"
        id_to   = f"{link}#id_to"

        new_links.append(Triple(subject=link, predicate="rdf:type", object="els:IsControlledBy"))
        new_links.append(Triple(subject=le_from, predicate="rdf:type", object="ls:LinkElement"))
        new_links.append(Triple(subject=le_to,   predicate="rdf:type", object="ls:LinkElement"))
        new_links.append(Triple(subject=link,    predicate="ls:hasFromLinkElement", object=le_from))
        new_links.append(Triple(subject=link,    predicate="ls:hasToLinkElement",   object=le_to))
        new_links.append(Triple(subject=le_from, predicate="ls:document", object=frm))
        new_links.append(Triple(subject=le_to,   predicate="ls:document", object=to))
        new_links.append(Triple(subject=id_from, predicate="rdf:type", object="ls:StringBasedIdentifier"))
        new_links.append(Triple(subject=id_from, predicate="ls:identifier",
                                object="whole-doc", is_literal=True, datatype="xsd:string"))
        new_links.append(Triple(subject=le_from, predicate="ls:hasIdentifier", object=id_from))
        new_links.append(Triple(subject=id_to, predicate="rdf:type", object="ls:StringBasedIdentifier"))
        new_links.append(Triple(subject=id_to, predicate="ls:identifier",
                                object="reg-ctx", is_literal=True, datatype="xsd:string"))
        new_links.append(Triple(subject=le_to, predicate="ls:hasIdentifier", object=id_to))

    return ICDDOutput(index_triples=parsed.index_triples, linkset_triples=new_links, provenance=parsed.provenance)

def _locate_ontology_resources() -> Optional[Path]:
    candidates = [
        Path("/workspace/icdd-rag-pipeline/static_resources/Ontology resources"),
        Path("/workspace/icdd-rag-pipeline/static_resources/ontology_resources"),
        Path("/workspace/icdd-rag-pipeline/static_resources/ontologies"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

# ---- write_outputs: SHACL-compliant (final) ----
def write_outputs(parsed: ICDDOutput, app_id: str, out_root: Path, auto_docs: dict | None = None) -> Path:
    """
    Build index.rdf + Payload triples/Doc_Application_Links.rdf,
    copy Ontology resources, and zip as ICDD_<APP_ID>.icdd.

    SHACL-friendly:
      - index filename is lowercase 'index.rdf'
      - no owl:imports
      - container has ct:conformanceIndicator and ct:publisher (publisher is an IRI, not a literal)
      - linkset: ONLY rdf:type + ct:filename; NO back-reference to the container
      - container -> ct:containsLinkset link registers the linkset
      - each document typed ct:Document and ct:InternalDocument, and has
            ct:filename, ct:filetype, ct:name, ct:belongsToContainer
      - filenames are relative to their folders (no folder prefix in values)
      - no ct:format properties (closed shapes often reject it)
    """
    run_dir = out_root / app_id
    p_docs = run_dir / "Payload documents"
    p_trps = run_dir / "Payload triples"
    p_onts = run_dir / "Ontology resources"
    p_docs.mkdir(parents=True, exist_ok=True)
    p_trps.mkdir(parents=True, exist_ok=True)
    p_onts.mkdir(parents=True, exist_ok=True)

    ns = ns_map(app_id)
    CT = ns["ct"]; LS = ns["ls"]; ELS = ns["els"]; EX = ns["ex"]

    # --- helper: build an rdflib Graph from model triples
    def build(triples: List[Triple]) -> Graph:
        g = Graph()
        g.bind("ct",  Namespace(str(CT)))
        g.bind("ls",  Namespace(str(LS)))
        g.bind("els", Namespace(str(ELS)))
        g.bind("ex",  Namespace(str(EX)))
        g.bind("rdf", ns["rdf"]); g.bind("rdfs", ns["rdfs"])
        g.bind("dcterms", ns["dcterms"]); g.bind("xsd", ns["xsd"])
        for t in triples:
            s = expand(t.subject, ns); p = expand(t.predicate, ns)
            if t.is_literal:
                dt = expand(t.datatype, ns) if t.datatype else None
                g.add((s, p, Literal(t.object, datatype=dt)))
            else:
                g.add((s, p, expand(t.object, ns)))
        return g

    # base graphs from the LLM output (we'll enforce mandatory triples next)
    g_index = build(parsed.index_triples)
    g_link  = build(parsed.linkset_triples)

    # -------- 1) Container mandatory triples --------
    container = EX[f"Container_{app_id}"]
    g_index.add((container, RDF.type, CT.ContainerDescription))
    g_index.add((container, CT.conformanceIndicator,
                 Literal("ICDD-Part1-Container", datatype=XSD.string)))
    # IMPORTANT: publisher must be a resource (IRI), not a literal
    # Easiest SHACL-safe value is the container itself.
    g_index.add((container, CT.publisher, container))

    # -------- 2) Documents (auto-created set) --------
    # values must be RELATIVE to "Payload documents/"
    if auto_docs:
        # Application.json
        doc_app = EX[f"Doc_Application_{app_id}"]
        g_index.add((doc_app, RDF.type, CT.Document))
        g_index.add((doc_app, RDF.type, CT.InternalDocument))
        g_index.add((doc_app, CT.filename, Literal(f"{app_id}/Application.json", datatype=XSD.string)))
        g_index.add((doc_app, CT.filetype, Literal("json", datatype=XSD.string)))
        g_index.add((doc_app, CT.name,  Literal(f"Application {app_id}", datatype=XSD.string)))
        g_index.add((doc_app, CT.belongsToContainer, container))
        g_index.add((container, CT.containsDocument, doc_app))

        # Regulations.txt
        doc_reg = EX[f"Doc_Regulations_{app_id}"]
        g_index.add((doc_reg, RDF.type, CT.Document))
        g_index.add((doc_reg, RDF.type, CT.InternalDocument))
        g_index.add((doc_reg, CT.filename, Literal(f"{app_id}/Regulations.txt", datatype=XSD.string)))
        g_index.add((doc_reg, CT.filetype, Literal("txt", datatype=XSD.string)))
        g_index.add((doc_reg, CT.name,  Literal(f"Regulations {app_id}", datatype=XSD.string)))
        g_index.add((doc_reg, CT.belongsToContainer, container))
        g_index.add((container, CT.containsDocument, doc_reg))

        # Self index (Turtle snapshot for human review)
        doc_idx = EX[f"Doc_Index_{app_id}"]
        g_index.add((doc_idx, RDF.type, CT.Document))
        g_index.add((doc_idx, RDF.type, CT.InternalDocument))
        g_index.add((doc_idx, CT.filename, Literal("BuildingApplicationIndex.ttl", datatype=XSD.string)))
        g_index.add((doc_idx, CT.filetype, Literal("ttl", datatype=XSD.string)))
        g_index.add((doc_idx, CT.name,  Literal(f"Index Turtle {app_id}", datatype=XSD.string)))
        g_index.add((doc_idx, CT.belongsToContainer, container))
        g_index.add((container, CT.containsDocument, doc_idx))

        # materialize the two payload docs for completeness
        (p_docs / f"{app_id}").mkdir(parents=True, exist_ok=True)
        app_src = run_dir / auto_docs.get("application", f"{app_id}/Application.json")
        reg_src = run_dir / auto_docs.get("regulations", f"{app_id}/Regulations.txt")
        (p_docs / f"{app_id}/Application.json").write_text(app_src.read_text() if app_src.exists() else "{}\n")
        (p_docs / f"{app_id}/Regulations.txt").write_text(reg_src.read_text() if reg_src.exists() else "No regulations provided.\n")
        (p_docs / "BuildingApplicationIndex.ttl").write_text(g_index.serialize(format="turtle"))

    # -------- 3) Register linkset in the index --------
    linkset = EX[f"Linkset_{app_id}"]
    g_index.add((linkset, RDF.type, CT.Linkset))
    g_index.add((linkset, CT.filename, Literal("Doc_Application_Links.rdf", datatype=XSD.string)))
    # DO NOT add any back-reference from the linkset to the container (closed shape)
    g_index.add((container, CT.containsLinkset, linkset))

    # -------- 4) Write graphs to disk --------
    # lower-case index.rdf at ZIP root
    (run_dir / "index.rdf").write_text(g_index.serialize(format="xml"))
    # linkset RDF: only link triples (no imports, no extraneous metadata)
    (p_trps / "Doc_Application_Links.rdf").write_text(g_link.serialize(format="xml"))

    # -------- 5) Copy Ontology resources (3 cores + any shapes if present) --------
    def _maybe_copy(src_dir: Path, names: list[str]):
        if not src_dir or not src_dir.exists():
            return
        for name in names:
            src = src_dir / name
            if src.exists():
                dst = p_onts / name
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(src.read_bytes())

    static1 = Path("/workspace/icdd-rag-pipeline/static_resources/ontology_resources")
    core = ["Container.rdf", "Linkset.rdf", "ExtendedLinkset.rdf"]
    shapes = ["Container.shapes.ttl", "Part1ClassesCheck.shapes.rdf", "Part2ClassesCheck.shapes.ttl"]
    _maybe_copy(static1, core + shapes)

    # -------- 6) Zip the container --------
    icdd_zip = out_root / f"ICDD_{app_id}.icdd"
    with zipfile.ZipFile(icdd_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in run_dir.rglob("*"):
            z.write(p, p.relative_to(run_dir))

    return run_dir



# ---- LLM (Transformers) ----
def call_transformers(system: str, user: str, model_id: str, max_new_tokens: int = 320) -> str:
    """
    Robust loader:
      - Forces eager attention (avoids SDPA/enable_gqa issue)
      - Avoids bitsandbytes
      - Falls back automatically to Phi-3-mini if primary model fails to generate
    """
    import gc, torch, traceback
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    use_cuda = torch.cuda.is_available()

    def _load(model_name: str):
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if use_cuda else None,
            torch_dtype=torch.float16 if use_cuda else "auto",
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        mdl.eval()
        return tok, mdl

    def _generate(tok, mdl, system_txt, user_txt, max_tokens: int):
        messages = [{"role": "system", "content": system_txt},
                    {"role": "user",   "content": user_txt}]
        try:
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = f"<|begin_of_text|><|system|>\n{system_txt}\n<|end|><|user|>\n{user_txt}\n<|end|><|assistant|>\n"

        pipe = pipeline("text-generation", model=mdl, tokenizer=tok)  # no device=... here
        out = pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=False,
            return_full_text=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )[0]["generated_text"]
        del pipe
        return out

    try:
        tok, mdl = _load(model_id)
        out = _generate(tok, mdl, system, user, max_new_tokens)
        del tok, mdl
        gc.collect()
        if use_cuda: torch.cuda.empty_cache()
        return out.strip()
    except TypeError as e:
        if "enable_gqa" in str(e):
            try:
                del tok, mdl
            except Exception:
                pass
            gc.collect()
            if use_cuda: torch.cuda.empty_cache()
            fallback = os.getenv("FALLBACK_MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")
            tok, mdl = _load(fallback)
            out = _generate(tok, mdl, system, user, max_new_tokens)
            del tok, mdl
            gc.collect()
            if use_cuda: torch.cuda.empty_cache()
            return out.strip()
        else:
            raise
    except Exception as e1:
        try:
            fallback = os.getenv("FALLBACK_MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")
            tok, mdl = _load(fallback)
            out = _generate(tok, mdl, system, user, max_new_tokens)
            del tok, mdl
            gc.collect()
            if use_cuda: torch.cuda.empty_cache()
            return out.strip()
        except Exception as e2:
            import traceback
            print("Model load/generation failed.\nFirst error:\n",
                  traceback.format_exc(), "\nSecond (fallback) error:\n",
                  "".join(traceback.format_exception(e2)))
            raise SystemExit("Generation failed on both primary and fallback models.")

# ---- Main ----
def main():
    # env
    uri  = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    pwd  = os.getenv("NEO4J_PASSWORD")
    db   = os.getenv("AURA_DB") or "neo4j"
    if not all([uri, user, pwd]):
        raise SystemExit("Missing NEO4J_* env vars.")

    model_id = os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    app_id   = os.getenv("APP_ID", "MUC-2024-0815")

    # 1) KG context
    r = Neo4jRetriever(uri, user, pwd, database=db)
    context = r.retrieve(app_id)  # hops=1 default
    r.close()

    # 2) LightRAG context (optional)
    text_context = dual_level_retrieve(
        query=f"Building application {app_id} regulations and required documents",
        index_dir=os.getenv("LIGHTRAG_INDEX", "/workspace/icdd-rag-pipeline/.lightrag_index"),
        top_k_low=5, top_k_high=5
    )

    # 3) Ensure run folder & minimal payload docs
    run_dir = OUT_ROOT / app_id
    run_dir.mkdir(parents=True, exist_ok=True)
    auto_docs = create_minimal_docs(app_id, run_dir, context)

    # 4) Prompt with both contexts
    user_prompt = USER_TPL.format(
        app_id=app_id,
        kg_context=context,
        text_context=text_context or "(no LightRAG text context available)"
    )

    # 5) Generate JSON + repair + validate
    raw = call_transformers(SYSTEM, user_prompt, model_id=model_id, max_new_tokens=700)
    js  = extract_json_only(raw)
    js_norm = repair_and_normalize_json_payload(js, app_id)
    try:
        parsed = ICDDOutput.model_validate_json(js_norm)
    except Exception:
       # 🔁 One strict retry with the same prompt, then fallback if still bad
       retry_user = user_prompt + "\n\nSTRICT RETRY: Return ONLY valid JSON with keys " \
                               '"index_triples","linkset_triples","provenance". ' \
                               "Each triple is an OBJECT with keys " \
                               '{"subject","predicate","object","is_literal","datatype?"}.'
       raw = call_transformers(SYSTEM, retry_user, model_id=model_id, max_new_tokens=600)
       js  = extract_json_only(raw)
       js_norm = repair_and_normalize_json_payload(js, app_id)
       try:
           parsed = ICDDOutput.model_validate_json(js_norm)
       except Exception:
        # 🛟 Final non-fatal fallback: no LLM triples – code will still build a valid ICDD
            parsed = ICDDOutput(
               index_triples=[],
               linkset_triples=[],
               provenance={"app_id": app_id, "sources": ["no-llm-fallback"]}
            )

    # 6) Post-fixers
    parsed = normalize_subjects_to_instances(parsed, app_id)
    parsed = canonicalize_iris(parsed, app_id)
    parsed = coerce_linkset_shape(parsed, app_id)

    # 7) Write everything + zip
    out_dir = write_outputs(parsed, app_id, OUT_ROOT, auto_docs)
    print("✅ Wrote:", out_dir / "index.rdf")
    print("✅ Wrote:", out_dir / "Payload triples" / "Doc_Application_Links.rdf")
    print("✅ ICDD :", OUT_ROOT / f"ICDD_{app_id}.icdd")

if __name__ == "__main__":
    main()