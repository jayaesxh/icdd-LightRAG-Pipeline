# main.py — Transformers-only (no Ollama, no APOC)
import os, json, zipfile
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, DCTERMS, XSD, OWL
from src.retriever import Neo4jRetriever
from src.lightrag_client import dual_level_retrieve  # optional, safe if index missing
import json, re 
import os
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
- INDEX (Index.rdf content): Create ct:ContainerDescription ex:Container_{app_id}
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
        "rdf": RDF, "rdfs": RDFS, "dcterms": DCTERMS, "xsd": XSD, "owl": OWL,
    }

def expand(term: Optional[str], ns: dict) -> URIRef:
    if not term:
        return URIRef("")
    term = sanitize_curie(term)

    # Fully qualified
    if term.startswith(("http://", "https://")):
        return URIRef(term)

    # CURIE handling
    if ":" in term:
        pfx, local = term.split(":", 1)
        if pfx in ns:
            return ns[pfx][local]

    # Fallback: put into example namespace
    return ns["ex"][term]


def _sanitize_ct_filename(value: str) -> str:
    v = (value or "").replace("\\", "/")
    return v[len("Payload documents/"):] if v.startswith("Payload documents/") else v

def normalize_subjects_to_instances(parsed: ICDDOutput, app_id: str) -> ICDDOutput:
    """
    Ensure the container subject is an instance, not a class; sanitize ct:filename literals.
    """
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

    # Sanitize curies on all non-literals
    def _clean_row(r):
        r["subject"]   = sanitize_curie(r.get("subject"))
        r["predicate"] = sanitize_curie(r.get("predicate"))
        if not r.get("is_literal", False):
            r["object"] = sanitize_curie(r.get("object"))
        # normalize datatype token
        dt = r.get("datatype")
        if isinstance(dt, str) and dt.lower() == "string":
            r["datatype"] = "xsd:string"
        return r

    data["index_triples"]   = [_clean_row(t) for t in fixed_index]
    data["linkset_triples"] = [_clean_row(t) for t in fixed_link]


def create_minimal_docs(app_id: str, run_dir: Path, kg_context: str):
    """
    Create minimal internal documents so ct:filename points to real files.
    Returns a list of document descriptors you can add to Index.rdf if missing.
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

    # 3) BuildingApplicationIndex.ttl is written later; we just register it here
    index_ttl_rel = "BuildingApplicationIndex.ttl"

    base = f"https://example.org/{app_id}/"
    return [
        {
            "iri": f"{base}Doc_Application_{app_id}",
            "filename": f"{app_id}/Application.json",
            "filetype": "application/json",
            "name": f"Application {app_id}"
        },
        {
            "iri": f"{base}Doc_Regulations_{app_id}",
            "filename": f"{app_id}/Regulations.txt",
            "filetype": "text/plain",
            "name": f"Regulations {app_id}"
        },
        {
            "iri": f"{base}Doc_Index_{app_id}",
            "filename": index_ttl_rel,
            "filetype": "text/turtle",
            "name": f"Index Turtle {app_id}"
        }
    ]

def repair_json_text(txt: str) -> str:
    """
    Lightweight, safe text fixes before Pydantic validation.
    """
    # common key typo
    txt = re.sub(r'"linkset_triple"\s*:', '"linkset_triples":', txt)
    # truncated booleans that models sometimes emit
    txt = re.sub(r'("is_literal"\s*:\s*)fals\b', r'\1false', txt)
    txt = re.sub(r'("is_literal"\s*:\s*)tru\b',  r'\1true', txt)
    return txt
import json, re

def _coerce_triple_row(row):
    """
    Accept either dict or list triple and return a dict:
    {subject, predicate, object, is_literal, datatype?}
    List form may be: [s, p, o], [s,p,o,is_literal], [s,p,o,is_literal,datatype]
    """
    if isinstance(row, dict):
        # normalize datatype spelling if needed
        dt = row.get("datatype")
        if isinstance(dt, str) and dt.lower() == "string":
            row["datatype"] = "xsd:string"
        return row

    if isinstance(row, (list, tuple)) and len(row) >= 3:
        s = row[0]
        p = row[1]
        o = row[2]
        is_lit = False
        dt = None
        if len(row) >= 4 and isinstance(row[3], bool):
            is_lit = row[3]
        if len(row) >= 5 and isinstance(row[4], str):
            dt = "xsd:string" if row[4].lower() == "string" else row[4]
        return {
            "subject": s,
            "predicate": p,
            "object": o,
            "is_literal": bool(is_lit),
            "datatype": dt
        }
    # Unusable entry → drop by returning None
    return None


def repair_and_normalize_json_payload(js_text: str, app_id: str) -> str:
    """
    - Fix common key/boolean typos in raw text.
    - Load to Python, coerce list-form triples to dict-form.
    - Ensure required keys exist and provenance is a dict.
    Returns a JSON string ready for Pydantic validation.
    """
    # light textual repairs
    js_text = re.sub(r'"linkset_triple"\s*:', '"linkset_triples":', js_text)
    js_text = re.sub(r'("is_literal"\s*:\s*)fals\b', r'\1false', js_text)
    js_text = re.sub(r'("is_literal"\s*:\s*)tru\b',  r'\1true',  js_text)

    data = json.loads(js_text)

    # Ensure keys exist
    if "index_triples" not in data or not isinstance(data["index_triples"], list):
        data["index_triples"] = []
    if "linkset_triples" not in data or not isinstance(data["linkset_triples"], list):
        data["linkset_triples"] = []

    # Coerce triples
    fixed_index = []
    for t in data["index_triples"]:
        ct = _coerce_triple_row(t)
        if ct: fixed_index.append(ct)
    fixed_link = []
    for t in data["linkset_triples"]:
        ct = _coerce_triple_row(t)
        if ct: fixed_link.append(ct)
    data["index_triples"] = fixed_index
    data["linkset_triples"] = fixed_link

    # Provenance must be a dict
    prov = data.get("provenance")
    if not isinstance(prov, dict):
        if prov is None:
            data["provenance"] = {"app_id": app_id, "notes": []}
        else:
            data["provenance"] = {"app_id": app_id, "note": str(prov)}

    return json.dumps(data)


def canonicalize_iris(parsed: ICDDOutput, app_id: str) -> ICDDOutput:
    """
    Force container/doc IRIs to the canonical ones for this APP_ID, no matter what the LLM wrote.
    """
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
    """
    If the model returned simple 'ls:controlledBy' between docs, convert it into a proper
    els:IsControlledBy link with link elements and string identifiers.
    If it's already in link-element form, leave as-is.
    """
    has_link_elements = any(t.predicate in ("ls:hasFromLinkElement", "ls:hasToLinkElement")
                            for t in parsed.linkset_triples)
    if has_link_elements:
        return parsed  # already proper

    # collect simplified edges
    edges = [(t.subject, t.object) for t in parsed.linkset_triples if t.predicate in ("ls:controlledBy", "els:IsControlledBy")]
    if not edges:
        # create one minimal link from Application -> Regulations
        edges = [(f"ex:Doc_Application_{app_id}", f"ex:Doc_Regulations_{app_id}")]

    new_links: List[Triple] = []
    for i, (frm, to) in enumerate(edges, 1):
        link = f"ex:Link_{i}_{app_id}"
        le_from = f"{link}#from"
        le_to   = f"{link}#to"
        id_from = f"{link}#id_from"
        id_to   = f"{link}#id_to"

        # link type
        new_links.append(Triple(subject=link, predicate="rdf:type", object="els:IsControlledBy"))
        # link elements
        new_links.append(Triple(subject=le_from, predicate="rdf:type", object="ls:LinkElement"))
        new_links.append(Triple(subject=le_to,   predicate="rdf:type", object="ls:LinkElement"))
        new_links.append(Triple(subject=link,    predicate="ls:hasFromLinkElement", object=le_from))
        new_links.append(Triple(subject=link,    predicate="ls:hasToLinkElement",   object=le_to))
        # document targets
        new_links.append(Triple(subject=le_from, predicate="ls:document", object=frm))
        new_links.append(Triple(subject=le_to,   predicate="ls:document", object=to))
        # identifiers (string-based)
        new_links.append(Triple(subject=id_from, predicate="rdf:type", object="ls:StringBasedIdentifier"))
        new_links.append(Triple(subject=id_from, predicate="ls:identifier", object="whole-doc", is_literal=True, datatype="xsd:string"))
        new_links.append(Triple(subject=le_from, predicate="ls:hasIdentifier", object=id_from))

        new_links.append(Triple(subject=id_to, predicate="rdf:type", object="ls:StringBasedIdentifier"))
        new_links.append(Triple(subject=id_to, predicate="ls:identifier", object="reg-ctx", is_literal=True, datatype="xsd:string"))
        new_links.append(Triple(subject=le_to, predicate="ls:hasIdentifier", object=id_to))

    return ICDDOutput(index_triples=parsed.index_triples, linkset_triples=new_links, provenance=parsed.provenance)

def sanitize_curie(term: Optional[str]) -> Optional[str]:
    if not isinstance(term, str):
        return term
    t = term.strip()
    # Drop leading colons like ":ex:...", ":ct:..."
    while t.startswith(":"):
        t = t[1:]
    # Collapse accidental double-colons like "ct::containsDocument"
    t = re.sub(r"([A-Za-z0-9_]+)::", r"\1:", t)
    return t


def _locate_ontology_resources() -> Optional[Path]:
    """
    Return the first existing ontology resources directory among common names.
    We normalize to the canonical container folder name when copying.
    """
    candidates = [
        Path("/workspace/icdd-rag-pipeline/static_resources/Ontology resources"),
        Path("/workspace/icdd-rag-pipeline/static_resources/ontology_resources"),
        Path("/workspace/icdd-rag-pipeline/static_resources/ontologies"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def write_outputs(parsed: ICDDOutput, app_id: str, out_root: Path, auto_docs: List[dict]) -> Path:
    run_dir = out_root / app_id
    # required top-level folders per ISO
    (run_dir / "Ontology resources").mkdir(parents=True, exist_ok=True)
    (run_dir / "Payload documents").mkdir(parents=True, exist_ok=True)
    (run_dir / "Payload triples").mkdir(parents=True, exist_ok=True)

    ns = ns_map(app_id)
    EX, CT, LS, ELS, OWL_ns, XSD_ns = ns["ex"], ns["ct"], ns["ls"], ns["els"], ns["owl"], ns["xsd"]

    def build(triples: List[Triple]) -> Graph:
        g = Graph()
        for k, v in ns.items(): g.bind(k, v)
        for t in triples:
            s = expand(t.subject, ns); p = expand(t.predicate, ns)
            if t.is_literal:
                dt = expand(t.datatype, ns) if t.datatype else None
                g.add((s, p, Literal(t.object, datatype=dt)))
            else:
                g.add((s, p, expand(t.object, ns)))
        return g

    # --- Index graph from LLM triples ---
    g_index = build(parsed.index_triples)

    # Ensure there is one container individual with required imports + conformance indicator
    container_iri = EX[f"Container_{app_id}"]
    g_index.add((container_iri, RDF.type, CT.ContainerDescription))
    g_index.add((container_iri, OWL_ns.imports,
                 URIRef("https://standards.iso.org/iso/21597/-1/ed-1/en/Container.rdf")))
    g_index.add((container_iri, CT.conformanceIndicator,
                 Literal("ICDD-Part1-Container", datatype=XSD_ns.string)))

    # Ensure the three minimal docs exist in the graph
    present_docs = set(str(s) for s, _, _ in g_index.triples((None, RDF.type, CT.InternalDocument)))
    for d in auto_docs:
        D = URIRef(d["iri"])
        if str(D) not in present_docs:
            g_index.add((D, RDF.type, CT.InternalDocument))
            g_index.add((container_iri, CT.containsDocument, D))
            g_index.add((D, CT.filename, Literal(_sanitize_ct_filename(d["filename"]), datatype=XSD_ns.string)))
            if ft := d.get("filetype"):
                g_index.add((D, CT.filetype, Literal(ft, datatype=XSD_ns.string)))
            if nm := d.get("name"):
                g_index.add((D, CT.name, Literal(nm, datatype=XSD_ns.string)))

    # Write the **root** Index.rdf (RDF/XML) + dev Turtle copy
    index_path = run_dir / "Index.rdf"
    g_index.serialize(destination=str(index_path), format="xml")
    (run_dir / "Payload documents" / "BuildingApplicationIndex.ttl").write_text(
        g_index.serialize(format="turtle"), encoding="utf-8"
    )

    # --- Linkset graph from LLM triples ---
    g_link = build(parsed.linkset_triples)
    g_link.add((container_iri, OWL_ns.imports, URIRef("Index.rdf")))
    g_link.add((container_iri, OWL_ns.imports,
                URIRef("https://standards.iso.org/iso/21597/-1/ed-1/en/Linkset.rdf")))
    linkset_path = run_dir / "Payload triples" / "Doc_Application_Links.rdf"
    g_link.serialize(destination=str(linkset_path), format="xml")  # RDF/XML

    # --- (Optional) Copy local ontologies if you have them in static_resources ---
    static_ont = _locate_ontology_resources()
    if static_ont:
        desired = {
            "Container.rdf":        ["Container.rdf", "container.rdf"],
            "Linkset.rdf":          ["Linkset.rdf", "Linkset (1).rdf", "linkset.rdf"],
            "ExtendedLinkset.rdf":  ["ExtendedLinkset.rdf", "extendedlinkset.rdf"],
        }
        for out_name, variants in desired.items():
            src = None
            for v in variants:
                cand = static_ont / v
                if cand.exists():
                    src = cand
                    break
            if src:
                dst = run_dir / "Ontology resources" / out_name  # canonical name inside the container
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    dst.write_bytes(src.read_bytes())

    # --- Build the .icdd OUTSIDE run_dir to avoid zipping itself ---
    icdd = out_root / f"ICDD_{app_id}.icdd"
    with zipfile.ZipFile(icdd, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for p in run_dir.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(run_dir))
    return run_dir

# ---- LLM (Transformers) ----
def call_transformers(system: str, user: str, model_id: str, max_new_tokens: int = 320) -> str:
    """
    Robust, error-proof loader:
      - Forces EAGER attention (avoids SDPA + enable_gqa issues)
      - Avoids bitsandbytes entirely
      - If a model still hits the SDPA path, auto-fallbacks to a safe small model
        (microsoft/Phi-3-mini-4k-instruct) and continues.
    """
    import os, gc, torch, traceback
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    use_cuda = torch.cuda.is_available()

    def _load(model_name: str):
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if use_cuda else None,   # let accelerate place it
            torch_dtype=torch.float16 if use_cuda else "auto",
            low_cpu_mem_usage=True,
            attn_implementation="eager",               # <- force eager attention (no SDPA)
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

        # IMPORTANT: do NOT pass device=... when model loaded with device_map/accelerate
        pipe = pipeline("text-generation", model=mdl, tokenizer=tok)
        out = pipe(
            prompt,
            max_new_tokens=max_tokens,   # 250–350 is plenty for strict JSON
            do_sample=False,             # deterministic
            return_full_text=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )[0]["generated_text"]
        del pipe
        return out

    # 1) Try with the requested model (eager attention)
    try:
        tok, mdl = _load(model_id)
        out = _generate(tok, mdl, system, user, max_new_tokens)
        del tok, mdl
        gc.collect()
        if use_cuda: torch.cuda.empty_cache()
        return out.strip()
    except TypeError as e:
        # Catch the specific SDPA/enable_gqa issue and fall back automatically
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
            # Re-raise other TypeErrors (unrelated to SDPA)
            raise
    except Exception:
        # As a last resort, try the fallback model too
        try:
            fallback = os.getenv("FALLBACK_MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")
            tok, mdl = _load(fallback)
            out = _generate(tok, mdl, system, user, max_new_tokens)
            del tok, mdl
            gc.collect()
            if use_cuda: torch.cuda.empty_cache()
            return out.strip()
        except Exception as e2:
            print("Model load/generation failed.\nFirst error:\n",
                  traceback.format_exc(), "\nSecond (fallback) error:\n",
                  "".join(traceback.format_exception(e2)))
            raise SystemExit("Generation failed on both primary and fallback models.")




def extract_json_only(txt: str) -> str:
    txt = txt.replace("```json","```").replace("```","")
    s = txt.find("{"); e = txt.rfind("}")
    return txt[s:e+1] if s!=-1 and e!=-1 and e>s else txt

# ---- Main ----
def main():
    # --- env & basics ---
    uri  = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    pwd  = os.getenv("NEO4J_PASSWORD")
    db   = os.getenv("AURA_DB") or "neo4j"
    if not all([uri, user, pwd]):
        raise SystemExit("Missing NEO4J_* env vars.")

    # pick an open model unless you’re 100% approved for Meta-Llama
    model_id = os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    app_id   = os.getenv("APP_ID", "MUC-2024-0815")

    # --- 1) KG context ---
    r = Neo4jRetriever(uri, user, pwd, database=db)
    context = r.retrieve(app_id)  # hops=1 by default
    r.close()

    # --- 2) LightRAG dual-level text context (optional; "" if no index available) ---
    text_context = dual_level_retrieve(
        query=f"Building application {app_id} regulations and required documents",
        index_dir=os.getenv("LIGHTRAG_INDEX", "/workspace/icdd-rag-pipeline/.lightrag_index"),
        top_k_low=5, top_k_high=5
    )

    # --- 3) Ensure run folder & create minimal internal docs we will reference via ct:filename ---
    run_dir = OUT_ROOT / app_id
    run_dir.mkdir(parents=True, exist_ok=True)
    auto_docs = create_minimal_docs(app_id, run_dir, context)

    # --- 4) Build the combined prompt (KG + LightRAG contexts) ---
    user_prompt = USER_TPL.format(
        app_id=app_id,
        kg_context=context,
        text_context=text_context or "(no LightRAG text context available)"
    )

    # --- 5) Call local Transformers LLM, extract strict JSON, validate ---
    raw = call_transformers(SYSTEM, user_prompt, model_id=model_id, max_new_tokens=700)
    js  = extract_json_only(raw)

    # First, normalize payload shape (lists -> objects, provenance to dict)
    try:
        js_norm = repair_and_normalize_json_payload(js, app_id)
        parsed = ICDDOutput.model_validate_json(js_norm)
    except Exception:
    # One strict retry if the first attempt still fails
        retry_user = user_prompt + "\n\nSTRICT RETRY: Your last output used list-form triples. " \
                               "Return ONLY valid JSON with triples as objects: " \
                               '{"subject","predicate","object","is_literal","datatype?"}.'
        raw = call_transformers(SYSTEM, retry_user, model_id=model_id, max_new_tokens=700)
        js   = extract_json_only(raw)
        js_norm = repair_and_normalize_json_payload(js, app_id)
        parsed  = ICDDOutput.model_validate_json(js_norm)

    # Post-fixes you already had
    parsed = normalize_subjects_to_instances(parsed, app_id)
    parsed = canonicalize_iris(parsed, app_id)
    parsed = coerce_linkset_shape(parsed, app_id)



    # --- 6) Post-fix subjects to be instances (not classes), sanitize filenames ---
    parsed = normalize_subjects_to_instances(parsed, app_id)

    # --- 7) Write Index.rdf (root) + Linkset (RDF/XML) + .icdd zip ---
    out_dir = write_outputs(parsed, app_id, OUT_ROOT, auto_docs)
    print("✅ Wrote:", out_dir / "Index.rdf")
    print("✅ Wrote:", out_dir / "Payload triples" / "Doc_Application_Links.rdf")
    print("✅ ICDD :", OUT_ROOT / f"ICDD_{app_id}.icdd")

if __name__ == "__main__":
    main()
