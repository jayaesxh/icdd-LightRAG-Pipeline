# main.py  — Transformers-only (no Ollama)
import os, json, zipfile
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, DCTERMS, XSD
from src.retriever import Neo4jRetriever

# ---- Fast knobs (override via env) ----
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "200"))   # was 700
MAX_CTX_CHARS  = int(os.getenv("MAX_CTX_CHARS",  "6000"))  # hard cap on KG text length
MODEL_ID       = os.getenv("MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")  # quicker smoke model



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
USER_TPL = """You get a graph-structured context (nodes + explicit edges).
Rules:
- INDEX (BuildingApplicationIndex.ttl):
  * Create ct:ContainerDescription ex:Container_{app_id}
  * For each document node, create ct:InternalDocument with subject ex:Doc_{{slug}}_{app_id}
    where {{slug}} is a lowercase slug from the document filename:
    - replace non-alphanumeric with "_", collapse repeats, trim leading/trailing "_"
  * Link container -> document via ct:containsDocument
  * Add ct:filename (xsd:string) and ct:filetype (xsd:string) if present
  * Use rdf:type for classes
- LINKS (Doc_Application_Links.rdf):
  * For each obpr:mustComplyWith edge from application to a regulation sentence,
    create an ls:Link of type els:IsControlledBy connecting two ls:LinkElement.
Context:
---
{context}
---
Application ID: {app_id}
Return JSON only."""

# ---- Namespaces & helpers ----
def ns_map(app_id: str) -> dict:
    return {
        "ct":  Namespace("https://standards.iso.org/iso/21597/-1/ed-1/en/Container#"),
        "ls":  Namespace("https://standards.iso.org/iso/21597/-1/ed-1/en/Linkset#"),
        "els": Namespace("https://standards.iso.org/iso/21597/-2/ed-1/en/ExtendedLinkset#"),
        "obpr":Namespace("https://w3id.org/ontobpr#"),
        "ex":  Namespace(f"https://example.org/{app_id}/"),
        "rdf": RDF, "rdfs": RDFS, "dcterms": DCTERMS, "xsd": XSD
    }

def expand(term: Optional[str], ns: dict) -> Optional[URIRef]:
    """Return None if term is missing/unparseable; otherwise a URIRef."""
    if not term:
        return None
    if term.startswith(("http://","https://")):
        return URIRef(term)
    if ":" in term:
        pfx, local = term.split(":",1)
        if pfx in ns:
            try:
                return ns[pfx][local]
            except Exception:
                return None
    # fallback to ex: namespace
    try:
        return ns["ex"][term]
    except Exception:
        return None

def write_outputs(parsed: ICDDOutput, app_id: str, out_root: Path) -> Path:
    run_dir = out_root / app_id
    p_docs = run_dir / "Payload documents"; p_trps = run_dir / "Payload triples"
    p_docs.mkdir(parents=True, exist_ok=True); p_trps.mkdir(parents=True, exist_ok=True)
    ns = ns_map(app_id)

    def build(triples: List[Triple]) -> Graph:
        g = Graph()
        for k, v in ns.items():
            try:
                g.bind(k, v)
            except Exception:
                pass
        for t in triples:
            s = expand(t.subject, ns)
            p = expand(t.predicate, ns)
            if s is None or p is None:
                continue
            if t.is_literal:
                dt = expand(t.datatype, ns) if t.datatype else None
                try:
                    g.add((s, p, Literal(t.object, datatype=dt)))
                except Exception:
                    # skip malformed literal/datatypes
                    continue
            else:
                o = expand(t.object, ns)
                if o is None:
                    continue
                g.add((s, p, o))
        return g

    g1 = build(parsed.index_triples)
    (p_docs / "BuildingApplicationIndex.ttl").write_text(g1.serialize(format="turtle"))

    g2 = build(parsed.linkset_triples)
    (p_trps / "Doc_Application_Links.rdf").write_text(g2.serialize(format="xml"))

    icdd = run_dir / f"{app_id}.icdd"
    # zip everything EXCEPT the zip we’re writing
    with zipfile.ZipFile(icdd, "w", zipfile.ZIP_DEFLATED) as z:
        for p in run_dir.rglob("*"):
            if p.resolve() == icdd.resolve():
                continue
            z.write(p, p.relative_to(run_dir))
    return run_dir

# ---- LLM (Transformers) ----
def call_transformers(system: str, user: str, model_id: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    import os
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    # avoid SDPA/GQA quirks on some CUDA builds
    os.environ["PYTORCH_SDP_DISABLE_FLASH"]="1"
    os.environ["PYTORCH_SDP_DISABLE_MEM_EFFICIENT"]="1"
    os.environ["PYTORCH_SDP_DISABLE_HEURISTIC"]="1"

    tok = AutoTokenizer.from_pretrained(model_id, token=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",          # BF16/FP16 on A100/H100
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    messages = [{"role":"system","content":system},{"role":"user","content":user}]
    try:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"<|begin_of_text|><|system|>\n{system}\n<|end|><|user|>\n{user}\n<|end|><|assistant|>\n"

    gen = pipeline("text-generation", model=mdl, tokenizer=tok)
    out = gen(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,     # deterministic
        return_full_text=False,
        eos_token_id=tok.eos_token_id,
    )[0]["generated_text"]
    return out.strip()

def extract_json_only(txt: str) -> str:
    txt = txt.replace("```json","```").replace("```","")
    s = txt.find("{"); e = txt.rfind("}")
    return txt[s:e+1] if s!=-1 and e!=-1 and e>s else txt

# ---- Main ----
def main():
    uri  = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    pwd  = os.getenv("NEO4J_PASSWORD")
    db   = os.getenv("AURA_DB") or "neo4j"
    if not all([uri, user, pwd]): raise SystemExit("Missing NEO4J_* env vars.")

    model_id = os.getenv("MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    app_id   = os.getenv("APP_ID", "MUC-2024-0815")

    r = Neo4jRetriever(uri, user, pwd, database=db, hops=int(os.getenv("HOPS","1")))  # 1-hop default
context = r.retrieve(app_id); r.close()

   context = context[:MAX_CTX_CHARS]  # cap context
user_prompt = USER_TPL.format(app_id=app_id, context=context)

raw = call_transformers(SYSTEM, user_prompt, model_id=MODEL_ID, max_new_tokens=MAX_NEW_TOKENS)

    js  = extract_json_only(raw)

    try:
        parsed = ICDDOutput.model_validate_json(js)
    except ValidationError as e:
        print("Raw LLM output (first 800 chars):\n", raw[:800])
        raise SystemExit(f"LLM JSON invalid: {e}")

    out_dir = write_outputs(parsed, app_id, OUT_ROOT)
    print("✅ Wrote:", out_dir / "Payload documents" / "BuildingApplicationIndex.ttl")
    print("✅ Wrote:", out_dir / "Payload triples" / "Doc_Application_Links.rdf")
    print("✅ ICDD :", out_dir / f"{app_id}.icdd")

if __name__ == "__main__":
    main()
