#!/usr/bin/env python3
# /workspace/icdd-rag-pipeline/driver_upload.py

import os, re, mimetypes, shutil, inspect, zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from rdflib import Namespace, URIRef, Graph, Literal
from rdflib.namespace import RDF, XSD

# --- your modules ---
import icdd_builder
try:
    import upload_ingest
except Exception:
    upload_ingest = None

try:
    from classifier_ensemble import classify_documents_ensemble
except Exception:
    classify_documents_ensemble = None  # safe fallback

UPLOAD_ROOT = Path("/workspace/icdd-rag-pipeline/upload")
OUT_ROOT    = Path("/workspace/icdd-rag-pipeline/output")
STATIC_ONT  = Path("/workspace/icdd-rag-pipeline/static_resources/ontology_resources")
ROLES_YAML  = Path("/workspace/icdd-rag-pipeline/roles.yaml")

CT  = Namespace("https://standards.iso.org/iso/21597/-1/ed-1/en/Container#")
LS  = Namespace("https://standards.iso.org/iso/21597/-1/ed-1/en/Linkset#")
ELS = Namespace("https://standards.iso.org/iso/21597/-2/ed-1/en/ExtendedLinkset#")

# ---------- small utilities ----------
def make_case_id(p: Path) -> str:
    base = p.stem if p.is_file() else p.name
    cid = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-_.")
    return (cid or "case")[:80]

def guess_mime_by_ext(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".json": return "application/json"
    if ext == ".ttl":  return "text/turtle"
    if ext == ".rdf":  return "application/rdf+xml"
    if ext == ".txt":  return "text/plain"
    if ext == ".pdf":  return "application/pdf"
    if ext == ".ifc":  return "application/octet-stream"
    m = mimetypes.guess_type(filename)[0]
    return m or "application/octet-stream"

def call_by_signature(fn, **pool):
    sig = inspect.signature(fn)
    kwargs = {}
    for name in sig.parameters:
        low = name.lower()
        if "run" in low and "dir" in low and "run_dir" in pool:
            kwargs[name] = pool["run_dir"]
        elif "case" in low and "id" in low and "case_id" in pool:
            kwargs[name] = pool["case_id"]
        elif any(k in low for k in ("item","upload","path","src","source")) and "item" in pool:
            kwargs[name] = pool["item"]
        elif "out" in low and "root" in low and "out_root" in pool:
            kwargs[name] = pool["out_root"]
    return fn(**kwargs)

# --- ensure run dir (module or fallback) ---
def ensure_clean_run_dir(out_root: Path, case_id: str) -> Path:
    if upload_ingest and hasattr(upload_ingest, "ensure_clean_run_dir"):
        try:
            return call_by_signature(upload_ingest.ensure_clean_run_dir, out_root=out_root, case_id=case_id)
        except TypeError:
            pass
    run_dir = out_root / case_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / "Ontology resources").mkdir(parents=True, exist_ok=True)
    (run_dir / "Payload documents" / case_id).mkdir(parents=True, exist_ok=True)
    (run_dir / "Payload triples").mkdir(parents=True, exist_ok=True)
    return run_dir

# --- stage uploads (module or fallback) ---
def fallback_stage(item: Path, case_id: str, run_dir: Path) -> List[Dict[str,str]]:
    payload_root = run_dir / "Payload documents" / case_id
    payload_root.mkdir(parents=True, exist_ok=True)
    rels = []

    def add_file(abs_path: Path):
        rel = f"{case_id}/{abs_path.name}"
        dst = payload_root / abs_path.name
        shutil.copy2(abs_path, dst)
        rels.append({"rel": rel, "abs": str(dst)})

    if item.is_file() and item.suffix.lower() == ".zip":
        with zipfile.ZipFile(item) as z:
            for name in z.namelist():
                if name.endswith("/"):
                    continue
                dst = payload_root / Path(name).name
                with z.open(name) as src, open(dst, "wb") as out:
                    shutil.copyfileobj(src, out)
                rels.append({"rel": f"{case_id}/{dst.name}", "abs": str(dst)})
    elif item.is_dir():
        for p in item.rglob("*"):
            if p.is_file():
                add_file(p)
    elif item.is_file():
        add_file(item)
    return rels

def stage_from_upload(item: Path, case_id: str, run_dir: Path):
    # try upload_ingest if available
    if upload_ingest and hasattr(upload_ingest, "stage_from_upload"):
        for attempt in (
            lambda: call_by_signature(upload_ingest.stage_from_upload, item=item, case_id=case_id, run_dir=run_dir),
            lambda: upload_ingest.stage_from_upload(item, run_dir),
            lambda: upload_ingest.stage_from_upload(item, case_id, run_dir),
            lambda: upload_ingest.stage_from_upload(run_dir, item, case_id),
        ):
            try:
                return attempt()
            except TypeError:
                continue
    # fallback
    return fallback_stage(item, case_id, run_dir)

def coerce_to_rel_abs_list(staged_items, run_dir: Path, case_id: str):
    payload_root = run_dir / "Payload documents" / case_id
    out = []

    def pick(o, *keys):
        for k in keys:
            if isinstance(o, dict) and k in o:
                return o[k]
            if hasattr(o, k):
                return getattr(o, k)
        return None

    def normalize_one(it):
        rel_cand = pick(it, "rel_filename","rel_path","relative","payload_rel","dst_rel","dest_rel","rel")
        abs_cand = pick(it, "abs_path","dst_path","dest_path","absolute","path","dst","dest","abs")

        if isinstance(it, (str, Path)) and rel_cand is None and abs_cand is None:
            p = Path(it)
            if p.is_absolute():
                abs_cand = str(p)
                try:
                    rel_cand = str(p.relative_to(payload_root)).replace("\\","/")
                except Exception:
                    rel_cand = p.name
            else:
                rel_cand = str(p)
                abs_cand = str((payload_root / p).resolve())

        if rel_cand is None and abs_cand is not None:
            try:
                rel_cand = str(Path(abs_cand).relative_to(payload_root)).replace("\\","/")
            except Exception:
                rel_cand = Path(abs_cand).name

        if abs_cand is None and rel_cand is not None:
            abs_cand = str((payload_root / rel_cand).resolve())

        if not rel_cand or not abs_cand:
            return None
        if not Path(abs_cand).exists():
            return None
        return {"rel": rel_cand, "abs": abs_cand}

    for it in (staged_items or []):
        norm = normalize_one(it)
        if norm: out.append(norm)
    return out

# --- Doc adapter expected by icdd_builder.build_index_graph ---
class Doc:
    __slots__ = ("iri","filename","rel_filename","rel_path","filetype","format","name","description","role")
    def __init__(self, iri: URIRef, rel_filename: str, filetype: str, fmt: str, name: str, description: str, role: str):
        self.iri = iri
        self.filename = rel_filename
        self.rel_filename = rel_filename
        self.rel_path = rel_filename         # some versions read rel_path
        self.filetype = filetype
        self.format = fmt
        self.name = name
        self.description = description
        self.role = role

def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+","-", s).strip("-.")

def doc_uri(case_id: str, rel_filename: str) -> URIRef:
    stem = slug(Path(rel_filename).stem)
    return URIRef(f"https://example.org/{case_id}/Doc_{stem}_{case_id}")

def docs_to_objects(rel_abs_list, classifications, case_id: str):
    out = []
    for ra, cls in zip(rel_abs_list, classifications):
        rel = ra["rel"]
        iri = doc_uri(case_id, rel)
        ext = Path(rel).suffix.lower().lstrip(".") or "bin"
        fmt = guess_mime_by_ext(rel)
        name = (cls or {}).get("name") or Path(rel).name
        desc = (cls or {}).get("description") or f"Document {Path(rel).name}"
        role = (cls or {}).get("role") or "Other"
        out.append(Doc(iri, rel, ext, fmt, name, desc, role))
    return out

# --- roles.yaml minimal (safe) ---
if not ROLES_YAML.exists():
    ROLES_YAML.write_text(
        "roles:\n"
        "  Building Application:\n"
        "    patterns: ['bauantrag','antrag','formular','application']\n"
        "  Site Plan:\n"
        "    patterns: ['lageplan','site plan','kataster','flur','map']\n"
        "  Building Model IFC:\n"
        "    patterns: ['\\\\.ifc$','ifc4','ifc2x3','modell','model']\n"
        "  Thermal Insulation Report:\n"
        "    patterns: ['wärme','dämm','energie','geg','enev','thermal']\n"
        "  Sound Insulation Report:\n"
        "    patterns: ['schall','lärm','sound','noise']\n"
        "  Structural Certificate:\n"
        "    patterns: ['statik','trag','stabil','structural']\n"
        "  Accessibility Report:\n"
        "    patterns: ['barriere','accessibility','din 18040']\n"
    )

# --- resilient writer (tries icdd_builder signatures, else writes a valid ICDD itself) ---
def write_icdd_files_flex(case_id: str, g_index: Graph, g_link: Graph, run_dir: Path, out_root: Path) -> Path:
    # try common signatures
    try: return icdd_builder.write_icdd_files(case_id, g_index, g_link)
    except TypeError: pass
    try: return icdd_builder.write_icdd_files(g_index, g_link, case_id)
    except TypeError: pass
    try: return icdd_builder.write_icdd_files(case_id=case_id, g_index=g_index, g_link=g_link)
    except TypeError: pass
    try: return icdd_builder.write_icdd_files(run_dir, g_index, g_link, case_id=case_id)
    except TypeError: pass

    # fallback: write index.rdf + linkset + pack zip with ISO layout
    idx_path  = run_dir / "index.rdf"
    link_path = run_dir / "Payload triples" / "Doc_Application_Links.rdf"
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.parent.mkdir(parents=True, exist_ok=True)
    g_index.serialize(destination=str(idx_path), format="xml")
    g_link.serialize(destination=str(link_path),  format="xml")

    ont_dir = run_dir / "Ontology resources"
    ont_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("Container.rdf","Linkset.rdf","ExtendedLinkset.rdf"):
        dst = ont_dir / fname
        src = STATIC_ONT / fname
        if not dst.exists() and src.exists():
            shutil.copy(src, dst)

    out_zip = out_root / f"ICDD_{case_id}.icdd"
    if out_zip.exists(): out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w") as z:
        z.write(idx_path, "index.rdf")
        for folder in ("Ontology resources","Payload documents","Payload triples"):
            base = run_dir / folder
            if base.exists():
                for f in sorted(base.rglob("*")):
                    if f.is_file():
                        z.write(f, f.relative_to(run_dir).as_posix())
    return out_zip

# --- main driver ---
def main():
    items = []
    if upload_ingest and hasattr(upload_ingest, "discover_upload_items"):
        try:
            items = upload_ingest.discover_upload_items(UPLOAD_ROOT)
        except Exception:
            pass
    if not items:
        # simple discovery: zip, folder, or files directly under UPLOAD_ROOT
        for p in sorted(UPLOAD_ROOT.iterdir()):
            if p.is_dir() or p.suffix.lower() == ".zip" or p.is_file():
                items.append(p)

    if not items:
        print("No submissions found in", UPLOAD_ROOT)
        return

    print("Found submissions:")
    for it in items: print(" -", it)

    for item in items:
        item = Path(item)
        CASE_ID = make_case_id(item)
        print(f"\n=== Processing {item.name} → CASE_ID={CASE_ID} ===")

        run_dir = ensure_clean_run_dir(OUT_ROOT, CASE_ID)
        staged_raw = stage_from_upload(item, CASE_ID, run_dir)
        rel_abs_list = coerce_to_rel_abs_list(staged_raw, run_dir, CASE_ID)
        if not rel_abs_list:
            print("No files discovered; skipping.")
            continue

        # classify (safe fallbacks)
        if classify_documents_ensemble:
            try:
                docs_abs = [Path(d["abs"]) for d in rel_abs_list]
                classified = classify_documents_ensemble(docs_abs, ROLES_YAML, llm=None)
                cls_dicts = []
                for i, obj in enumerate(classified):
                    role = getattr(obj, "role", None) or "Other"
                    nm   = getattr(obj, "name", None) or docs_abs[i].stem
                    desc = getattr(obj, "description", None) or f"{role} for permit case."
                    cls_dicts.append({"role": role, "name": nm, "description": desc})
            except Exception as e:
                print("(classifier fallback:", e, ")")
                cls_dicts = [{"role":"Other","name":Path(d["rel"]).stem,"description":f"Document {Path(d['rel']).name}"} for d in rel_abs_list]
        else:
            cls_dicts = [{"role":"Other","name":Path(d["rel"]).stem,"description":f"Document {Path(d['rel']).name}"} for d in rel_abs_list]

        doc_objs = docs_to_objects(rel_abs_list, cls_dicts, CASE_ID)

        # graphs (let your builder guarantee the ISO/SHACL details)
        g_link = icdd_builder.build_linkset_graph(CASE_ID, [])  # start with empty linkset
        g_index = icdd_builder.build_index_graph(
            case_id=CASE_ID,
            docs=doc_objs,
            linkset_filename="Doc_Application_Links.rdf",
            publisher_text="icdd-rag-pipeline"
        )

        # write + check
        zip_path = write_icdd_files_flex(CASE_ID, g_index, g_link, run_dir, OUT_ROOT)
        print("✅ ICDD written:", zip_path)
        try:
            icdd_builder.quick_structural_check(run_dir)
        except TypeError:
            try:
                icdd_builder.quick_structural_check(str(run_dir))
            except Exception as e:
                print("(quick_structural_check skipped:", e, ")")

if __name__ == "__main__":
    main()
