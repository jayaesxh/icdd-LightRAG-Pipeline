from pathlib import Path
from dataclasses import dataclass
import os, re, shutil, tempfile, zipfile, time
from typing import List, Tuple

ACCEPT = {".pdf",".ifc",".dwg",".dxf",".docx",".txt",".csv",".xml",".json",".ttl",".md",".yaml",".yml",".rdf"}

def guess_mime_and_filetype(p: Path) -> Tuple[str,str]:
    ext = p.suffix.lower().lstrip(".")
    filetype = ext or "bin"
    mime_map = {
        "pdf":"application/pdf",
        "ifc":"application/octet-stream",
        "dwg":"application/octet-stream",
        "dxf":"application/vnd.dxf",
        "docx":"application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "txt":"text/plain",
        "csv":"text/csv",
        "xml":"application/xml",
        "json":"application/json",
        "ttl":"text/turtle",
        "md":"text/markdown",
        "yaml":"application/yaml",
        "yml":"application/yaml",
        "rdf":"application/rdf+xml",
    }
    return filetype, mime_map.get(ext, "application/octet-stream")

def _derive_case_id_from_name(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-_")
    return stem or "CASE"

def _extract_if_zip(item: Path, tmp_root: Path) -> Path:
    if item.suffix.lower() == ".zip":
        out = tmp_root / item.stem
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(item) as z:
            z.extractall(out)
        return out
    if item.is_dir():
        return item
    out = tmp_root / item.stem
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(item, out / item.name)
    return out

def _collapse_singleton_root(d: Path) -> Path:
    kids = [p for p in d.iterdir() if not p.name.startswith("__MACOSX")]
    if len(kids)==1 and kids[0].is_dir():
        return kids[0]
    return d

@dataclass
class StagedDoc:
    abs_path: Path
    rel_payload_path: str
    name: str
    filetype: str
    fmt: str
    role: str = "Other"
    description: str = ""

def stage_from_upload(upload_item: Path, case_id: str, run_dir: Path) -> List[StagedDoc]:
    p_docs = run_dir / "Payload documents" / case_id
    p_docs.mkdir(parents=True, exist_ok=True)
    staged: List[StagedDoc] = []
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        src_root = _extract_if_zip(upload_item, tdir)
        src_root = _collapse_singleton_root(src_root)
        for p in src_root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in ACCEPT:
                continue
            rel = p.relative_to(src_root)
            dst = p_docs / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            filetype, fmt = guess_mime_and_filetype(dst)
            rel_payload_path = f"{case_id}/{rel.as_posix()}"
            name = p.stem.replace("_"," ").strip() or dst.name
            staged.append(StagedDoc(
                abs_path=dst,
                rel_payload_path=rel_payload_path,
                name=name,
                filetype=filetype,
                fmt=fmt
            ))
    return staged

def discover_upload_items(upload_root: Path):
    zips = sorted([p for p in upload_root.iterdir() if p.suffix.lower()==".zip"])
    dirs = sorted([p for p in upload_root.iterdir() if p.is_dir()])
    files = sorted([p for p in upload_root.iterdir() if p.is_file() and p.suffix.lower()!=".zip"])
    items = []
    items.extend(zips)
    items.extend(dirs)
    if files and not (zips or dirs):
        items.append(upload_root)
    return items

def ensure_clean_run_dir(run_dir: Path):
    if run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / "Payload documents").mkdir(parents=True, exist_ok=True)
    (run_dir / "Payload triples").mkdir(parents=True, exist_ok=True)
    (run_dir / "Ontology resources").mkdir(parents=True, exist_ok=True)

def default_case_id_for_item(item: Path) -> str:
    cid = _derive_case_id_from_name(item.stem if item.suffix.lower()==".zip" else item.name)
    if len(cid) < 6:
        cid = f"{cid}-{int(time.time())}"
    return cid
