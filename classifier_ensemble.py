import json, re, yaml
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any, Optional
from extractors import read_preview

__all__ = [
    "load_roles",
    "heuristic_role",
    "llm_json",
    "classify_documents_ensemble",
]

def load_roles(yaml_path: Path) -> Tuple[list, list]:
    """Return (compiled_patterns, label_set).
    compiled_patterns: List[Tuple[str, List[re.Pattern]]]
    label_set: List[str]
    """
    cfg = yaml.safe_load(Path(yaml_path).read_text())
    roles = cfg.get("roles") or {}
    compiled = []
    for role, data in roles.items():
        pats = data.get("patterns") or []
        regs = [re.compile(p, re.I) for p in pats]
        compiled.append((role, regs))
    label_set = list(roles.keys())
    # Ensure 'Other' exists as a safety class
    if "Other" not in label_set:
        compiled.append(("Other", []))
        label_set.append("Other")
    return compiled, label_set

def heuristic_role(filename: str, preview: str, compiled) -> Tuple[str, float]:
    """Very simple scoring by pattern matches over filename+preview."""
    text = f"{filename.lower()}\n{(preview or '').lower()}"
    scores = {}
    for role, regs in compiled:
        scores[role] = sum(1 for r in regs if r.search(text))
    if any(scores.values()):
        role = max(scores, key=scores.get)
        conf = scores[role] / max(1, sum(scores.values()))
        return role, float(conf)
    return "Other", 0.0

def llm_json(prompt_callable, filename: str, preview: str, label_set: List[str]) -> Optional[Dict[str, Any]]:
    """Call an external LLM function if provided.
    Must return STRICT JSON with keys: role,name,description.
    Fallback to None on any failure.
    """
    if not prompt_callable:
        return None
    prompt = (
        "You classify building-permit docs. Choose exactly one role from:\n"
        f"{label_set}\n"
        "Return STRICT JSON only: {\"role\":\"...\",\"name\":\"...\",\"description\":\"...\"}\n\n"
        f"FILENAME: {filename}\nPREVIEW:\n{(preview or '')[:2000]}"
    )
    try:
        out = prompt_callable(prompt)
        if isinstance(out, str):
            out = json.loads(out)
        if not isinstance(out, dict):
            return None
        if not {"role","name","description"}.issubset(out.keys()):
            return None
        if out["role"] not in label_set:
            out["role"] = "Other"
        return out
    except Exception:
        return None

def _get_attr(obj: Any, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)

def _set_attrs(obj: Any, **fields):
    if isinstance(obj, dict):
        obj.update(fields)
        return obj
    for k, v in fields.items():
        setattr(obj, k, v)
    return obj

def classify_documents_ensemble(
    docs: Iterable[Any],
    roles_yaml: Path,
    llm: Optional[callable] = None
) -> List[Any]:
    """Classify staged documents.
    Each 'doc' can be a simple object or dict with at least: path (str/Path).
    We set/override: role, name, description.
    """
    compiled, label_set = load_roles(Path(roles_yaml))
    results = []
    for d in docs:
        p = Path(_get_attr(d, "path"))
        prev = read_preview(p)
        r_h, c_h = heuristic_role(p.name, prev, compiled)
        jl = llm_json(llm, p.name, prev, label_set)
        if jl:
            # prefer LLM only if it disagrees AND heuristic was weak
            final_role = jl.get("role") if (jl.get("role") != r_h and c_h < 0.6) else r_h
            name = jl.get("name") or p.stem.replace("_", " ")
            desc = jl.get("description") or f"{final_role} for permit case."
        else:
            final_role = r_h
            name = p.stem.replace("_", " ")
            desc = f"{final_role} for permit case."
        results.append(_set_attrs(d, role=final_role, name=name, description=desc))
    return results
