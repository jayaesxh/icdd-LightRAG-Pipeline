import json, re, yaml
from pathlib import Path
from extractors import read_preview

def _load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def load_roles(base_yaml: Path, local_yaml: Path | None = None):
    """
    Load base roles.yaml and optionally merge roles.local.yaml (overrides or adds).
    Returns: compiled_patterns, label_set
    """
    cfg = _load_yaml(base_yaml)
    if local_yaml and local_yaml.exists():
        local_cfg = _load_yaml(local_yaml)
        # merge/override 'roles'
        cfg["roles"].update(local_cfg.get("roles", {}))

    compiled = []
    labels = []
    for role, body in cfg["roles"].items():
        pats = body.get("patterns", [])
        regs = [re.compile(p, re.I) for p in pats]
        compiled.append((role, regs))
        labels.append(role)
    return compiled, labels

def heuristic_role(filename: str, preview: str, compiled):
    text = f"{filename.lower()} {preview.lower()}"
    scores = {}
    for role, regs in compiled:
        scores[role] = sum(1 for r in regs if r.search(text))
    if any(scores.values()):
        role = max(scores, key=scores.get)
        conf = scores[role] / max(1, sum(scores.values()))
    else:
        role, conf = "Other", 0.0
    return role, conf

def classify_documents_ensemble(paths, roles_yaml: Path, roles_yaml_local: Path | None = None, llm=None):
    """
    paths: list[Path] to staged files
    returns: list[dict] with keys: rel_filename, role, name, description
    """
    compiled, label_set = load_roles(roles_yaml, roles_yaml_local)

    out = []
    for p in paths:
        prev = read_preview(p)
        r_h, c_h = heuristic_role(p.name, prev, compiled)

        # keep it heuristic-only for now (llm plug later)
        final_role = r_h
        name = p.stem.replace("_"," ")
        desc = f"{final_role} for permit case."

        out.append({
            "rel_filename": str(p),  # caller should pass RELATIVE paths already
            "role": final_role,
            "name": name,
            "description": desc
        })
    return out
