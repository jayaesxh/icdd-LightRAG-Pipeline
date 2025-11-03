from pathlib import Path

def read_preview(path: Path, max_chars=6000) -> str:
    """
    Return a short string we can use for heuristic/LLM classification.
    Never crashes: always returns something (at least the stem).
    """
    try:
        suf = path.suffix.lower()
        if suf == ".ifc":
            # IFC is plain text; header + entity names is often enough
            return path.read_text(errors="ignore")[:max_chars]
        if suf in [".txt", ".csv", ".xml", ".json", ".md", ".yaml", ".yml"]:
            return path.read_text(errors="ignore")[:max_chars]
        if suf in [".pdf", ".docx", ".dwg", ".dxf", ".png", ".jpg", ".jpeg"]:
            # Keep it simple for now. You can plug in pdfminer/ocr/docx later.
            return path.stem
        return path.stem
    except Exception:
        return path.stem
