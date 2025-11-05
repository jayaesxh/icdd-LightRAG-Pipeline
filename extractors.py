from pathlib import Path

def read_preview(path: Path, max_chars: int = 6000) -> str:
    """Lightweight, dependency-free preview extraction.
    Fallback: return filename stem if anything fails.
    """
    try:
        suf = path.suffix.lower()
        if suf in {".txt", ".csv", ".xml", ".json", ".md", ".yaml", ".yml"}:
            return path.read_text(errors="ignore")[:max_chars]
        if suf == ".ifc":
            # IFC is text; header + first N chars are often enough
            return path.read_text(errors="ignore")[:max_chars]
        # For pdf/docx/dwg/jpg/png: keep it dependency-free here; return filename
        return path.stem
    except Exception:
        return path.stem
