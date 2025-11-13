from __future__ import annotations
import re, subprocess, shutil
from pathlib import Path
from typing import Dict, Any, Optional, List

def enrich_ifc_text(path: Path, max_chars: int = 16000) -> str:
    """
    Try ifcopenshell first; if unavailable, do STEP text scanning for key facts.
    Returns a text blob to feed into RAG.
    """
    try:
        import ifcopenshell  # pip install ifcopenshell
        f = ifcopenshell.open(str(path))
        out = []
        proj = f.by_type("IfcProject")
        if proj:
            out.append(f"IfcProject: {proj[0].Name or ''}")
        sites = f.by_type("IfcSite")
        if sites:
            s = sites[0]
            out.append(f"IfcSite: {s.Name or ''}")
            try:
                addr = s.RefLatitude, s.RefLongitude
                out.append(f"SiteRef: latlon={addr}")
            except Exception:
                pass
        bldg = f.by_type("IfcBuilding")
        if bldg:
            out.append(f"IfcBuilding: {bldg[0].Name or ''}")
        storeys = f.by_type("IfcBuildingStorey")
        if storeys:
            out.append(f"Storeys: {len(storeys)}")
        # Postal address
        addrs = f.by_type("IfcPostalAddress")
        if addrs:
            a0 = addrs[0]
            out.append(f"PostalAddress: {a0.Town or ''}, {a0.Region or ''}, {a0.PostalCode or ''}")
        return "\n".join(out)[:max_chars]
    except Exception:
        # Fallback: STEP text scan
        txt = path.read_text("utf-8", errors="ignore")
        lines = []
        for pat, lbl in [
            (r"IFCPROJECT\((.*?)\)", "IfcProject"),
            (r"IFCBUILDING\((.*?)\)", "IfcBuilding"),
            (r"IFCBUILDINGSTOREY\(", "IfcBuildingStorey"),
            (r"IFCPOSTALADDRESS\((.*?)\)", "IfcPostalAddress"),
        ]:
            m = re.search(pat, txt, re.IGNORECASE | re.DOTALL)
            if m:
                lines.append(f"{lbl}: {m.group(0)[:300]}")
        # crude storey count
        storey_count = len(re.findall(r"IFCBUILDINGSTOREY", txt, re.IGNORECASE))
        lines.append(f"Storeys~approx: {storey_count}")
        return "\n".join(lines)[:max_chars]

def enrich_dwg_or_dxf_text(path: Path, max_chars: int = 16000) -> str:
    """
    DWG: try to convert to DXF using ODA Converter if available, then parse text via ezdxf.
    DXF: parse directly. If neither path works, return a short placeholder.
    """
    ext = path.suffix.lower()
    target = path
    if ext == ".dwg":
        oda = shutil.which("ODAFileConverter") or shutil.which("ODAFileConverter.exe")
        if oda:
            out_dxf = path.with_suffix(".dxf")
            try:
                # ODA CLI syntax can vary; adjust as needed in your environment.
                subprocess.run([oda, str(path.parent), str(path.parent), "ACAD2013", "DXF", "0", "1"],
                               check=True)
                if out_dxf.exists():
                    target = out_dxf
            except Exception:
                pass
        else:
            return "[DWG text unavailable; no converter present]"
    # Parse DXF text with ezdxf if available
    try:
        import ezdxf  # pip install ezdxf
        doc = ezdxf.readfile(str(target))
        msp = doc.modelspace()
        texts = []
        for e in msp.query("TEXT MTEXT"):
            try:
                t = e.text if e.dxftype() == "TEXT" else e.plain_text()
                if t and t.strip():
                    texts.append(t.strip())
            except Exception:
                continue
        return "\n".join(texts)[:max_chars] if texts else "[DXF has no TEXT/MTEXT]"
    except Exception:
        return "[DXF parser not available]"
