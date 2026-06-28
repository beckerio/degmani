# src/sad/config/paths.py

from pathlib import Path
import os

# Machine-specific candidates (fallback order)
CANDIDATES = [
    Path("/data/SSD/datasets/SAD_Datasets"),  # minga_new
    Path("/media/ste82041/8e0ea439-7dc8-4082-af83-332123df3cf4/datasets2/SAD_Datasets/"),
    Path("/mnt/15TB-NVME/ste82041/datasets"),# GS007
]

# Put all possible roots (across all machines) in priority order.
DEFAULT_CANDIDATES = [
    Path("/data/SSD/datasets/SAD_Datasets"),#minga
    Path("/media/ste82041/DATA-2TB/"),
    Path("/mnt/15TB-NVME/ste82041/"),
    # add the second folders per system here:
    Path("/media/ste82041/8e0ea439-7dc8-4082-af83-332123df3cf4/datasets2/"),
    Path("/mnt/Data-512GB/datasets_02"),
]

def _resolve_data_root() -> Path:
    env = os.getenv("DATA_ROOT")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
        raise RuntimeError(f"DATA_ROOT does not exist: {p}")
    for p in CANDIDATES:
        if p.exists():
            return p
    raise RuntimeError("No valid data root found. Set DATA_ROOT or update CANDIDATES.")

DATA_ROOT: Path = _resolve_data_root()

def data_path(*parts: str) -> Path:
    """Build a path under the data root."""
    return DATA_ROOT.joinpath(*parts)


def _parse_env_roots(var: str = "DATA_ROOTS") -> list[Path]:
    # Colon-separated on Unix, semicolon on Windows (os.pathsep handles both)
    val = os.getenv(var)
    if not val:
        return []
    return [Path(p).expanduser() for p in val.split(os.pathsep) if p]

def _existing(paths: list[Path]) -> list[Path]:
    seen = set()
    out: list[Path] = []
    for p in paths:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if rp in seen:
            continue
        if rp.exists():
            out.append(rp)
            seen.add(rp)
    return out

# Order: env override first, else defaults. Keep only those that exist.

DATA_ROOTS: list[Path] = _existing(_parse_env_roots() or DEFAULT_CANDIDATES)
if not DATA_ROOTS:
    raise RuntimeError(
        "No valid data roots found. Set DATA_ROOTS or update DEFAULT_CANDIDATES."
    )

PRIMARY_ROOT: Path = DATA_ROOTS[0]

def data_path(*parts: str, root: Path | None = None) -> Path:
    """Build a path under the chosen root (defaults to PRIMARY_ROOT)."""
    base = root or PRIMARY_ROOT
    return base.joinpath(*parts)

def find_existing(*parts: str) -> Path | None:
    """Return the first existing path across all roots (for reading)."""
    rel = Path(*parts)
    for r in DATA_ROOTS:
        cand = r / rel
        if cand.exists():
            return cand
    return None

def all_locations(*parts: str) -> list[Path]:
    """Return all existing locations of a relative path across roots."""
    rel = Path(*parts)
    return [r / rel for r in DATA_ROOTS if (r / rel).exists()]


DATASETS_ALIASES = ["datasets", "datasets2"]

def datasets_root(create_if_missing: bool = True) -> Path:
    """Locate the datasets folder across roots, trying alias names."""
    for r in DATA_ROOTS:
        for name in DATASETS_ALIASES:
            p = r / name
            if p.exists():
                return p
    # Default to primary/datasets
    p = PRIMARY_ROOT / "datasets"
    #if create_if_missing:
    #    p.mkdir(parents=True, exist_ok=True)
    return p