from pathlib import Path
import os

# Machine-specific candidates (fallback order)
CANDIDATES = [
    Path("/data/SSD/checkpoints/"),#minga
    Path("/media/ste82041/8e0ea439-7dc8-4082-af83-332123df3cf4/checkpoints/"),#kalle
    Path("/mnt/15TB-NVME/ste82041/checkpoints"), # GS007
]

def _resolve_checkpoint_root() -> Path:
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

CHECKPOINT_ROOT: Path = _resolve_checkpoint_root()

def checkpoint_path(*parts: str) -> Path:
    """Build a path under the data root."""
    return CHECKPOINT_ROOT.joinpath(*parts)