"""Config loading. Paths in config.json are relative to the repository root."""
import hashlib
import json
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
V2_DIR = PKG_DIR.parent                 # revision_v2/
REPO_ROOT = V2_DIR.parent               # evd-biomarker-pipeline/


class Config(dict):
    """dict with path resolution and a content hash."""

    def path(self, key):
        return (REPO_ROOT / self["paths"][key]).resolve()

    @property
    def repeat_seeds(self):
        s = self["seeds"]
        return list(range(s["repeat_start"], s["repeat_start"] + s["n_repeats"]))


def load_config(path=None):
    path = Path(path) if path else V2_DIR / "config.json"
    raw = path.read_bytes()
    cfg = Config(json.loads(raw))
    cfg["_config_path"] = str(path)
    cfg["_config_sha256"] = hashlib.sha256(raw).hexdigest()
    return cfg


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
