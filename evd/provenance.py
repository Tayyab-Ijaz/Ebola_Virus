"""One run = one directory, one log, one config, one manifest. Outputs cannot be overwritten
and every stage records the hash of every file it wrote, so a run can be verified later and
outputs from different runs cannot be silently mixed."""
import datetime as dt
import importlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .config import REPO_ROOT, sha256_file

PACKAGES = ["numpy", "scipy", "pandas", "sklearn", "statsmodels", "matplotlib", "joblib", "shap"]


def _git(*args):
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"unavailable: {exc}"


def package_versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count()}
    for p in PACKAGES:
        try:
            out[p] = importlib.import_module(p).__version__
        except Exception:
            out[p] = None
    return out


def r_versions(cfg):
    code = ("cat(R.version.string, '\\n'); for (p in c('DESeq2','limma','edgeR','jsonlite','immunedeconv','EPIC','MCPcounter')) "
            "cat(p, as.character(tryCatch(packageVersion(p), error=function(e) NA)), '\\n')")
    try:
        r = subprocess.run([cfg["rscript"], "-e", code], capture_output=True, text=True, timeout=120)
        return [ln for ln in r.stdout.splitlines() if ln.strip()]
    except Exception as exc:
        return [f"unavailable: {exc}"]


class RunContext:
    def __init__(self, cfg, run_id=None, stages=None, resume=False):
        self.cfg = cfg
        self.run_id = run_id or dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_dir = REPO_ROOT / cfg["paths"]["results_root"] / self.run_id
        self.resuming = False
        if self.run_dir.exists():
            if not resume:
                raise FileExistsError(f"{self.run_dir} already exists; a run directory is never reused "
                                      "(--resume continues a run only under the same run id and commit)")
            self._check_resumable(cfg)
            self.resuming = True
        elif resume:
            raise FileNotFoundError(f"--resume given but {self.run_dir} does not exist")
        else:
            self.run_dir.mkdir(parents=True)
        self.n_jobs = int(cfg["n_jobs"])
        self._ds = None
        self.cache = {}                  # in-run results shared between stages (never persisted by itself)
        self.r_warnings = []             # every warning printed by the R scripts (options(warn = 1))
        self.stage_times = {}
        self._setup_logging()
        status = _git("status", "--porcelain")
        if self.resuming:
            self.manifest = json.loads((self.run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.manifest.setdefault("resumes", []).append(
                {"at": dt.datetime.now().isoformat(timespec="seconds"), "commit": _git("rev-parse", "HEAD"),
                 "config_sha256": cfg["_config_sha256"]})
            self.manifest.pop("finished", None)
            self.stage_times = dict(self.manifest.get("stage_seconds", {}))
            self.log.warning("RESUMING run %s (same run id, commit %s, config %s)", self.run_id,
                             self.manifest["git"]["commit"][:10], cfg["_config_sha256"][:10])
            self._write_manifest()
            return
        (self.run_dir / "config.used.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        self.manifest = {
            "run_id": self.run_id,
            "started": dt.datetime.now().isoformat(timespec="seconds"),
            "command": sys.argv,
            "stages_requested": stages,
            "config_sha256": cfg["_config_sha256"],
            "frozen_plan_sha256": sha256_file(cfg.path("frozen_plan")),
            "git": {"branch": _git("rev-parse", "--abbrev-ref", "HEAD"), "commit": _git("rev-parse", "HEAD"),
                    "tag": _git("describe", "--tags", "--exact-match"),
                    "dirty": bool(status), "dirty_files": status.splitlines()[:50]},
            "packages": package_versions(),
            "inputs_sha256": {k: sha256_file(cfg.path(k)) for k in
                              ("counts", "annotation", "metadata", "stored_deseq2_all", "stored_vst",
                               "authors_cpm", "series_matrix") if k in cfg["paths"] and cfg.path(k).exists()},
            "stage_status": {},
        }
        if self.manifest["git"]["dirty"]:
            self.log.warning("working tree is dirty; the manifest records the dirty files")
        self._write_manifest()

    def _check_resumable(self, cfg):
        """A run may be continued only under the same run id, the same commit and the same config, from a clean tree."""
        mp = self.run_dir / "manifest.json"
        if not mp.exists():
            raise RuntimeError(f"cannot resume {self.run_id}: no manifest.json")
        old = json.loads(mp.read_text(encoding="utf-8"))
        now = {"commit": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain"))}
        problems = []
        if old["run_id"] != self.run_id:
            problems.append("run id differs")
        if old["git"]["commit"] != now["commit"]:
            problems.append(f"commit differs ({old['git']['commit'][:10]} vs {now['commit'][:10]})")
        if old["config_sha256"] != cfg["_config_sha256"]:
            problems.append("config hash differs")
        if old["git"]["dirty"] or now["dirty"]:
            problems.append("working tree is dirty (now or at the start of the run)")
        if problems:
            raise RuntimeError(f"refusing to resume {self.run_id}: " + "; ".join(problems) + ". Start a new run id from scratch.")

    # ---- logging -----------------------------------------------------------------------------
    def _setup_logging(self):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
        self.log = logging.getLogger("evd.run")
        self.log.setLevel(logging.INFO)
        self.log.handlers.clear()
        fmt = logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%H:%M:%S")
        fh = logging.FileHandler(self.run_dir / "run.log", encoding="utf-8")
        sh = logging.StreamHandler(sys.stdout)
        for h in (fh, sh):
            h.setFormatter(fmt)
            self.log.addHandler(h)

    # ---- data --------------------------------------------------------------------------------
    @property
    def ds(self):
        if self._ds is None:
            from .data import infer_sex, load_dataset
            self._ds = load_dataset(self.cfg)
            sx = infer_sex(self._ds, self.cfg)                      # label-free (expression of Y genes only)
            self._ds.sex_male = (sx["sex"] == "male").to_numpy(int)
            self._ds.cov = self._ds.sex_male[:, None].astype(float)
            n1 = int(self._ds.y.sum())
            self.log.info("dataset: %d samples x %d genes (raw file %d genes); Healthy %d, EVD-S %d",
                          self._ds.n, self._ds.G, self._ds.n_raw_genes, self._ds.n - n1, n1)
        return self._ds

    # ---- outputs -----------------------------------------------------------------------------
    def stage_dir(self, stage):
        d = self.run_dir / stage
        d.mkdir(exist_ok=True)
        return d

    def out(self, stage, name):
        """Path for a new output file. Refuses to overwrite."""
        p = self.stage_dir(stage) / name
        if p.exists():
            raise FileExistsError(f"refusing to overwrite {p}")
        return p

    def inputs_dir(self):
        """Shared, gene-filtered input tables for the R scripts (written once per run)."""
        d = self.run_dir / "inputs"
        if d.exists():
            return d
        import pandas as pd
        d.mkdir()
        ds = self.ds
        from .data import infer_sex
        sex = infer_sex(ds, self.cfg)
        sex.to_csv(d / "samples.tsv", sep="\t", index=False)
        pd.DataFrame(ds.counts.T, index=ds.gene_ids, columns=ds.sample_ids).rename_axis("GeneID").to_csv(
            d / "counts_gene_filtered.tsv", sep="\t")
        return d

    # ---- R -----------------------------------------------------------------------------------
    def run_r(self, script, *args):
        cmd = [self.cfg["rscript"], str(Path(__file__).resolve().parent.parent / "r" / script), *map(str, args)]
        self.log.info("R: %s", " ".join(cmd[1:]))
        t = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        lines = (r.stdout + r.stderr).splitlines()
        for i, ln in enumerate(lines):
            if ln.strip():
                self.log.info("  [R] %s", ln)
            if ln.startswith("Warning"):             # options(warn = 1) prints each warning with its call
                msg, k = ln.strip(), i + 1
                while msg.endswith(":") and k < len(lines):      # "Warning in f(x) :" -> message on the next line(s)
                    msg += " " + lines[k].strip()
                    k += 1
                self.r_warnings.append(f"{script}: {msg}")
        if r.returncode != 0:
            raise RuntimeError(f"R script {script} failed (exit {r.returncode})")
        self.log.info("R script %s finished in %.0fs", script, time.time() - t)

    # ---- bookkeeping -------------------------------------------------------------------------
    def _write_manifest(self):
        (self.run_dir / "manifest.json").write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")

    def run_stage(self, name, fn):
        self.log.info("=" * 70)
        self.log.info("STAGE %s", name)
        self.log.info("=" * 70)
        if self.resuming:
            self._reset_stage_for_resume(name, fn)
        self.manifest["stage_status"][name] = "running"
        self._write_manifest()
        t = time.time()
        try:
            fn(self)
        except Exception:
            self.manifest["stage_status"][name] = "FAILED"
            self._write_manifest()
            self.log.exception("stage %s failed", name)
            raise
        self.finish_stage(name, time.time() - t)

    def _reset_stage_for_resume(self, name, fn):
        """On resume every stage is recomputed (the in-run cache that later stages rely on is not persisted), within
        the same run id and commit. Only a stage that declares `resumable` keeps its checksummed `chunks/` folder."""
        d = self.run_dir / name
        (self.run_dir / f"{name}.STAGE_DONE.json").unlink(missing_ok=True)
        if not d.exists():
            return
        keep = d / "chunks" if getattr(fn, "resumable", False) else None
        for p in list(d.iterdir()):
            if keep is not None and p == keep:
                continue
            shutil.rmtree(p) if p.is_dir() else p.unlink()
        self.log.info("resume: stage %s recomputed; %s", name,
                      "chunks kept" if keep is not None and keep.exists() else "previous partial output discarded")

    def finish_stage(self, name, seconds):
        d = self.run_dir / name
        files = {}
        if d.exists():
            for p in sorted(d.rglob("*")):
                if p.is_file():
                    files[str(p.relative_to(self.run_dir)).replace("\\", "/")] = sha256_file(p)
        rec = {"stage": name, "seconds": round(seconds, 1), "config_sha256": self.cfg["_config_sha256"],
               "git_commit": self.manifest["git"]["commit"], "files": files}
        (self.run_dir / f"{name}.STAGE_DONE.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        self.manifest["stage_status"][name] = "done"
        self.stage_times[name] = round(seconds, 1)
        self.manifest["stage_seconds"] = self.stage_times
        self._write_manifest()
        self.log.info("stage %s done in %.0fs (%d files hashed)", name, seconds, len(files))

    def close(self):
        (self.run_dir / "r_warnings.txt").write_text("\n".join(self.r_warnings) + "\n", encoding="utf-8")
        self.manifest["finished"] = dt.datetime.now().isoformat(timespec="seconds")
        self._write_manifest()


def verify_run(run_dir):
    """Re-hash every output of a run and check that all stages share one config hash and commit."""
    run_dir = Path(run_dir)
    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    problems, seen = [], set()
    cfgs, commits = set(), set()
    for rec_path in sorted(run_dir.glob("*.STAGE_DONE.json")):
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        cfgs.add(rec["config_sha256"])
        commits.add(rec["git_commit"])
        for rel, h in rec["files"].items():
            seen.add(rel)
            p = run_dir / rel
            if not p.exists():
                problems.append(f"missing {rel}")
            elif sha256_file(p) != h:
                problems.append(f"modified {rel}")
    for p in run_dir.rglob("*"):
        if p.is_file():
            rel = str(p.relative_to(run_dir)).replace("\\", "/")
            top = rel.split("/")[0]
            if rel in seen or top == "inputs" or "/" not in rel:
                continue
            problems.append(f"unrecorded file {rel}")
    if len(cfgs) > 1:
        problems.append(f"stages used different configs: {cfgs}")
    if len(commits) > 1:
        problems.append(f"stages ran on different commits: {commits}")
    if cfgs and man["config_sha256"] not in cfgs:
        problems.append("config hash in manifest differs from stage records")
    return problems
