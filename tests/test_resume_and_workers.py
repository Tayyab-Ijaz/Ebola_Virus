"""Permutation results must not depend on the number of workers or their order, and a resume must never mix
chunks or stages across run ids, commits or configs."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from joblib import Parallel, delayed

from evd import provenance
from evd.nested import ModelCfg
from evd.provenance import RunContext
from evd.runs import perm_once
from stages import item2_permutation as item2


@pytest.fixture
def tmp_path():
    """Own temp folder: pytest's tmp_path needs <temp>/pytest-of-<user>, which is not writable on the analysis machine; the run log of a test RunContext stays open on Windows."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        yield Path(d)


def _meta(commit="c1", cfg_hash="h1", tag="primary", a=0, b=50, run_id="r1"):
    return {"run_id": run_id, "git_commit": commit, "config_sha256": cfg_hash, "tag": tag, "first": a, "stop": b, "B": 1000}


def test_first_ten_permutations_identical_for_one_and_six_workers(cfg, real):
    mc = ModelCfg.from_config(cfg)
    seed = cfg["seeds"]["reference"]
    one = [perm_once(real, real.y, i, mc, seed, None) for i in range(10)]
    six = Parallel(n_jobs=6)(delayed(perm_once)(real, real.y, i, mc, seed, None) for i in reversed(range(10)))
    six = sorted(six, key=lambda r: r["i"])
    assert one == six                                   # exact equality: AUCs, empty fits and chosen K


def test_chunk_from_another_commit_config_or_run_is_refused(tmp_path):
    p = tmp_path / "primary_0000.json"
    p.write_text(json.dumps({**_meta(), "rows": [{"i": 0}]}), encoding="utf-8")
    assert item2._load_chunk(None, p, _meta()) == [{"i": 0}]
    for bad in (_meta(commit="c2"), _meta(cfg_hash="h2"), _meta(run_id="r2"), _meta(tag="S1_hard_gate"), _meta(a=50, b=100)):
        with pytest.raises(RuntimeError, match="does not belong"):
            item2._load_chunk(None, p, bad)


def test_resume_only_with_same_run_id_commit_config_and_clean_tree(cfg, tmp_path, monkeypatch):
    cfg = type(cfg)(cfg)
    cfg["paths"] = {**cfg["paths"], "results_root": str(tmp_path)}
    state = {"commit": "aaa", "status": ""}
    monkeypatch.setattr(provenance, "_git", lambda *a: state["commit"] if a[:1] == ("rev-parse",) and "HEAD" in a
                        else state["status"] if a[:1] == ("status",) else "x")
    RunContext(cfg, "r1", ["item2_permutation"]).close()
    with pytest.raises(FileExistsError):
        RunContext(cfg, "r1", ["item2_permutation"])                       # never reused without --resume
    with pytest.raises(FileNotFoundError):
        RunContext(cfg, "nope", [], resume=True)
    assert RunContext(cfg, "r1", ["item2_permutation"], resume=True).resuming
    state["commit"] = "bbb"
    with pytest.raises(RuntimeError, match="commit differs"):
        RunContext(cfg, "r1", [], resume=True)
    state["commit"], state["status"] = "aaa", " M file.py"
    with pytest.raises(RuntimeError, match="dirty"):
        RunContext(cfg, "r1", [], resume=True)
    state["status"] = ""
    other = type(cfg)(cfg)
    other["_config_sha256"] = "different"
    with pytest.raises(RuntimeError, match="config hash differs"):
        RunContext(other, "r1", [], resume=True)
