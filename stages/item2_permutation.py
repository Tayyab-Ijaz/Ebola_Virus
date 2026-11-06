"""Item 2: permutation tests in which the COMPLETE pipeline (selection, inner tuning, fit) is rerun on every
permuted label vector. B from config (1000). Permutation i uses default_rng([42, i]).
Results are written in chunks of 50 permutations (chunks/<tag>_<first>.json, each carrying run id, commit hash and
config hash). A resume (same run id, same commit, same config, clean tree - enforced by RunContext) reuses only chunks
whose metadata match exactly; any mismatch is an error, never a silent mix. Each permutation's labels depend on the
permutation index alone (default_rng([42, i])), so the null does not depend on n_jobs or on the order of execution.
Nulls: primary (all 59 labels permuted), S1 hard gate (why the legacy null was degenerate), and the primary
pipeline with labels permuted within inferred-sex strata. Tested statistics: seed-42 pooled OOF AUC and
seed-42 mean-fold AUC, each against its own null; p = (1 + #{null >= observed}) / (1 + B)."""
import json
import os
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from evd.nested import ModelCfg
from evd.runs import get_variant, perm_once

STAGE = "item2_permutation"
CHUNK = 50


def _chunk_meta(ctx, tag, a, b):
    m = ctx.manifest
    return {"run_id": ctx.run_id, "git_commit": m["git"]["commit"], "config_sha256": ctx.cfg["_config_sha256"],
            "tag": tag, "first": a, "stop": b, "B": ctx.cfg["item2"]["B"]}


def _load_chunk(ctx, path, meta):
    rec = json.loads(path.read_text(encoding="utf-8"))
    bad = {k: (rec.get(k), v) for k, v in meta.items() if rec.get(k) != v}
    if bad:
        raise RuntimeError(f"chunk {path.name} does not belong to this run/commit/config: {bad}; start a new run id")
    return rec["rows"]


def _null(ctx, tag, mc, strata, obs):
    cfg, ds, log, y = ctx.cfg, ctx.ds, ctx.log, ctx.ds.y
    B, seed = cfg["item2"]["B"], cfg["seeds"]["reference"]
    cdir = ctx.stage_dir(STAGE) / "chunks"
    cdir.mkdir(exist_ok=True)
    rows, t0 = [], time.time()
    for a in range(0, B, CHUNK):
        b = min(B, a + CHUNK)
        meta = _chunk_meta(ctx, tag, a, b)
        path = cdir / f"{tag}_{a:04d}.json"
        if path.exists():
            rows += _load_chunk(ctx, path, meta)
            log.info("[%s] permutations %d-%d reused from chunk file", tag, a, b - 1)
            continue
        new = Parallel(n_jobs=ctx.n_jobs)(delayed(perm_once)(ds, y, i, mc, seed, strata) for i in range(a, b))
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({**meta, "rows": new}), encoding="utf-8")
        os.replace(tmp, path)
        rows += new
        log.info("[%s] %d/%d permutations (%.0fs)", tag, len(rows), B, time.time() - t0)
    df = pd.DataFrame(rows).sort_values("i")
    assert list(df.i) == list(range(B)), "permutation indices are not 0..B-1"
    df.to_csv(ctx.out(STAGE, f"{tag}_null.csv"), index=False)
    out = {"B": B, "strata": "inferred sex" if strata is not None else None,
           "frac_fits_with_empty_selection": float(df.empty_fits.sum() / (5 * B)),
           "frac_permutations_all_folds_empty": float((df.empty_fits == 5).mean())}
    for stat, ob in (("pooled_auc", obs["pooled_oof_auc"]), ("mean_fold_auc", obs["per_fold_mean"])):
        v = df[stat].to_numpy()
        out[stat] = {"observed": ob, "null_mean": float(v.mean()), "null_sd": float(v.std(ddof=1)),
                     "null_min": float(v.min()), "null_p2.5": float(np.percentile(v, 2.5)),
                     "null_median": float(np.median(v)), "null_p97.5": float(np.percentile(v, 97.5)),
                     "null_p99": float(np.percentile(v, 99)), "null_max": float(v.max()),
                     "n_distinct_values": int(len(np.unique(np.round(v, 12)))),
                     "frac_exactly_0.5": float((np.abs(v - 0.5) < 1e-12).mean()),
                     "n_null_ge_observed": int((v >= ob).sum()), "p_value": float((1 + (v >= ob).sum()) / (1 + B))}
    log.info("[%s] null pooled AUC: mean %.3f SD %.3f [%.3f, %.3f], %d distinct values, %.1f%% exactly 0.5 | observed %.4f, "
             "p = %.4f | mean-fold null mean %.3f, observed %.4f, p = %.4f | empty selections %.1f%% of fits",
             tag, out["pooled_auc"]["null_mean"], out["pooled_auc"]["null_sd"], out["pooled_auc"]["null_min"],
             out["pooled_auc"]["null_max"], out["pooled_auc"]["n_distinct_values"],
             100 * out["pooled_auc"]["frac_exactly_0.5"], obs["pooled_oof_auc"], out["pooled_auc"]["p_value"],
             out["mean_fold_auc"]["null_mean"], obs["per_fold_mean"], out["mean_fold_auc"]["p_value"],
             100 * out["frac_fits_with_empty_selection"])
    return out


def run(ctx):
    cfg = ctx.cfg
    base = ModelCfg.from_config(cfg)
    specs = [("primary", base, None), ("S1_hard_gate", replace(base, mode="gate"), None),
             ("primary_sex_stratified", base, ctx.ds.sex_male)]
    res = {}
    for tag, mc, strata in specs:
        obs_tag = "primary" if tag.startswith("primary") else tag
        run_ = get_variant(ctx, obs_tag, mc if strata is None else base, with_repeats=False)
        ref = run_["ref"]
        obs = {"pooled_oof_auc": ref["pooled_auc"], "per_fold_mean": ref["mean_fold_auc"]}
        res[tag] = _null(ctx, tag, mc, strata, obs)
    ctx.out(STAGE, "permutation_summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


run.resumable = True
