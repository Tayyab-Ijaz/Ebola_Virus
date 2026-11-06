"""S2: nested CV (seed 42 only) with the limma-voom moderated t from R as the in-split ranker.
The R tables come from stage item7 of the SAME run and were computed on exactly the outer and inner training
splits used here; PrecomputedRanker refuses to answer for any other sample set."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd

from evd.nested import ModelCfg, run_nested_cv
from evd.runs import register, variant_stats
from evd.selection import PrecomputedRanker, welch_rank
from stages.item1 import _fold_table, _inner_counts

STAGE = "item1_s2"


def run(ctx):
    cfg, ds, log, y = ctx.cfg, ctx.ds, ctx.log, ctx.ds.y
    d7 = ctx.run_dir / "item7"
    if not (d7 / "splits_seed42.json").exists():
        raise RuntimeError("S2 needs stage item7 in the same run (limma-voom tables on the seed-42 splits)")
    sj = json.loads((d7 / "splits_seed42.json").read_text())
    tables, train = {}, {}
    for o in sj["outer"]:
        f = o["fold"]
        keys = [(f"outer{f}", o["train"])] + [(f"outer{f}_inner{s['j']}", s["train"]) for s in o["inner"]]
        for key, tr in keys:
            tables[key] = pd.read_csv(d7 / f"rank_limma_{key}.tsv", sep="\t", dtype={"GeneID": str})
            train[key] = tr
    ranker = PrecomputedRanker(tables, train, ds.gene_ids, cfg["selection"]["gate_fdr"], cfg["selection"]["gate_lfc"])
    for o in sj["outer"]:                         # the R filter must keep exactly the genes the Python filter keeps
        tr = np.array(o["train"])
        w = welch_rank(ds, tr, y[tr])
        same = set(ds.gene_ids[w.order]) == set(tables[f"outer{o['fold']}"].GeneID)
        log.info("outer fold %d: limma-voom gene universe equals the Welch universe: %s (%d genes)", o["fold"], same, w.n_kept)
        assert same
    mc = replace(ModelCfg.from_config(cfg), ranker=ranker)
    ref = run_nested_cv(ds, y, cfg["seeds"]["reference"], mc, n_jobs=ctx.n_jobs)
    run_ = {"ref": ref, "reps": None, "mc": mc}
    register(ctx, "S2_limma_voom", ref, None)
    _fold_table(ds, ref).to_csv(ctx.out(STAGE, "S2_limma_voom_seed42_folds.csv"), index=False)
    _inner_counts(ref, cfg["seeds"]["reference"]).to_csv(ctx.out(STAGE, "S2_limma_voom_seed42_inner_gene_counts.csv"), index=False)
    pd.DataFrame({"sample": ds.sample_ids, "y": y, "fold": ref["fold_id"], "oof": ref["oof"]}).to_csv(
        ctx.out(STAGE, "S2_limma_voom_seed42_oof.csv"), index=False)
    st = variant_stats(y, run_, cfg["item1"]["bootstrap_B"], cfg["seeds"]["bootstrap"])
    ctx.out(STAGE, "summary.json").write_text(json.dumps(st, indent=2), encoding="utf-8")
    s = st["seed42"]
    log.info("[S2 limma-voom] seed 42: pooled OOF AUC %.4f (CI-B DeLong %.3f-%.3f) | per-fold %s mean %.4f | K %s",
             s["pooled_oof_auc"], s["pooled_ci_B_delong"]["lo"], s["pooled_ci_B_delong"]["hi"],
             np.round(s["per_fold_auc"], 4).tolist(), s["per_fold_mean"], s["K_chosen"])
    for r in ref["folds"]:
        log.info("  fold %d: K=%d genes: %s", r["fold"], r["K"], ";".join(ds.symbols[r["sel"]][:12]))
