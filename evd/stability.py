"""Stability analysis (item 3): per-gene selection statistics over many fits, the pre-specified tie-break
ordering (decision D4) and the panel-recurrence measures (P5).

Tie-break, applied in this order (ANALYSIS_PLAN_FROZEN.md section 5):
  (1) higher selection frequency (fraction of fits in which the elastic-net coefficient is non-zero);
  (2) higher mean |standardised coefficient| across the fits in which the gene was selected (= non-zero);
  (3) lower median Welch-|t| rank across all fits (rank 201 when outside the stored top 200);
  (4) alphabetical symbol.
"""
import numpy as np
import pandas as pd

RANK_CENSOR = 201


def gene_stats(fits, ds, rank_default=RANK_CENSOR):
    """fits: dicts with `sel` (global gene columns, the top-K actually used), `coef` (standardised, aligned
    with `sel`) and `top_order` (ranking of the fit's training split). One row per gene ever selected."""
    n = len(fits)
    sel_ct, nz_ct, coefs = {}, {}, {}
    for f in fits:
        for g, c in zip(f["sel"], f["coef"]):
            sel_ct[g] = sel_ct.get(g, 0) + 1
            if c != 0:
                nz_ct[g] = nz_ct.get(g, 0) + 1
                coefs.setdefault(g, []).append(abs(float(c)))
    pos = [{int(g): i + 1 for i, g in enumerate(f["top_order"])} for f in fits]
    rows = []
    for g in sorted(sel_ct):
        ranks = [p.get(int(g), rank_default) for p in pos]
        rows.append({"col": int(g), "symbol": ds.symbols[g], "gene_id": ds.gene_ids[g],
                     "sex_chrom": bool(ds.sex_chrom[g]), "n_fits": n, "n_selected_topK": sel_ct[g],
                     "n_nonzero": nz_ct.get(g, 0), "freq_nonzero": nz_ct.get(g, 0) / n,
                     "freq_selected_topK": sel_ct[g] / n,
                     "mean_abs_std_coef": float(np.mean(coefs[g])) if g in coefs else 0.0,
                     "median_rank": float(np.median(ranks))})
    return pd.DataFrame(rows)


def tie_break_sort(df):
    d = df.assign(_f=df["freq_nonzero"].round(12), _c=df["mean_abs_std_coef"].round(12))
    d = d.sort_values(["_f", "_c", "median_rank", "symbol"], ascending=[False, False, True, True], kind="mergesort")
    return d.drop(columns=["_f", "_c"]).reset_index(drop=True)


def _jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 1.0


def recurrence(fits, ordered_cols, K):
    """How often does the final panel (first K genes of the ordering) recur in a set of fits?"""
    panel = set(map(int, ordered_cols[:K]))
    topk = [set(map(int, f["sel"])) for f in fits]
    nonz = [set(int(g) for g, c in zip(f["sel"], f["coef"]) if c != 0) for f in fits]
    out = {"n_fits": len(fits), "panel_size": K,
           "exact_equal_topK_set": float(np.mean([s == panel for s in topk])),
           "contains_panel_topK_set": float(np.mean([panel <= s for s in topk])),
           "contains_panel_nonzero_set": float(np.mean([panel <= s for s in nonz])),
           "exact_equal_nonzero_set": float(np.mean([s == panel for s in nonz])),
           "mean_jaccard_topK_set": float(np.mean([_jaccard(s, panel) for s in topk])),
           "mean_jaccard_nonzero_set": float(np.mean([_jaccard(s, panel) for s in nonz]))}
    prefixes = []
    for k in range(1, K + 1):
        pk = set(map(int, ordered_cols[:k]))
        prefixes.append({"k": k, "prefix_contained_in_nonzero_set": float(np.mean([pk <= s for s in nonz])),
                         "prefix_contained_in_topK_set": float(np.mean([pk <= s for s in topk])),
                         "topK_set_equals_prefix": float(np.mean([s == pk for s in topk]))})
    return out, pd.DataFrame(prefixes)
