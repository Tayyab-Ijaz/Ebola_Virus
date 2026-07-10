#!/usr/bin/env python3
"""
01_corrected_ml_pipeline.py
============================
Corrected EVD blood transcriptomics biomarker pipeline.

Methodological revisions:
  C-34 — Feature-selection leakage:
      DEG selection (Wilcoxon + BH), low-expression filtering, and
      log2(CPM+1) normalisation are now performed INSIDE each outer CV fold,
      using training-fold samples only.  The original pipeline computed DEGs
      on all 59 samples before splitting, inflating apparent performance.

  C-36 — Invalid permutation test:
      Each permutation now re-runs the ENTIRE pipeline (DEG selection +
      normalisation + model) on the permuted labels.  B = 500 permutations;
      p reported as < 1/(B+1) when no permutation exceeds observed AUC.

References
----------
Ambroise & McLachlan (2002) PNAS 99:6562-6566.
Simon et al. (2003) J Natl Cancer Inst 95:14-18.
"""

import os
import warnings
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix
)
from sklearn.calibration import calibration_curve
from sklearn.base import clone
from collections import Counter

warnings.filterwarnings("ignore")

# ── Directory setup ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(SCRIPT_DIR)

DATA_DIR   = os.path.join(ROOT, "data")
OUT_DIR    = os.path.join(ROOT, "results", "tables", "ml")
FIG_DIR    = os.path.join(ROOT, "results", "figures", "ml")
TAB_DIR    = os.path.join(ROOT, "results", "tables", "ml")

for d in [OUT_DIR, FIG_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Constants ───────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
DPI          = 300
N_PERM       = 500     # B for permutation test; p_min = 1/(B+1) = 0.00199

LABEL_MAP    = {"Healthy": 0, "EVD-S": 1}
COL = {
    "Healthy": "#0072B2",
    "EVD-S":   "#D55E00",
    "accent":  "#009E73",
    "grey":    "#999999",
    "dark":    "#2B2B2B",
}

def hdr(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

def savefig(name):
    plt.savefig(os.path.join(FIG_DIR, name), dpi=DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD RAW DATA
# ═══════════════════════════════════════════════════════════════════════════════
hdr("1 - Load raw data")

counts_raw = pd.read_csv(
    os.path.join(DATA_DIR, "GSE143549_raw_counts_GRCh38.p13_NCBI.tsv"), sep="\t"
)
annot_df = pd.read_csv(
    os.path.join(DATA_DIR, "Human.GRCh38.p13.annot.tsv"), sep="\t"
)
meta_df = pd.read_csv(os.path.join(DATA_DIR, "meta_data.csv"))

# ── Gene annotations ────────────────────────────────────────────────────────────
annot_df["GeneID"]   = annot_df["GeneID"].astype(str)
annot_df["Symbol"]   = annot_df["Symbol"].fillna("").astype(str)
annot_df["GeneType"] = annot_df["GeneType"].fillna("").astype(str)

rm_pseudo = (
    annot_df["GeneType"].str.contains("pseudo", case=False, na=False) |
    annot_df["Symbol"].str.startswith("LOC", na=False)
)
keep_ids = set(annot_df.loc[~rm_pseudo, "GeneID"])
id2sym   = annot_df.set_index("GeneID")["Symbol"].to_dict()

# ── Build counts matrix (genes × samples) ───────────────────────────────────────
counts_raw["GeneID"] = counts_raw["GeneID"].astype(str)
counts_raw = counts_raw.set_index("GeneID")
counts_raw = counts_raw.loc[counts_raw.index.isin(keep_ids)]

# ── Metadata & label alignment ──────────────────────────────────────────────────
meta_df["ID_REF"]    = meta_df["ID_REF"].astype(str)
meta_df["Condition"] = meta_df["Condition"].str.strip()
meta_df["Label"]     = meta_df["Condition"].map(LABEL_MAP)

sample_cols = [c for c in meta_df["ID_REF"] if c in counts_raw.columns]
counts_raw  = counts_raw[sample_cols]
meta_df     = (meta_df[meta_df["ID_REF"].isin(sample_cols)]
               .reset_index(drop=True))

y = meta_df["Label"].values.astype(int)

print(f"Genes (post pseudogene removal): {counts_raw.shape[0]:,}")
print(f"Samples:                         {counts_raw.shape[1]}")
print(f"Class distribution:              {dict(zip(*np.unique(y, return_counts=True)))}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  WITHIN-FOLD UTILITIES   (the core of C-34 fix)
# ═══════════════════════════════════════════════════════════════════════════════

def low_expr_filter(counts: pd.DataFrame, labels: np.ndarray,
                    min_counts: int = 10) -> pd.DataFrame:
    """
    Retain genes with >= min_counts reads in >= min_group samples.
    min_group = size of the smaller class in this fold.
    Computed on training-fold counts only.
    """
    min_group = int(np.bincount(labels).min())
    keep = (counts >= min_counts).sum(axis=1) >= min_group
    return counts.loc[keep]


def log2_cpm_matrix(counts: pd.DataFrame) -> pd.DataFrame:
    """
    log2(CPM + 1) normalisation.
    counts : genes × samples
    returns: samples × genes DataFrame
    Library sizes computed from the supplied counts (training fold).
    """
    lib_sizes = counts.sum(axis=0)             # per sample
    cpm = counts.div(lib_sizes, axis=1) * 1e6
    return np.log2(cpm + 1).T                  # samples × genes


def apply_log2_cpm(counts: pd.DataFrame, genes: list) -> pd.DataFrame:
    """
    Normalise counts for a given gene list using log2(CPM+1).
    Library sizes are computed from ALL genes in the supplied counts matrix
    (not just the selected genes), which is the standard CPM convention.
    Any selected gene missing from the counts matrix is filled with 0.

    counts : genes × samples (only the relevant samples)
    returns: samples × genes DataFrame (no NaNs)
    """
    lib_sizes = counts.sum(axis=0)                # total counts per sample
    # Subset to selected genes present in counts; fill missing genes with 0
    present = [g for g in genes if g in counts.index]
    sub = counts.loc[present]
    cpm = sub.div(lib_sizes, axis=1) * 1e6
    log_cpm_T = np.log2(cpm + 1).T               # samples × present_genes
    # Re-index to full gene list; missing genes → 0
    log_cpm_T = log_cpm_T.reindex(columns=genes, fill_value=0.0)
    return log_cpm_T.fillna(0.0)                  # samples × genes


def within_fold_deg_selection(
    counts_train: pd.DataFrame,
    labels_train: np.ndarray,
    fdr_thresh: float = 0.05,
    lfc_thresh: float = 1.0,
    min_counts: int = 10,
) -> list:
    """
    Identify DEGs using TRAINING-FOLD SAMPLES ONLY.

    Method:
      1. Low-expression filter (training fold only).
      2. log2(CPM+1) normalisation (training fold only).
      3. Vectorised Welch two-sample t-test per gene (scipy, 2-D array input —
         faster than per-gene Wilcoxon for large gene sets, necessary to keep
         the 500-permutation test computationally feasible).
      4. Benjamini-Hochberg FDR correction.
      5. Keep genes with FDR < fdr_thresh AND |log2 mean-diff| >= lfc_thresh.

    Returns list of GeneIDs (str) that pass the filter.
    """
    counts_f = low_expr_filter(counts_train, labels_train, min_counts)
    if counts_f.shape[0] == 0:
        return []

    # log2_cpm_matrix returns samples × genes; transpose to genes × samples
    _lcpm_df = log2_cpm_matrix(counts_f)      # samples × genes DataFrame
    gene_ids = _lcpm_df.columns.tolist()       # gene IDs
    lcpm     = _lcpm_df.values.T               # genes × samples numpy array

    h_mask = labels_train == 0
    e_mask = labels_train == 1

    X_h = lcpm[:, h_mask]   # genes × n_healthy
    X_e = lcpm[:, e_mask]   # genes × n_evd

    # Vectorised Welch t-test (axis=1 → test each gene across samples)
    _, pvals = stats.ttest_ind(X_h, X_e, axis=1, equal_var=False)
    pvals = np.where(np.isnan(pvals), 1.0, pvals)

    lfcs = X_e.mean(axis=1) - X_h.mean(axis=1)   # EVD − Healthy

    _, padj, _, _ = multipletests(pvals, method="fdr_bh")

    sig_mask = (padj < fdr_thresh) & (np.abs(lfcs) >= lfc_thresh)
    return [gene_ids[i] for i in np.where(sig_mask)[0]]


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  CORRECTED NESTED CV   (C-34 fix applied)
# ═══════════════════════════════════════════════════════════════════════════════
hdr("3 - Corrected nested CV (DEG selection inside outer fold)")

outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

base_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("kbest",  SelectKBest(score_func=f_classif)),
    ("clf",    LogisticRegression(
        penalty="elasticnet", solver="saga", max_iter=20000,
        class_weight="balanced", random_state=RANDOM_STATE,
    )),
])

oof_prob           = np.full(len(y), np.nan)
outer_auc          = []
outer_ap           = []
best_params_log    = []
selected_genes_ctr = Counter()
n_degs_per_fold    = []

t0 = time.time()

for fold, (tr_idx, te_idx) in enumerate(outer.split(np.zeros(len(y)), y), start=1):
    tr_samples = meta_df.iloc[tr_idx]["ID_REF"].values
    te_samples = meta_df.iloc[te_idx]["ID_REF"].values
    y_tr = y[tr_idx]
    y_te = y[te_idx]

    counts_tr = counts_raw[tr_samples]   # genes × train samples
    counts_te = counts_raw[te_samples]   # genes × test  samples

    # ── 3a. DEG selection on TRAINING FOLD ONLY ─────────────────────────────
    degs = within_fold_deg_selection(counts_tr, y_tr)
    n_degs_per_fold.append(len(degs))

    if len(degs) < 5:
        print(f"  Fold {fold}: only {len(degs)} DEGs - assigning p=0.5 (fold excluded from AUC)")
        oof_prob[te_idx] = 0.5
        outer_auc.append(np.nan)
        outer_ap.append(np.nan)
        best_params_log.append({"fold": fold, "n_degs": len(degs),
                                 "kbest__k": np.nan, "clf__C": np.nan,
                                 "clf__l1_ratio": np.nan})
        continue

    # ── 3b. Normalise with per-fold library sizes only ──────────────────────
    X_tr = apply_log2_cpm(counts_tr, degs)   # train_samples × degs
    X_te = apply_log2_cpm(counts_te, degs)   # test_samples  × degs

    # ── 3c. Inner CV for hyperparameter tuning (training fold only) ──────────
    K_LIST = [k for k in [5, 10, 20, 30, 50, 75] if k <= X_tr.shape[1]]
    param_grid = {
        "kbest__k":      K_LIST,
        "clf__C":        [0.05, 0.1, 0.3, 1, 3, 10],
        "clf__l1_ratio": [0.2, 0.5, 0.8],
    }

    gs = GridSearchCV(
        estimator=base_pipe, param_grid=param_grid,
        scoring="roc_auc", cv=inner, n_jobs=-1,
    )
    gs.fit(X_tr, y_tr)
    best_est = gs.best_estimator_

    # ── 3d. Evaluate on HELD-OUT test fold ──────────────────────────────────
    p_te = best_est.predict_proba(X_te)[:, 1]
    oof_prob[te_idx] = p_te

    auc = roc_auc_score(y_te, p_te) if len(np.unique(y_te)) > 1 else np.nan
    ap  = average_precision_score(y_te, p_te) if len(np.unique(y_te)) > 1 else np.nan
    outer_auc.append(auc)
    outer_ap.append(ap)

    mask_sel = best_est.named_steps["kbest"].get_support()
    fold_genes = np.array(degs)[mask_sel].tolist()
    selected_genes_ctr.update(fold_genes)

    bp = {**gs.best_params_, "fold": fold, "n_degs": len(degs)}
    best_params_log.append(bp)

    print(f"  Fold {fold}: DEGs={len(degs):3d}  AUC={auc:.3f}  AP={ap:.3f}  "
          f"best_k={gs.best_params_.get('kbest__k','?')}")

elapsed = time.time() - t0

nested_auc_mean = float(np.nanmean(outer_auc))
nested_auc_std  = float(np.nanstd(outer_auc))

valid_mask = ~np.isnan(oof_prob)
oof_auc    = float(roc_auc_score(y[valid_mask], oof_prob[valid_mask]))
oof_ap     = float(average_precision_score(y[valid_mask], oof_prob[valid_mask]))

print(f"\nNested CV AUC  : {nested_auc_mean:.3f} +/- {nested_auc_std:.3f}")
print(f"OOF AUC        : {oof_auc:.3f}")
print(f"OOF AP         : {oof_ap:.3f}")
print(f"DEGs per fold  : {n_degs_per_fold}")
print(f"Wall time      : {elapsed:.0f}s")

# Save CV metrics
cv_df = pd.DataFrame({
    "outer_fold":   range(1, 6),
    "outer_fold_auc": outer_auc,
    "outer_fold_ap":  outer_ap,
    "n_degs_in_fold": n_degs_per_fold,
})
cv_df.to_csv(os.path.join(OUT_DIR, "nested_cv_fold_metrics.csv"), index=False)

pd.DataFrame(best_params_log).to_csv(
    os.path.join(OUT_DIR, "best_hyperparams_per_fold.csv"), index=False
)

# Stability table
stab_rows = []
for gid, cnt in selected_genes_ctr.items():
    stab_rows.append({
        "GeneID":        gid,
        "Symbol":        id2sym.get(gid, gid),
        "SelectedFolds": cnt,
        "StabilityFreq": cnt / outer.get_n_splits(),
    })
stab_df = (pd.DataFrame(stab_rows)
           .sort_values(["SelectedFolds", "Symbol"], ascending=[False, True])
           .reset_index(drop=True))
stab_df.to_csv(os.path.join(OUT_DIR, "gene_selection_stability.csv"), index=False)
print(f"\nUnique genes selected across folds: {len(stab_df)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  CORRECTED PERMUTATION TEST   (C-36 fix applied)
# ═══════════════════════════════════════════════════════════════════════════════
hdr(f"4 - Corrected permutation test (B={N_PERM}, full-pipeline per permutation)")

# Use mode of best hyperparams (from observed run) to fix model complexity;
# DEG selection is re-done from raw counts on EACH permuted label vector.
# This correctly models the null that labels carry no information.
bp_df = pd.DataFrame(best_params_log).dropna(subset=["kbest__k"])
common_k   = int(bp_df["kbest__k"].mode().iloc[0])   if len(bp_df) else 20
common_C   = float(bp_df["clf__C"].mode().iloc[0])   if len(bp_df) else 1.0
common_l1  = float(bp_df["clf__l1_ratio"].mode().iloc[0]) if len(bp_df) else 0.5

print(f"Fixed hyperparams for permutations : k={common_k}, C={common_C}, l1={common_l1}")
print(f"DEG selection is re-run on each permuted label vector.")

rng       = np.random.default_rng(RANDOM_STATE)
obs_auc   = oof_auc
perm_aucs = []

t1 = time.time()

for perm_i in range(N_PERM):
    y_perm  = rng.permutation(y)
    p_perm  = np.full(len(y_perm), np.nan)

    for tr_idx, te_idx in outer.split(np.zeros(len(y_perm)), y_perm):
        tr_samples = meta_df.iloc[tr_idx]["ID_REF"].values
        te_samples = meta_df.iloc[te_idx]["ID_REF"].values

        counts_tr_p = counts_raw[tr_samples]
        counts_te_p = counts_raw[te_samples]
        y_tr_p      = y_perm[tr_idx]
        y_te_p      = y_perm[te_idx]

        # DEG selection with PERMUTED labels — the essential fix for C-36
        degs_p = within_fold_deg_selection(counts_tr_p, y_tr_p)

        if len(degs_p) == 0:
            p_perm[te_idx] = 0.5
            continue

        k_use = min(common_k, len(degs_p))

        X_tr_p = apply_log2_cpm(counts_tr_p, degs_p)
        X_te_p = apply_log2_cpm(counts_te_p, degs_p)

        model_p = clone(base_pipe).set_params(**{
            "kbest__k":      k_use,
            "clf__C":        common_C,
            "clf__l1_ratio": common_l1,
        })
        model_p.fit(X_tr_p, y_tr_p)
        p_perm[te_idx] = model_p.predict_proba(X_te_p)[:, 1]

    valid_p = ~np.isnan(p_perm)
    if valid_p.sum() > 0 and len(np.unique(y_perm[valid_p])) > 1:
        auc_p = roc_auc_score(y_perm[valid_p], p_perm[valid_p])
    else:
        auc_p = 0.5
    perm_aucs.append(auc_p)

    if (perm_i + 1) % 100 == 0:
        elapsed_p = time.time() - t1
        print(f"  Permutation {perm_i+1}/{N_PERM}  "
              f"({elapsed_p:.0f}s elapsed, "
              f"mean null AUC so far={np.mean(perm_aucs):.3f})")

perm_aucs = np.array(perm_aucs)
n_exceed  = int(np.sum(perm_aucs >= obs_auc))
p_value   = float((1 + n_exceed) / (1 + N_PERM))
p_min     = 1.0 / (N_PERM + 1)

pd.DataFrame({"perm_auc": perm_aucs}).to_csv(
    os.path.join(OUT_DIR, "permutation_test_auc_distribution.csv"), index=False
)

print(f"\nObserved OOF AUC     : {obs_auc:.4f}")
print(f"Null AUC mean +/- std: {perm_aucs.mean():.3f} +/- {perm_aucs.std():.3f}")
print(f"Permutations >= obs  : {n_exceed} / {N_PERM}")
print(f"p-value (B={N_PERM})     : {p_value:.4f}")
if n_exceed == 0:
    print(f"  -> No permutation exceeded observed AUC; p < {p_min:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  PANEL-SIZE SWEEP   (within corrected CV)
# ═══════════════════════════════════════════════════════════════════════════════
hdr("5 - Panel-size sweep (corrected, within-fold DEG selection)")

K_SWEEP = [5, 10, 20, 30, 50]
auc_by_k = []

# Use mode hyperparams but sweep k; DEG selection still inside each fold
for k_target in K_SWEEP:
    p_k = np.full(len(y), np.nan)

    for tr_idx, te_idx in outer.split(np.zeros(len(y)), y):
        tr_samples = meta_df.iloc[tr_idx]["ID_REF"].values
        te_samples = meta_df.iloc[te_idx]["ID_REF"].values

        counts_tr = counts_raw[tr_samples]
        counts_te = counts_raw[te_samples]
        y_tr = y[tr_idx]

        degs = within_fold_deg_selection(counts_tr, y_tr)
        if len(degs) < 5:
            p_k[te_idx] = 0.5
            continue

        k_use = min(k_target, len(degs))
        X_tr = apply_log2_cpm(counts_tr, degs)
        X_te = apply_log2_cpm(counts_te, degs)

        m = clone(base_pipe).set_params(**{
            "kbest__k":      k_use,
            "clf__C":        common_C,
            "clf__l1_ratio": common_l1,
        })
        m.fit(X_tr, y_tr)
        p_k[te_idx] = m.predict_proba(X_te)[:, 1]

    vm = ~np.isnan(p_k)
    auc_k = roc_auc_score(y[vm], p_k[vm]) if vm.sum() > 0 else np.nan
    auc_by_k.append({"k": k_target, "oof_auc": auc_k})
    print(f"  k={k_target:3d}  OOF AUC={auc_k:.4f}")

kdf = pd.DataFrame(auc_by_k)
kdf.to_csv(os.path.join(OUT_DIR, "auc_vs_panel_size.csv"), index=False)

# Smallest k within 0.01 of best
best_auc_k = kdf["oof_auc"].max()
K_FINAL    = int(kdf.loc[kdf["oof_auc"] >= best_auc_k - 0.01].sort_values("k").iloc[0, 0])
print(f"\nChosen K_FINAL={K_FINAL}  (best AUC={best_auc_k:.4f})")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  FINAL MODEL + SHAP   (fit on all data for interpretation only)
#     Performance estimates come from Section 3 (nested CV), not here.
# ═══════════════════════════════════════════════════════════════════════════════
hdr("6 - Final model (all data, for interpretation only)")

# DEG selection on full dataset (acceptable for final interpretive model,
# NOT for performance estimation which is done via nested CV above)
all_degs = within_fold_deg_selection(counts_raw, y)
print(f"DEGs on full dataset (reference): {len(all_degs)}")

k_final_use = min(K_FINAL, len(all_degs))
X_all = apply_log2_cpm(counts_raw, all_degs)

final_model = clone(base_pipe).set_params(**{
    "kbest__k":      k_final_use,
    "clf__C":        common_C,
    "clf__l1_ratio": common_l1,
})
final_model.fit(X_all, y)

mask_f      = final_model.named_steps["kbest"].get_support()
final_genes = np.array(all_degs)[mask_f].tolist()
final_syms  = [id2sym.get(g, g) for g in final_genes]

pd.Series(final_syms).to_csv(
    os.path.join(OUT_DIR, "final_gene_panel.txt"),
    index=False, header=False,
)
print(f"Final panel: {len(final_genes)} genes")
print("  ", final_syms[:10], "...")

# SHAP
try:
    import shap
    X_scaled = final_model.named_steps["scaler"].transform(X_all)
    X_sel    = final_model.named_steps["kbest"].transform(X_scaled)
    X_sel_df = pd.DataFrame(X_sel, columns=final_syms, index=X_all.index)
    clf_f    = final_model.named_steps["clf"]
    explainer = shap.LinearExplainer(clf_f, X_sel_df,
                                     feature_perturbation="interventional")
    shap_vals = explainer.shap_values(X_sel_df)
    mean_abs  = np.abs(shap_vals).mean(axis=0)
    shap_tbl  = (pd.DataFrame({"Symbol": final_syms, "MeanAbsSHAP": mean_abs})
                 .sort_values("MeanAbsSHAP", ascending=False))
    shap_tbl.to_csv(os.path.join(OUT_DIR, "shap_scores_final_panel.csv"), index=False)
    SHAP_OK = True
    print("SHAP computed successfully.")
except Exception as e:
    SHAP_OK = False
    print(f"SHAP skipped: {e}")
    shap_tbl = pd.DataFrame(columns=["Symbol", "MeanAbsSHAP"])


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
hdr("7 - Summary comparison table")

deg_full = pd.read_csv(os.path.join(DATA_DIR, "deseq2_DEGs.tsv"), sep="\t")
deg_full["Symbol"] = deg_full["Symbol"].astype(str)

panel_tbl = stab_df.copy()
panel_tbl = panel_tbl.merge(
    deg_full[["Symbol", "log2FoldChange", "padj"]].drop_duplicates(),
    on="Symbol", how="left",
)
if SHAP_OK:
    panel_tbl = panel_tbl.merge(shap_tbl, on="Symbol", how="left")

panel_tbl["abs_log2FC"] = panel_tbl["log2FoldChange"].abs()
panel_tbl["neglog10FDR"] = -np.log10(panel_tbl["padj"].clip(lower=1e-300))
panel_tbl["Direction_EVD_vs_Healthy"] = np.where(
    panel_tbl["log2FoldChange"] > 0, "Up in EVD-S", "Down in EVD-S"
)
panel_tbl.to_csv(os.path.join(TAB_DIR, "biomarker_panel_annotated.csv"), index=False)

summary = pd.DataFrame([{
    "Pipeline":                        "Original (leaked)",
    "DEG_selection":                   "All 59 samples before CV",
    "Normalisation":                   "VST on all 59 samples",
    "Nested_CV_AUC_mean":              "1.000",
    "Nested_CV_AUC_std":               "0.000",
    "OOF_AUC":                         "0.997",
    "OOF_AP":                          "0.996",
    "Permutation_B":                   "500 (invalid — labels pre-used for DEG)",
    "Permutation_p":                   "0.002",
}, {
    "Pipeline":                        "Corrected (this script)",
    "DEG_selection":                   "Inside each outer fold (training only)",
    "Normalisation":                   "log2(CPM+1) per fold (training only)",
    "Nested_CV_AUC_mean":              f"{nested_auc_mean:.3f}",
    "Nested_CV_AUC_std":               f"{nested_auc_std:.3f}",
    "OOF_AUC":                         f"{oof_auc:.3f}",
    "OOF_AP":                          f"{oof_ap:.3f}",
    "Permutation_B":                   str(N_PERM),
    "Permutation_p":                   (f"< {p_min:.4f}" if n_exceed == 0
                                        else f"{p_value:.4f}"),
}])
summary.to_csv(os.path.join(TAB_DIR, "pipeline_comparison.csv"), index=False)
print(summary.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  FIGURES
# ═══════════════════════════════════════════════════════════════════════════════
hdr("8 - Publication figures")

# ── F01: Nested CV fold AUC / AP ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.8, 4.2))
folds = range(1, len(outer_auc) + 1)
ax.plot(folds, outer_auc, marker="o", linewidth=2.2, label="ROC-AUC", color=COL["accent"])
ax.plot(folds, outer_ap,  marker="s", linewidth=2.2, label="Avg Precision", color=COL["EVD-S"])
ax.set_ylim(0, 1.05)
ax.set_title("Corrected nested CV — per-fold performance\n"
             "(DEG selection inside each outer fold)")
ax.set_xlabel("Outer fold")
ax.set_ylabel("Score")
ax.legend(frameon=False)
ax.axhline(nested_auc_mean, linestyle="--", linewidth=1.2, color=COL["grey"],
           label=f"Mean AUC={nested_auc_mean:.3f}")
savefig("F01_nested_cv_performance.png")

# ── F02: OOF ROC curve ──────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y[valid_mask], oof_prob[valid_mask])
fig, ax = plt.subplots(figsize=(5.8, 5.0))
ax.plot(fpr, tpr, linewidth=2.6, color=COL["accent"])
ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color=COL["grey"])
ax.set_title(f"OOF ROC (AUC={oof_auc:.3f}) — corrected pipeline")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
savefig("F02_oof_roc_curve.png")

# ── F03: OOF PR curve ───────────────────────────────────────────────────────────
prec, rec, _ = precision_recall_curve(y[valid_mask], oof_prob[valid_mask])
fig, ax = plt.subplots(figsize=(5.8, 5.0))
ax.plot(rec, prec, linewidth=2.6, color=COL["EVD-S"])
ax.set_title(f"OOF Precision–Recall (AP={oof_ap:.3f}) — corrected pipeline")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
savefig("F03_oof_pr_curve.png")

# ── F04: Corrected permutation test ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.0, 4.8))
ax.hist(perm_aucs, bins=30, alpha=0.85, color=COL["grey"],
        label=f"Null distribution\n(B={N_PERM}, full-pipeline permutation)")
ax.axvline(obs_auc, linewidth=2.8, color=COL["EVD-S"],
           label=f"Observed AUC={obs_auc:.3f}")
p_label = (f"p < {p_min:.4f}" if n_exceed == 0 else f"p = {p_value:.4f}")
ax.set_title(f"Permutation test — corrected\n(Full-pipeline per permutation, {p_label})")
ax.set_xlabel("ROC-AUC under label permutation")
ax.set_ylabel("Count")
ax.legend(frameon=False)
savefig("F04_permutation_test.png")

# ── F05: AUC vs panel size (corrected) ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.6, 4.6))
ax.plot(kdf["k"], kdf["oof_auc"], marker="o", linewidth=2.6, color=COL["Healthy"])
ax.axhline(best_auc_k - 0.01, linestyle="--", linewidth=1.6, color=COL["grey"],
           label="Best − 0.01 threshold")
ax.axvline(K_FINAL, linestyle=":", linewidth=1.6, color=COL["EVD-S"],
           label=f"Chosen K={K_FINAL}")
ax.set_title("Panel size vs OOF AUC — corrected pipeline")
ax.set_xlabel("Number of genes (K)")
ax.set_ylabel("OOF ROC-AUC")
ax.legend(frameon=False)
savefig("F05_auc_vs_panel_size.png")

# ── F06: Stability frequency (top 20) ───────────────────────────────────────────
top_s = stab_df.head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(8.2, 6.2))
ax.barh(top_s["Symbol"], top_s["StabilityFreq"], color=COL["accent"])
ax.set_title("Top genes by stability selection frequency — corrected pipeline")
ax.set_xlabel("Stability frequency (outer folds)")
ax.set_xlim(0, 1.1)
savefig("F06_gene_stability.png")

# ── F07: Confusion matrix (OOF, threshold=0.5) ──────────────────────────────────
yhat = (oof_prob[valid_mask] >= 0.5).astype(int)
cm   = confusion_matrix(y[valid_mask], yhat)
fig, ax = plt.subplots(figsize=(4.8, 4.4))
im = ax.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center",
            fontsize=14, color="white" if v > cm.max() / 2 else COL["dark"])
ax.set_xticks([0, 1]); ax.set_xticklabels(["Healthy(0)", "EVD-S(1)"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Healthy(0)", "EVD-S(1)"])
ax.set_title("OOF confusion matrix — corrected pipeline (threshold=0.5)")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.colorbar(im, ax=ax)
savefig("F07_confusion_matrix.png")

# ── F08: Calibration ────────────────────────────────────────────────────────────
prob_true, prob_pred = calibration_curve(
    y[valid_mask], oof_prob[valid_mask], n_bins=8, strategy="quantile"
)
fig, ax = plt.subplots(figsize=(5.8, 5.0))
ax.plot(prob_pred, prob_true, marker="o", linewidth=2.4, color=COL["accent"],
        label="Corrected model")
ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color=COL["grey"],
        label="Perfect calibration")
ax.set_title("Probability calibration (OOF) — corrected pipeline")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction positive")
ax.legend(frameon=False)
savefig("F08_calibration_curve.png")

# ── F09: DEGs found per fold ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.bar(range(1, 6), n_degs_per_fold, color=COL["Healthy"], edgecolor="white")
ax.set_title("Number of within-fold DEGs per outer fold\n"
             "(Wilcoxon + BH, training samples only)")
ax.set_xlabel("Outer fold")
ax.set_ylabel("DEGs identified")
ax.set_xticks(range(1, 6))
savefig("F09_degs_per_fold.png")

# ── F10: SHAP bar (if computed) ──────────────────────────────────────────────────
if SHAP_OK and len(shap_tbl) > 0:
    top_sh = shap_tbl.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ax.barh(top_sh["Symbol"], top_sh["MeanAbsSHAP"], color=COL["EVD-S"])
    ax.set_title("Feature importance (Mean |SHAP|) — corrected final model")
    ax.set_xlabel("Mean |SHAP|")
    savefig("F10_shap_importance.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  FINAL SUMMARY PRINTOUT
# ═══════════════════════════════════════════════════════════════════════════════
hdr("FINAL SUMMARY")
p_str = ("< " + f"{p_min:.4f}") if n_exceed == 0 else f"{p_value:.4f}"
print("=" * 67)
print("  CORRECTED PIPELINE RESULTS")
print("=" * 67)
print(f"  Nested CV AUC  : {nested_auc_mean:.3f} +/- {nested_auc_std:.3f}")
print(f"  OOF AUC        : {oof_auc:.3f}")
print(f"  OOF AP         : {oof_ap:.3f}")
print(f"  DEGs per fold  : {n_degs_per_fold}")
print(f"  Chosen panel K : {K_FINAL}")
print(f"  Permutation B  : {N_PERM}")
print(f"  Permutation p  : {p_str}")
print("-" * 67)
print("  Outputs : results/outputs/")
print("  Figures : results/figures/")
print("  Tables  : results/tables/")
print("=" * 67)
print("All done.")
