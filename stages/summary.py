"""All variants side by side (one row per model), plus a digest of permutation, stability, P1 and R warnings.

Column vocabulary (never mixed up):
  pooled_*         seed-42 pooled out-of-fold AUC. ci_B_delong = DeLong CI. ci_A_boot = stratified subject-bootstrap CI.
  per_fold_*       seed-42 mean and SD (ddof = 1) of the 5 outer-fold AUCs
  repeated_*       mean over 100 repeat seeds of the mean-fold AUC; ci_A_boot = subject-bootstrap CI (main interval)
  partition_*      2.5 / 97.5 percentiles of the per-repeat statistic = variability caused by the random split (NOT a CI)"""
import collections
import json

import pandas as pd

from evd.runs import variant_stats

STAGE = "summary"
ROLE = {"primary": "PRIMARY (D1)", "S1_hard_gate": "S1 sensitivity", "S2_limma_voom": "S2 sensitivity (seed 42 only)",
        "S3_no_sex_chrom": "S3 sensitivity", "S4_legacy_structure": "S4 diagnostic (old structure)",
        "S5_effect_size_filter": "S5 sensitivity", "S6_K_to_100": "S6 sensitivity (grid)",
        "PC_only": "baseline", "S7_PC_only_to_20": "S7 sensitivity (grid)", "composition_all": "baseline (exploratory)",
        "composition_epic": "baseline (exploratory)", "composition_quantiseq": "baseline (exploratory)",
        "composition_mcp_counter": "baseline (exploratory)", "sex_only": "baseline", "primary_plus_sex": "primary + inferred sex"}


def _read(ctx, rel):
    p = ctx.run_dir / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def run(ctx):
    cfg, y, log = ctx.cfg, ctx.ds.y, ctx.log
    B, rs = cfg["item1"]["bootstrap_B"], cfg["seeds"]["bootstrap"]
    rows = []
    for tag in ROLE:
        run_ = ctx.cache.get("runs", {}).get(tag)
        if run_ is None:
            log.warning("variant %s not available in this run", tag)
            continue
        st = variant_stats(y, run_, B, rs)
        s = st["seed42"]
        r = {"variant": tag, "role": ROLE[tag], "pooled_auc": s["pooled_oof_auc"],
             "pooled_ci_B_delong_lo": s["pooled_ci_B_delong"]["lo"], "pooled_ci_B_delong_hi": s["pooled_ci_B_delong"]["hi"],
             "pooled_ci_A_boot_lo": s["pooled_ci_A_subject_bootstrap"]["boot_lo"], "pooled_ci_A_boot_hi": s["pooled_ci_A_subject_bootstrap"]["boot_hi"],
             "pooled_ap": s["pooled_oof_ap"], "per_fold_mean": s["per_fold_mean"], "per_fold_sd": s["per_fold_sd_ddof1"],
             "K_chosen_seed42": ";".join(map(str, s["K_chosen"])) if s["K_chosen"] else ""}
        if "repeated_cv" in st:
            rc = st["repeated_cv"]
            r.update({"repeated_mean_fold_auc": rc["mean_fold_auc"], "repeated_ci_A_boot_lo": rc["ci_A_subject_bootstrap"]["lo"],
                      "repeated_ci_A_boot_hi": rc["ci_A_subject_bootstrap"]["hi"],
                      "partition_p2.5": rc["partition_variability"]["p2.5"], "partition_p97.5": rc["partition_variability"]["p97.5"],
                      "repeated_mean_fold_ap": rc["mean_fold_ap"], "K_chosen_median_100x5": rc["K_chosen_median"]})
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(ctx.out(STAGE, "side_by_side_all_variants.csv"), index=False)
    view = df[["variant", "pooled_auc", "pooled_ci_B_delong_lo", "pooled_ci_B_delong_hi", "per_fold_mean", "repeated_mean_fold_auc",
               "repeated_ci_A_boot_lo", "repeated_ci_A_boot_hi", "partition_p2.5", "partition_p97.5"]]
    log.info("ALL VARIANTS SIDE BY SIDE (pooled = seed 42; repeated = 100 seeds; CI-A bootstrap, CI-B DeLong, partition = split variability):\n%s",
             view.round(4).to_string(index=False))

    head = {"permutation": _read(ctx, "item2_permutation/permutation_summary.json"),
            "stability": _read(ctx, "item3_stability/stability_summary.json"),
            "P1": _read(ctx, "item6b/P1_paired_comparisons.json"),
            "deconvolution_sensitivity": _read(ctx, "item9_sens/deconvolution_sensitivity.json"),
            "ablation_K5_vs_K4": _read(ctx, "item4_panel_size/ablation_K5_vs_K4.json")}
    ctx.out(STAGE, "headline.json").write_text(json.dumps(head, indent=2), encoding="utf-8")

    cnt = collections.Counter(w.split(":", 1)[1].strip() if ":" in w else w for w in ctx.r_warnings)
    lines = [f"{n:3d} x {w}" for w, n in cnt.most_common()]
    ctx.out(STAGE, "r_warnings_digest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("R warnings printed during this run (%d, %d distinct):\n%s", len(ctx.r_warnings), len(cnt), "\n".join(lines) or "none")
