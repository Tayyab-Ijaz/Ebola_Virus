"""Stage registry. Each stage module exposes run(ctx). `--stages all` runs IMPLEMENTED in this order; the order
matters because later stages reuse results cached by earlier ones in the same run (and fall back to computing
them again when run alone).

  item7              Welch vs DESeq2 vs limma-voom rankings (R) - also writes the limma tables used by S2
  item10a            DESeq2 with and without inferred sex (R)
  item9              cell-type deconvolution (R, immunedeconv)
  item6              PCA / VST, PC-only baseline (+ S7: up to 20 PCs)
  item1              leakage tests, primary pipeline and sensitivity runs S1, S3, S4, S5, S6
  item1_s2           S2: limma-voom as the in-split ranker (seed 42)
  item2_permutation  full-pipeline permutation nulls (B = 1000): primary, S1 hard gate, sex-stratified
  item3_stability    stability, tie-break ordering, final K, final panel, SHAP, P5
  item4_panel_size   fixed-K sweep (descriptive) and K = 5 vs 4 ablation inside CV
  item5_cis          threshold metrics, calibration, Wilson CIs
  item6b             PC context (sex, library size, globin, composition), composition-only baseline, P1
  item9_sens         deconvolution without the two non-converged samples (GSM4261710, GSM4261715)
  item8              concordance with the authors' hg19 / GENCODE v19 CPM table
  item10bcde         sex-only and panel + sex baselines, error analysis by sex
  summary            all variants side by side
"""
import importlib

IMPLEMENTED = ["item7", "item10a", "item9", "item6", "item1", "item1_s2", "item2_permutation", "item3_stability",
               "item4_panel_size", "item5_cis", "item6b", "item9_sens", "item8", "item10bcde", "summary"]
PENDING = []
R_STAGES = {"item6", "item7", "item10a", "item9", "item9_sens"}


def get(name):
    return importlib.import_module(f"stages.{name}").run
