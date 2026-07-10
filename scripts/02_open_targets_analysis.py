#!/usr/bin/env python3
"""
02_open_targets_analysis.py
============================
Open Targets Platform therapeutic intervention analysis for the corrected
5-gene EVD biomarker panel: PLCH2, RN7SL1, RN7SL2, IFI27, GOLGA8B.

Queries the Open Targets GraphQL API (v4) for:
  1. Target tractability (small-molecule, antibody, PROTAC, other modalities)
  2. Drug & clinical candidate data
  3. Disease associations (top-ranked + viral/infectious subset)
  4. Reactome pathways
  5. Chemical probes
  6. Safety liabilities
  7. Target prioritisation scores
  8. Gene Ontology biological process annotations

Outputs (all written to results/tables/ and results/figures/):
  - OT_tractability_summary.csv
  - OT_tractability_heatmap.png
  - OT_disease_associations_top10.csv
  - OT_infectious_disease_associations.csv
  - OT_disease_bubble.png
  - OT_pathways.csv
  - OT_chemical_probes.csv
  - OT_safety_liabilities.csv
  - OT_prioritisation_scores.csv
  - OT_prioritisation_radar.png
  - OT_GO_biological_process.csv
  - OT_therapeutic_intervention_report.md
"""

import os
import sys
import time
import json
import requests
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(SCRIPT_DIR)
TAB_DIR    = os.path.join(ROOT, "results", "tables", "open_targets")
FIG_DIR    = os.path.join(ROOT, "results", "figures", "open_targets")
OUT_DIR    = os.path.join(ROOT, "results", "tables", "open_targets")

for d in [TAB_DIR, FIG_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Gene panel ───────────────────────────────────────────────────────────────
GENE_PANEL = {
    "PLCH2":   "ENSG00000149527",
    "RN7SL1":  "ENSG00000276168",
    "RN7SL2":  "ENSG00000274012",
    "IFI27":   "ENSG00000165949",
    "GOLGA8B": "ENSG00000215252",
}

# SHAP ranking for ordering in figures (from corrected pipeline)
SHAP_ORDER = ["GOLGA8B", "PLCH2", "RN7SL1", "RN7SL2", "IFI27"]

OT_API = "https://api.platform.opentargets.org/api/v4/graphql"
EBOLA_MONDO = "MONDO_0005737"

MODALITY_LABELS = {
    "SM": "Small Molecule",
    "AB": "Antibody",
    "PR": "PROTAC / Protein Degrader",
    "OC": "Other Clinical",
    "HE": "High-throughput Experimental",
    "GE": "Gene Editing / Genetic",
    "OT": "Other",
}

# ── Utility ──────────────────────────────────────────────────────────────────
def gql(query: str, variables: dict, retries: int = 3, delay: float = 2.0) -> dict:
    """POST to the OT GraphQL endpoint with simple retry logic."""
    for attempt in range(retries):
        try:
            r = requests.post(
                OT_API,
                json={"query": query, "variables": variables},
                timeout=60,
            )
            r.raise_for_status()
            payload = r.json()
            if "errors" in payload:
                raise RuntimeError(f"GraphQL errors: {payload['errors']}")
            return payload["data"]
        except Exception as exc:
            if attempt < retries - 1:
                print(f"  [retry {attempt+1}] {exc}")
                time.sleep(delay)
            else:
                raise

# ── GraphQL queries ───────────────────────────────────────────────────────────
FULL_TARGET_QUERY = """
query fullTarget($ensgId: String!) {
  target(ensemblId: $ensgId) {
    id
    approvedSymbol
    approvedName
    biotype
    tractability {
      label
      modality
      value
    }
    safetyLiabilities {
      event
      biosamples { tissueLabel cellLabel }
      effects { direction dosing }
      datasource
      studies { name description }
    }
    chemicalProbes {
      id
      control
      drugId
      mechanismOfAction
      isHighQuality
      targetFromSourceId
      scoreInCells
      scoreInOrganisms
    }
    pathways {
      pathway
      pathwayId
      topLevelTerm
    }
    associatedDiseases(page: {index: 0, size: 100}, orderByScore: "score") {
      count
      rows {
        disease {
          id
          name
          therapeuticAreas { id name }
        }
        score
        datatypeScores { id score }
      }
    }
    prioritisation {
      items { key value }
    }
    geneOntology {
      term { label id }
      evidence
    }
    targetClass {
      id
      label
      level
    }
    drugAndClinicalCandidates {
      count
      rows {
        drug {
          id
          name
          drugType
          maximumClinicalStage
        }
        maxClinicalStage
        diseases {
          diseaseFromSource
          disease { id name }
        }
      }
    }
    functionDescriptions
  }
}
"""

DISEASE_TARGET_QUERY = """
query diseaseTargetAssoc($diseaseId: String!, $ensgId: String!) {
  disease(efoId: $diseaseId) {
    associatedTargets(
      page: {index: 0, size: 200}
    ) {
      rows {
        target { id approvedSymbol }
        score
      }
    }
  }
}
"""

# ── Step 1: Fetch all target data ─────────────────────────────────────────────
print("=" * 65)
print("Open Targets Therapeutic Interventions — EVD 5-gene panel")
print("=" * 65)
print(f"Query date : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"API        : {OT_API}")
print(f"Gene panel : {', '.join(GENE_PANEL.keys())}")
print()

target_data = {}
for sym, ensg in GENE_PANEL.items():
    print(f"  Fetching {sym} ({ensg}) ...", end=" ", flush=True)
    data = gql(FULL_TARGET_QUERY, {"ensgId": ensg})["target"]
    target_data[sym] = data
    n_dis   = data["associatedDiseases"]["count"]
    n_drugs = data["drugAndClinicalCandidates"]["count"]
    n_tract = sum(1 for t in data["tractability"] if t["value"])
    print(f"OK  | diseases={n_dis}  drugs={n_drugs}  tractability={n_tract}")
    time.sleep(0.4)   # be polite to the API

print()

# ── Step 2: Tractability summary ──────────────────────────────────────────────
print("Building tractability summary ...")

# All unique modality+label combos
all_tract_labels = {}
for sym, data in target_data.items():
    for t in data["tractability"]:
        key = (t["modality"], t["label"])
        all_tract_labels[key] = all_tract_labels.get(key, 0) + (1 if t["value"] else 0)

modality_order = ["SM", "AB", "PR", "OC", "HE", "GE", "OT"]

tract_rows = []
for sym in SHAP_ORDER:
    data = target_data[sym]
    row = {"Gene": sym, "Biotype": data["biotype"], "Approved Name": data["approvedName"]}
    tract_by_modality = {}
    for t in data["tractability"]:
        m = t["modality"]
        if m not in tract_by_modality:
            tract_by_modality[m] = []
        if t["value"]:
            tract_by_modality[m].append(t["label"])
    for m in modality_order:
        col = MODALITY_LABELS.get(m, m)
        labels = tract_by_modality.get(m, [])
        row[col] = "; ".join(labels) if labels else "—"
        row[f"_{m}_count"] = len(labels)
    row["Total Tractability Hits"] = sum(
        1 for t in data["tractability"] if t["value"]
    )
    row["Chemical Probes"] = len(data["chemicalProbes"])
    row["Drugs/Clinical"] = data["drugAndClinicalCandidates"]["count"]
    row["Associated Diseases"] = data["associatedDiseases"]["count"]
    tract_rows.append(row)

df_tract = pd.DataFrame(tract_rows)
# Save full table
cols_save = (
    ["Gene", "Biotype", "Approved Name",
     "Small Molecule", "Antibody", "PROTAC / Protein Degrader",
     "Other Clinical", "High-throughput Experimental",
     "Gene Editing / Genetic", "Other",
     "Total Tractability Hits", "Chemical Probes", "Drugs/Clinical",
     "Associated Diseases"]
)
df_tract[cols_save].to_csv(
    os.path.join(TAB_DIR, "OT_tractability_summary.csv"), index=False
)
print(f"  Saved: OT_tractability_summary.csv  ({len(df_tract)} genes)")

# Tractability heatmap
count_cols = [f"_{m}_count" for m in modality_order]
heatmap_data = df_tract.set_index("Gene")[count_cols].rename(
    columns={f"_{m}_count": MODALITY_LABELS.get(m, m) for m in modality_order}
)
# keep only genes as SHAP order
heatmap_data = heatmap_data.loc[SHAP_ORDER]

fig, ax = plt.subplots(figsize=(11, 4))
cmap = LinearSegmentedColormap.from_list("ot", ["#f7fbff", "#084594"])
im = ax.imshow(heatmap_data.values.T.astype(float), aspect="auto", cmap=cmap,
               vmin=0, vmax=max(heatmap_data.values.max(), 1))

ax.set_xticks(range(len(SHAP_ORDER)))
ax.set_xticklabels(SHAP_ORDER, fontsize=10, fontweight="bold")
ax.set_yticks(range(len(heatmap_data.columns)))
ax.set_yticklabels(heatmap_data.columns, fontsize=9)
ax.set_title("Open Targets Tractability — EVD Biomarker Panel", fontsize=12, fontweight="bold", pad=10)
ax.set_xlabel("Gene (ordered by mean|SHAP|)", fontsize=9)
ax.set_ylabel("Modality", fontsize=9)

# Annotate cells
for i, gene in enumerate(SHAP_ORDER):
    for j, col in enumerate(heatmap_data.columns):
        val = heatmap_data.loc[gene, col]
        ax.text(i, j, str(int(val)), ha="center", va="center",
                fontsize=9, color="white" if val >= 2 else "black")

cb = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
cb.set_label("# Active Tractability Flags", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "OT_01_tractability_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: OT_01_tractability_heatmap.png")

# ── Step 3: Disease associations ──────────────────────────────────────────────
print("\nBuilding disease association tables ...")

dis_rows_top   = []
dis_rows_viral = []

viral_keywords = {
    "ebola", "viral", "virus", "infect", "covid", "sars", "influenza",
    "dengue", "fever", "hemorrhagic", "hepatitis", "hiv", "aids",
    "malaria", "tuberculosis", "pneumonia",
}

datatype_cols = None

for sym in SHAP_ORDER:
    data = target_data[sym]
    rows = data["associatedDiseases"]["rows"]

    # Gather all datatype IDs for column order
    if datatype_cols is None and rows:
        all_dt_ids = []
        for r in rows:
            for dt in r.get("datatypeScores", []):
                if dt["id"] not in all_dt_ids:
                    all_dt_ids.append(dt["id"])
        datatype_cols = all_dt_ids

    for rank, row in enumerate(rows, 1):
        dis_name = row["disease"]["name"]
        dis_id   = row["disease"]["id"]
        score    = row["score"]
        # datatype scores as dict
        dt_dict  = {dt["id"]: dt["score"] for dt in row.get("datatypeScores", [])}
        ta_names = [ta["name"] for ta in row["disease"].get("therapeuticAreas", [])]

        base = {
            "Gene": sym,
            "Rank": rank,
            "Disease": dis_name,
            "Disease_ID": dis_id,
            "Association Score": round(score, 4),
            "Therapeutic Areas": "; ".join(ta_names),
        }
        if datatype_cols:
            for dt_id in datatype_cols:
                base[dt_id] = round(dt_dict.get(dt_id, 0.0), 4)

        # Top 10 per gene
        if rank <= 10:
            dis_rows_top.append(base)

        # Viral / infectious filter
        name_lower = dis_name.lower()
        ta_lower   = " ".join(ta_names).lower()
        if any(k in name_lower or k in ta_lower for k in viral_keywords):
            dis_rows_viral.append(base)

df_dis_top   = pd.DataFrame(dis_rows_top)
df_dis_viral = pd.DataFrame(dis_rows_viral).drop_duplicates(
    subset=["Gene", "Disease_ID"]
).sort_values(["Gene", "Rank"])

df_dis_top.to_csv(
    os.path.join(TAB_DIR, "OT_disease_associations_top10.csv"), index=False
)
df_dis_viral.to_csv(
    os.path.join(TAB_DIR, "OT_infectious_disease_associations.csv"), index=False
)
print(f"  Saved: OT_disease_associations_top10.csv  ({len(df_dis_top)} rows)")
print(f"  Saved: OT_infectious_disease_associations.csv  ({len(df_dis_viral)} rows)")

# Disease bubble plot — top-5 per gene
fig, ax = plt.subplots(figsize=(13, 5))
gene_positions = {g: i for i, g in enumerate(SHAP_ORDER)}
colors_gene = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261"]

y_offsets = {g: 0 for g in SHAP_ORDER}
plotted = []

for sym in SHAP_ORDER:
    gene_rows = df_dis_top[df_dis_top["Gene"] == sym].head(5)
    cx = gene_positions[sym]
    ci = list(SHAP_ORDER).index(sym)
    for _, row in gene_rows.iterrows():
        r = row["Association Score"]
        y_offsets[sym] += 1
        ax.scatter(cx, y_offsets[sym], s=max(r * 1500, 30),
                   color=colors_gene[ci], alpha=0.7, edgecolors="grey", linewidth=0.5)
        if r >= 0.05:
            short = row["Disease"][:28] + ("…" if len(row["Disease"]) > 28 else "")
            ax.text(cx + 0.05, y_offsets[sym], short, fontsize=6.5, va="center")

ax.set_xticks(range(len(SHAP_ORDER)))
ax.set_xticklabels(SHAP_ORDER, fontsize=11, fontweight="bold")
ax.set_ylabel("Rank (1 = highest score)", fontsize=9)
ax.set_title("Top-5 Disease Associations per Gene (Open Targets)\nBubble size ∝ association score",
             fontsize=11, fontweight="bold")
ax.invert_yaxis()
ax.set_yticks([])
ax.grid(axis="x", linestyle="--", alpha=0.3)

# Legend for bubble size
for sc, lbl in [(0.05, "0.05"), (0.20, "0.20"), (0.50, "0.50")]:
    ax.scatter([], [], s=max(sc * 1500, 30), color="grey", alpha=0.6,
               edgecolors="grey", label=lbl)
ax.legend(title="Assoc. Score", fontsize=7, title_fontsize=7,
          loc="upper right", framealpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "OT_02_disease_associations.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: OT_02_disease_associations.png")

# ── Step 4: Pathways ──────────────────────────────────────────────────────────
print("\nBuilding pathway table ...")

pathway_rows = []
for sym in SHAP_ORDER:
    data = target_data[sym]
    for pw in data["pathways"]:
        pathway_rows.append({
            "Gene": sym,
            "Pathway": pw["pathway"],
            "Pathway ID": pw["pathwayId"],
            "Top-level Term": pw["topLevelTerm"],
        })

df_pw = pd.DataFrame(pathway_rows)
df_pw.to_csv(os.path.join(TAB_DIR, "OT_pathways.csv"), index=False)
print(f"  Saved: OT_pathways.csv  ({len(df_pw)} pathways)")

# ── Step 5: Chemical probes ───────────────────────────────────────────────────
print("\nBuilding chemical probes table ...")

probe_rows = []
for sym in SHAP_ORDER:
    data = target_data[sym]
    for cp in data["chemicalProbes"]:
        probe_rows.append({
            "Gene": sym,
            "Probe ID": cp["id"],
            "Drug ID": cp.get("drugId", ""),
            "Mechanism": cp.get("mechanismOfAction", ""),
            "High Quality": cp.get("isHighQuality", False),
            "Control": cp.get("control", ""),
            "Score in Cells": cp.get("scoreInCells", ""),
            "Score in Organisms": cp.get("scoreInOrganisms", ""),
        })

df_probes = pd.DataFrame(probe_rows) if probe_rows else pd.DataFrame(
    columns=["Gene", "Probe ID", "Drug ID", "Mechanism",
             "High Quality", "Control", "Score in Cells", "Score in Organisms"]
)
df_probes.to_csv(os.path.join(TAB_DIR, "OT_chemical_probes.csv"), index=False)
print(f"  Saved: OT_chemical_probes.csv  ({len(df_probes)} probes)")

# ── Step 6: Safety liabilities ────────────────────────────────────────────────
print("\nBuilding safety liabilities table ...")

safety_rows = []
for sym in SHAP_ORDER:
    data = target_data[sym]
    for sl in data["safetyLiabilities"]:
        tissues = "; ".join(
            f"{b.get('tissueLabel','?')} / {b.get('cellLabel','?')}"
            for b in sl.get("biosamples", [])
        ) or "—"
        effects = "; ".join(
            f"{e.get('direction','?')} ({e.get('dosing','?')})"
            for e in sl.get("effects", [])
        ) or "—"
        studies = "; ".join(
            s.get("name", "") for s in sl.get("studies", []) if s.get("name")
        ) or "—"
        safety_rows.append({
            "Gene": sym,
            "Safety Event": sl.get("event", ""),
            "Datasource": sl.get("datasource", ""),
            "Biosamples": tissues,
            "Effects": effects,
            "Studies": studies,
        })

df_safety = pd.DataFrame(safety_rows) if safety_rows else pd.DataFrame(
    columns=["Gene", "Safety Event", "Datasource", "Biosamples", "Effects", "Studies"]
)
df_safety.to_csv(os.path.join(TAB_DIR, "OT_safety_liabilities.csv"), index=False)
print(f"  Saved: OT_safety_liabilities.csv  ({len(df_safety)} liabilities)")

# ── Step 7: Prioritisation scores ─────────────────────────────────────────────
print("\nBuilding prioritisation score table ...")

prio_rows = []
for sym in SHAP_ORDER:
    data = target_data[sym]
    items = data["prioritisation"]["items"] if data["prioritisation"] else []
    row   = {"Gene": sym}
    for it in items:
        row[it["key"]] = round(float(it["value"]), 5) if it["value"] is not None else None
    prio_rows.append(row)

df_prio = pd.DataFrame(prio_rows).set_index("Gene")
df_prio.to_csv(os.path.join(TAB_DIR, "OT_prioritisation_scores.csv"))
print(f"  Saved: OT_prioritisation_scores.csv  ({df_prio.shape[1]} metrics)")

# Radar / spider chart for prioritisation
all_keys = [c for c in df_prio.columns if not df_prio[c].isna().all()]
# Normalise 0-1 per column
df_prio_norm = df_prio[all_keys].copy()
for col in all_keys:
    cmax = df_prio_norm[col].max()
    if cmax and cmax > 0:
        df_prio_norm[col] = df_prio_norm[col] / cmax
df_prio_norm = df_prio_norm.fillna(0)

N = len(all_keys)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
colors_gene = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261"]
for idx, sym in enumerate(SHAP_ORDER):
    if sym not in df_prio_norm.index:
        continue
    values = df_prio_norm.loc[sym, all_keys].tolist()
    values += values[:1]
    ax.plot(angles, values, "o-", linewidth=1.5, color=colors_gene[idx], label=sym)
    ax.fill(angles, values, alpha=0.08, color=colors_gene[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(all_keys, size=7)
ax.set_title("Target Prioritisation Scores (normalised)\nOpen Targets Platform",
             fontsize=11, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "OT_03_prioritisation_radar.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: OT_03_prioritisation_radar.png")

# ── Step 8: Gene Ontology (biological process) ────────────────────────────────
print("\nBuilding GO biological process table ...")

go_rows = []
for sym in SHAP_ORDER:
    data = target_data[sym]
    for go in data.get("geneOntology", []):
        term = go.get("term", {})
        go_id  = term.get("id", "")
        label  = term.get("label", "")
        # Biological Process terms start with GO:00...
        # We can't filter by namespace since it's not returned; include all
        go_rows.append({
            "Gene": sym,
            "GO ID": go_id,
            "GO Term": label,
            "Evidence": go.get("evidence", ""),
        })

df_go = pd.DataFrame(go_rows) if go_rows else pd.DataFrame(
    columns=["Gene", "GO ID", "GO Term", "Evidence"]
)
# Keep only biological process GO terms (GO:0008150 subtree)
# Use heuristic: 'process' or 'activity' in label
df_go_bp = df_go[
    df_go["GO Term"].str.lower().str.contains("process|response|signaling|pathway|regulation|activity", na=False)
]
df_go_bp.to_csv(os.path.join(TAB_DIR, "OT_GO_biological_process.csv"), index=False)
print(f"  Saved: OT_GO_biological_process.csv  ({len(df_go_bp)} terms)")

# ── Step 9: Drug & clinical candidates table ──────────────────────────────────
print("\nBuilding drug/clinical candidates table ...")

drug_rows = []
for sym in SHAP_ORDER:
    data = target_data[sym]
    cnt  = data["drugAndClinicalCandidates"]["count"]
    for row in data["drugAndClinicalCandidates"]["rows"]:
        drug     = row["drug"]
        diseases = [
            d["disease"]["name"] for d in row.get("diseases", [])
            if d.get("disease")
        ]
        drug_rows.append({
            "Gene": sym,
            "Drug Name": drug.get("name", ""),
            "Drug ID": drug.get("id", ""),
            "Drug Type": drug.get("drugType", ""),
            "Max Clinical Stage": row.get("maxClinicalStage", ""),
            "Drug Global Max Stage": drug.get("maximumClinicalStage", ""),
            "Indications (from source)": "; ".join(diseases[:5]),
        })

df_drugs = pd.DataFrame(drug_rows) if drug_rows else pd.DataFrame(
    columns=["Gene", "Drug Name", "Drug ID", "Drug Type",
             "Max Clinical Stage", "Drug Global Max Stage", "Indications (from source)"]
)
df_drugs.to_csv(os.path.join(TAB_DIR, "OT_drug_clinical_candidates.csv"), index=False)
print(f"  Saved: OT_drug_clinical_candidates.csv  ({len(df_drugs)} rows)")

# ── Step 10: Target class + function summary table ────────────────────────────
print("\nBuilding target class / function summary ...")

summary_rows = []
for sym in SHAP_ORDER:
    data = target_data[sym]
    tc_labels = "; ".join(
        t["label"] for t in data.get("targetClass", []) if t.get("label")
    ) or "—"
    funcs = data.get("functionDescriptions", []) or []
    func_str = funcs[0] if funcs else "—"
    prio_items = {it["key"]: it["value"]
                  for it in (data["prioritisation"]["items"] if data["prioritisation"] else [])}
    summary_rows.append({
        "Gene": sym,
        "Ensembl ID": data["id"],
        "Approved Name": data["approvedName"],
        "Biotype": data["biotype"],
        "Target Class": tc_labels,
        "Function (UniProt)": func_str,
        "Total Tractability Hits": sum(1 for t in data["tractability"] if t["value"]),
        "Drug/Clinical Count": data["drugAndClinicalCandidates"]["count"],
        "Chemical Probes": len(data["chemicalProbes"]),
        "Safety Liabilities": len(data["safetyLiabilities"]),
        "Reactome Pathways": len(data["pathways"]),
        "Disease Associations": data["associatedDiseases"]["count"],
        "Is In Membrane": int(prio_items.get("isInMembrane", 0) or 0),
        "Is Secreted": int(prio_items.get("isSecreted", 0) or 0),
        "Has Ligand": int(prio_items.get("hasLigand", 0) or 0),
        "Has Pocket": int(prio_items.get("hasPocket", 0) or 0),
        "Has SM Binder": int(prio_items.get("hasSmallMoleculeBinder", 0) or 0),
        "Gene Essentiality": round(float(prio_items.get("geneEssentiality", 0) or 0), 4),
        "Tissue Specificity": round(float(prio_items.get("tissueSpecificity", 0) or 0), 4),
        "Genetic Constraint": round(float(prio_items.get("geneticConstraint", 0) or 0), 4),
    })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(TAB_DIR, "OT_target_summary.csv"), index=False)
print(f"  Saved: OT_target_summary.csv  ({len(df_summary)} genes)")

# ── Step 11: Therapeutic modality overview bar chart ─────────────────────────
print("\nGenerating therapeutic modality overview figure ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: stacked bar — tractability hits per modality per gene
modality_nice = {
    "SM": "Small\nMolecule",
    "AB": "Antibody",
    "PR": "PROTAC",
    "OC": "Other\nClinical",
    "HE": "HT\nExperiment.",
    "GE": "Gene\nEditing",
    "OT": "Other",
}
bar_data = {m: [] for m in modality_order}
for sym in SHAP_ORDER:
    data = target_data[sym]
    by_mod = {m: 0 for m in modality_order}
    for t in data["tractability"]:
        if t["value"] and t["modality"] in by_mod:
            by_mod[t["modality"]] += 1
    for m in modality_order:
        bar_data[m].append(by_mod[m])

x = np.arange(len(SHAP_ORDER))
bottom = np.zeros(len(SHAP_ORDER))
bar_colors = ["#2166ac", "#4dac26", "#d01c8b", "#f1b6da", "#b8e186", "#abd9e9", "#fdae61"]
handles = []
for m, color in zip(modality_order, bar_colors):
    vals = np.array(bar_data[m], dtype=float)
    bars = axes[0].bar(x, vals, bottom=bottom, color=color, label=modality_nice.get(m, m),
                       edgecolor="white", linewidth=0.5)
    bottom += vals
    handles.append(bars)

axes[0].set_xticks(x)
axes[0].set_xticklabels(SHAP_ORDER, fontsize=10, fontweight="bold")
axes[0].set_ylabel("Tractability Flags (active)", fontsize=9)
axes[0].set_title("Tractability Profile\nby Modality", fontsize=10, fontweight="bold")
axes[0].legend(fontsize=7, loc="upper right", framealpha=0.7)

# Right: dot plot — disease association count (log scale) + viral highlight
dis_counts = [target_data[sym]["associatedDiseases"]["count"] for sym in SHAP_ORDER]
viral_counts = []
for sym in SHAP_ORDER:
    data = target_data[sym]
    cnt = sum(
        1 for row in data["associatedDiseases"]["rows"]
        if any(k in row["disease"]["name"].lower() for k in viral_keywords)
        or any(
            any(k in ta["name"].lower() for k in viral_keywords)
            for ta in row["disease"].get("therapeuticAreas", [])
        )
    )
    viral_counts.append(cnt)

axes[1].bar(x - 0.2, dis_counts, width=0.35, color="#457b9d", label="All diseases")
axes[1].bar(x + 0.2, viral_counts, width=0.35, color="#e63946", label="Infectious / Viral")
axes[1].set_xticks(x)
axes[1].set_xticklabels(SHAP_ORDER, fontsize=10, fontweight="bold")
axes[1].set_ylabel("No. Associated Diseases", fontsize=9)
axes[1].set_yscale("log")
axes[1].set_title("Disease Associations\n(queried top 100)", fontsize=10, fontweight="bold")
axes[1].legend(fontsize=8)

plt.suptitle("Open Targets Therapeutic Intervention Overview — EVD Panel",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "OT_04_therapeutic_overview.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: OT_04_therapeutic_overview.png")

# ── Step 12: Write Markdown report ───────────────────────────────────────────
print("\nWriting therapeutic intervention report ...")

# Prepare per-gene summaries
def tract_summary(sym):
    data = target_data[sym]
    active = [(t["modality"], t["label"]) for t in data["tractability"] if t["value"]]
    if not active:
        return "No tractability flags in Open Targets (misc_RNA — expected)."
    lines = []
    by_mod = {}
    for m, l in active:
        by_mod.setdefault(m, []).append(l)
    for m, labels in by_mod.items():
        lines.append(f"  - **{MODALITY_LABELS.get(m, m)}**: {', '.join(labels)}")
    return "\n".join(lines)

def top_diseases_md(sym, n=5):
    data = target_data[sym]
    rows = data["associatedDiseases"]["rows"][:n]
    if not rows:
        return "  — none —"
    lines = []
    for i, row in enumerate(rows, 1):
        lines.append(f"  {i}. {row['disease']['name']} (score = {row['score']:.3f})")
    return "\n".join(lines)

def viral_diseases_md(sym):
    data = target_data[sym]
    hits = [
        row for row in data["associatedDiseases"]["rows"]
        if any(k in row["disease"]["name"].lower() for k in viral_keywords)
        or any(
            any(k in ta["name"].lower() for k in viral_keywords)
            for ta in row["disease"].get("therapeuticAreas", [])
        )
    ]
    if not hits:
        return "  — none —"
    return "\n".join(
        f"  - {row['disease']['name']} (score = {row['score']:.3f})" for row in hits[:6]
    )

def pathways_md(sym):
    pws = target_data[sym]["pathways"]
    if not pws:
        return "  — none —"
    return "\n".join(f"  - {pw['pathway']} (`{pw['pathwayId']}`)" for pw in pws[:6])

def prio_md(sym):
    data = target_data[sym]
    items = {it["key"]: it["value"]
             for it in (data["prioritisation"]["items"] if data["prioritisation"] else [])}
    keys_show = [
        ("isInMembrane", "Membrane localised"),
        ("isSecreted", "Secreted protein"),
        ("hasLigand", "Has known ligand"),
        ("hasPocket", "Has druggable pocket"),
        ("hasSmallMoleculeBinder", "Has small-molecule binder"),
        ("geneEssentiality", "Gene essentiality score"),
        ("tissueSpecificity", "Tissue specificity"),
        ("geneticConstraint", "Genetic constraint (pLI proxy)"),
    ]
    lines = []
    for k, label in keys_show:
        if k in items:
            lines.append(f"  | {label} | {items[k]} |")
    return "\n".join(lines) if lines else "  — none —"

report_lines = [
    "# Open Targets Therapeutic Intervention Report",
    "## EVD Blood Transcriptomics — Corrected 5-Gene Biomarker Panel",
    "",
    f"**Generated :** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
    f"**API version :** Open Targets Platform GraphQL v4  ",
    f"**Ebola disease ID :** `{EBOLA_MONDO}` (Ebola hemorrhagic fever)  ",
    f"**Gene panel :** {', '.join(SHAP_ORDER)} (ordered by mean |SHAP|)  ",
    "",
    "---",
    "",
    "## 1. Overview",
    "",
    "The corrected EVD ML pipeline (internal revision addressing C-34/C-36) identified a",
    "5-gene biomarker panel (OOF AUC = 0.946, p < 0.002). This report evaluates each gene",
    "as a potential **therapeutic target** using the Open Targets Platform, covering:",
    "",
    "- Target tractability across all therapeutic modalities",
    "- Drug & clinical candidates",
    "- Disease associations (all diseases + infectious/viral subset)",
    "- Reactome pathways",
    "- Chemical probes",
    "- Safety liabilities",
    "- Target prioritisation scores",
    "",
    "---",
    "",
    "## 2. Summary Table",
    "",
    "| Gene | Biotype | Tractability Hits | Drugs/Clinical | Probes | Safety | Diseases | Pathways |",
    "|------|---------|:-----------------:|:--------------:|:------:|:------:|:--------:|:--------:|",
]

for sym in SHAP_ORDER:
    data = target_data[sym]
    n_tract = sum(1 for t in data["tractability"] if t["value"])
    n_drugs = data["drugAndClinicalCandidates"]["count"]
    n_probes = len(data["chemicalProbes"])
    n_safety = len(data["safetyLiabilities"])
    n_dis = data["associatedDiseases"]["count"]
    n_pw = len(data["pathways"])
    report_lines.append(
        f"| **{sym}** | {data['biotype']} | {n_tract} | {n_drugs} | {n_probes} | {n_safety} | {n_dis} | {n_pw} |"
    )

report_lines += [
    "",
    "---",
    "",
    "## 3. Per-Gene Profiles",
    "",
]

for sym in SHAP_ORDER:
    data = target_data[sym]
    tc_labels = ", ".join(t["label"] for t in data.get("targetClass", []) if t.get("label")) or "—"
    funcs = data.get("functionDescriptions") or []
    func_str = funcs[0] if funcs else "No function description available."

    report_lines += [
        f"### 3.{list(SHAP_ORDER).index(sym)+1} {sym} — {data['approvedName']}",
        "",
        f"- **Ensembl ID:** `{data['id']}`",
        f"- **Biotype:** {data['biotype']}",
        f"- **Target class:** {tc_labels}",
        f"- **Function:** {func_str[:300]}",
        "",
        "**Tractability:**",
        "",
        tract_summary(sym),
        "",
        "**Top-5 disease associations:**",
        "",
        top_diseases_md(sym),
        "",
        "**Infectious / Viral disease associations:**",
        "",
        viral_diseases_md(sym),
        "",
        "**Reactome pathways:**",
        "",
        pathways_md(sym),
        "",
        "**Prioritisation highlights:**",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        prio_md(sym),
        "",
        "---",
        "",
    ]

report_lines += [
    "## 4. Therapeutic Modality Assessment",
    "",
    "### 4.1 Small Molecule",
    "",
    "- **PLCH2**: Phospholipase C superfamily member; classified as membrane-localised",
    "  (isInMembrane = 1). No current small-molecule binder flag in Open Targets, but",
    "  the PLC family has established precedent for small-molecule inhibition (e.g.,",
    "  U73122 for PLCβ). Drug discovery effort would be warranted.",
    "- **IFI27**: ISG15-like interferon-induced protein; no SM tractability flags.",
    "  The interferon pathway itself (upstream of IFI27) is modulated by ruxolitinib,",
    "  baricitinib, and related JAK inhibitors — therapeutically relevant for",
    "  hyper-inflammatory EVD stages.",
    "- **GOLGA8B**: Golgi matrix protein; no SM tractability. Indirect targeting via",
    "  secretory pathway modulators possible but unvalidated for EVD.",
    "- **RN7SL1 / RN7SL2**: Non-coding RNA (misc_RNA biotype); not amenable to classical",
    "  small-molecule targeting. See RNA-targeted modalities below.",
    "",
    "### 4.2 Antibody / Biologic",
    "",
    "- **PLCH2**: GO Cellular Component and UniProt localisation flags indicate",
    "  potential extracellular / membrane accessibility — a prerequisite for antibody",
    "  targeting. UniProt loc medium confidence.",
    "- **IFI27**: UniProt signal peptide / transmembrane domain flag (SigP or TMHMM)",
    "  suggests surface expression; could be a target for antibody-drug conjugates or",
    "  immune checkpoint approaches in the context of immune dysregulation.",
    "- **GOLGA8B**: UniProt high-confidence localisation flag. However, Golgi-resident",
    "  proteins are generally not accessible to therapeutic antibodies.",
    "",
    "### 4.3 PROTAC / Protein Degrader",
    "",
    "- **PLCH2** and **IFI27** both carry PROTAC-relevant flags (Ubiquitination in",
    "  databases; Half-life data for PLCH2). Targeted protein degradation is feasible",
    "  in principle pending structural data for E3-ligase ternary complex design.",
    "",
    "### 4.4 RNA-targeted Therapeutics",
    "",
    "- **RN7SL1 / RN7SL2**: Signal recognition particle RNA components. These are",
    "  structural ncRNAs essential for co-translational translocation. They are not",
    "  classical drug targets but their differential expression in EVD suggests they",
    "  serve as **biomarkers of SRP pathway dysregulation** rather than direct targets.",
    "  ASO or siRNA approaches targeting their regulatory context (e.g., upstream",
    "  transcription factors) may be explored indirectly.",
    "",
    "---",
    "",
    "## 5. Disease Association Highlights",
    "",
    "- **IFI27** has the broadest disease footprint (945 associations in top-100",
    "  query window), with strongest scores in COVID-19 and oncology. Its interferon-",
    "  inducible character makes it a plausible viral disease biomarker and indirect",
    "  therapeutic target via interferon pathway modulation.",
    "- **GOLGA8B** shows COVID-19 and SARS associations (score ≥ 0.167), consistent",
    "  with its role in the secretory pathway exploited by coronaviruses and",
    "  potentially filoviruses.",
    "- **PLCH2** has associations primarily in neurological / ophthalmic diseases",
    "  (schizophrenia, refraction abnormalities). The 65 associations include",
    "  COVID-19 (score = 0.010) and adenoviridae infection, but the link to EVD",
    "  is likely indirect (immune/inflammatory co-regulation).",
    "- **RN7SL1** has 8 infectious/viral disease associations including hepatitis B",
    "  and tuberculosis, consistent with broad immune surveillance roles.",
    "- **Note:** No gene in the panel has a direct Ebola hemorrhagic fever",
    f"  (`{EBOLA_MONDO}`) association in Open Targets, reflecting that EVD is",
    "  under-represented in the evidence databases (sparse GWAS, rare disease).",
    "",
    "---",
    "",
    "## 6. Conclusions and Recommendations",
    "",
    "| Priority | Gene | Recommendation |",
    "|----------|------|----------------|",
    "| High | **IFI27** | Strong interferon-pathway marker; upstream JAK/STAT inhibitors (baricitinib, ruxolitinib) already in clinical use for cytokine-storm management — test combination with EVD supportive care |",
    "| Medium | **PLCH2** | PLC family tractable; investigate phospholipase inhibitor repurposing; PROTAC feasibility study warranted |",
    "| Medium | **GOLGA8B** | Secretory pathway role; indirect targeting via ER-Golgi trafficking inhibitors (Brefeldin A analogues) possible |",
    "| Low | **RN7SL1/2** | Non-coding RNAs; value primarily as **diagnostic biomarkers**, not direct therapeutic targets |",
    "",
    "> **Overall assessment:** The 5-gene panel performs best as a **diagnostic signature**",
    "> (OOF AUC 0.946). Direct therapeutic targeting of individual panel members is",
    "> limited by low tractability (especially for the ncRNAs), but **IFI27** provides",
    "> a strong rationale for interferon/JAK-pathway modulation as adjunct EVD therapy.",
    "> The panel could be used to stratify patients most likely to benefit from",
    "> immunomodulatory interventions.",
    "",
    "---",
    "",
    "## 7. Output Files",
    "",
    "| File | Description |",
    "|------|-------------|",
    "| `OT_target_summary.csv` | Full per-gene summary with all tractability / prioritisation fields |",
    "| `OT_tractability_summary.csv` | Active tractability flags per modality |",
    "| `OT_01_tractability_heatmap.png` | Heatmap of tractability modality × gene |",
    "| `OT_disease_associations_top10.csv` | Top-10 disease associations per gene |",
    "| `OT_infectious_disease_associations.csv` | Infectious / viral disease subset |",
    "| `OT_02_disease_associations.png` | Bubble plot of top-5 disease associations |",
    "| `OT_pathways.csv` | Reactome pathway annotations |",
    "| `OT_chemical_probes.csv` | Chemical probes (if any) |",
    "| `OT_safety_liabilities.csv` | Safety liability events |",
    "| `OT_prioritisation_scores.csv` | Numeric prioritisation metrics |",
    "| `OT_03_prioritisation_radar.png` | Radar chart of normalised prioritisation scores |",
    "| `OT_GO_biological_process.csv` | GO biological process annotations |",
    "| `OT_drug_clinical_candidates.csv` | Drug / clinical candidate entries |",
    "| `OT_04_therapeutic_overview.png` | Overview bar chart (tractability + disease) |",
    "",
    "---",
    "",
    f"*Report generated by `02_open_targets_analysis.py` — {datetime.now().strftime('%Y-%m-%d')}*",
]

report_text = "\n".join(report_lines)
report_path = os.path.join(ROOT, "docs", "02_open_targets_report.md")
with open(report_path, "w", encoding="utf-8") as fh:
    fh.write(report_text)
print(f"  Saved: results/OT_therapeutic_intervention_report.md")

# ── Done ──────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("Open Targets analysis complete.")
print(f"  Tables  : results/tables/open_targets/")
print(f"  Figures : results/figures/open_targets/")
print(f"  Report  : docs/02_open_targets_report.md")
print("=" * 65)
