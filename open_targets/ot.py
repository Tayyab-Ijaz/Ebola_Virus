"""Open Targets annotation of the genes of the final model (exploratory annotation only).
usage: python ot.py <run_dir> <out.csv> ; the API/data version is stored in <out>.meta.json"""
import csv
import json
import sys
import time
from pathlib import Path

import requests

URL = "https://api.platform.opentargets.org/api/v4/graphql"
SEARCH = 'query($s:String!){ search(queryString:$s, entityNames:["target"], page:{index:0,size:5}){ hits{ id object{... on Target{ approvedSymbol }}}}}'
TARGET = ("query($id:String!){ target(ensemblId:$id){ id approvedSymbol biotype tractability{label modality value} "
          "associatedDiseases(page:{index:0,size:3}){count rows{score disease{name}}} drugAndClinicalCandidates{count} }}")


def gql(q, v=None, tries=4):
    for i in range(tries):
        try:
            r = requests.post(URL, json={"query": q, "variables": v or {}}, timeout=60)
            if r.status_code == 200 and "errors" not in r.json():
                return r.json()["data"]
        except requests.RequestException:
            pass
        time.sleep(2 * (i + 1))
    return None


def main(run_dir, out):
    run_dir = Path(run_dir)
    genes = [r["symbol"] for r in csv.DictReader(open(run_dir / "item3_stability/final_panel.csv", encoding="utf-8"))]
    meta = gql("{ meta { apiVersion { x y z } dataVersion { year month } } }")["meta"]
    rows = []
    for g in genes:
        row = {"symbol": g, "ensembl_id": "", "biotype": "", "tractability_modalities": "", "n_modalities_with_evidence": "",
               "n_associated_diseases": "", "top_disease": "", "top_disease_score": "", "drug_or_clinical_candidates": "", "status": "not found"}
        s = gql(SEARCH, {"s": g})
        hits = [h for h in (s or {}).get("search", {}).get("hits", []) if (h.get("object") or {}).get("approvedSymbol") == g]
        if hits:
            t = gql(TARGET, {"id": hits[0]["id"]})["target"]
            mods = sorted({x["modality"] for x in t["tractability"] if x["value"]})
            ad = t["associatedDiseases"]
            row.update(ensembl_id=t["id"], biotype=t["biotype"], tractability_modalities=";".join(mods), n_modalities_with_evidence=len(mods),
                       n_associated_diseases=ad["count"], top_disease=(ad["rows"][0]["disease"]["name"] if ad["rows"] else ""),
                       top_disease_score=(round(ad["rows"][0]["score"], 3) if ad["rows"] else ""),
                       drug_or_clinical_candidates=t["drugAndClinicalCandidates"]["count"] if t.get("drugAndClinicalCandidates") else 0, status="ok")
        rows.append(row)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    Path(str(out) + ".meta.json").write_text(json.dumps({"api": meta, "queried": time.strftime("%Y-%m-%d"), "n_genes": len(rows),
                                                          "n_found": sum(r["status"] == "ok" for r in rows)}, indent=1), encoding="utf-8")
    print("found", sum(r["status"] == "ok" for r in rows), "of", len(rows), meta)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
