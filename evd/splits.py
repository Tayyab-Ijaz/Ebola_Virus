"""CV split generation shared by Python code and the R ranking scripts."""
import numpy as np
from sklearn.model_selection import StratifiedKFold


def outer_splits(y, seed, n_splits=5):
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in skf.split(np.zeros(len(y)), y)]


def inner_splits(y_tr, seed, n_splits=3):
    """Splits over positions 0..len(y_tr)-1 of the outer-training fold."""
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    return [(a, b) for a, b in skf.split(np.zeros(len(y_tr)), y_tr)]


def splits_json(y, seed, outer_n=5, inner_n=3):
    """Serialisable description of all outer and inner splits (0-based sample indices)."""
    out = []
    for f, (tr, te) in enumerate(outer_splits(y, seed, outer_n), 1):
        inner = []
        for j, (a, b) in enumerate(inner_splits(y[tr], seed, inner_n), 1):
            inner.append({"j": j, "train": tr[a].tolist(), "val": tr[b].tolist()})
        out.append({"fold": f, "train": tr.tolist(), "test": te.tolist(), "inner": inner})
    return {"seed": seed, "outer": out}
