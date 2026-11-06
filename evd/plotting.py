"""Shared figure style. Two-class palette (Okabe-Ito blue / vermillion) passes the dataviz
validator (CVD min dE 21.9, contrast >= 3:1 on white) and is always paired with direct labels
or marker shape, never colour alone."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HEALTHY = "#0072B2"
EVD = "#D55E00"
INK = "#222222"
MUTED = "#6b6b6b"
GRID = "#e4e4e4"


def style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "regular", "axes.labelsize": 9,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.labelcolor": INK,
        "legend.frameon": False, "lines.linewidth": 1.6, "lines.markersize": 5,
    })


def save(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
