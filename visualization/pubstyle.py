"""Publication style shared by all figures (scientific-visualization conventions: Okabe-Ito colorblind-safe
palette, 3.5 in single-column width, despined axes, 7-9 pt sans-serif text, vector PDF output)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OKABE_ITO = {"orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
             "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000", "grey": "#7f7f7f"}
METHOD_COLOR = {"zero_shot": OKABE_ITO["grey"], "transclip": OKABE_ITO["skyblue"], "stata": OKABE_ITO["blue"],
                "moon": OKABE_ITO["purple"], "count_shrinkage": OKABE_ITO["green"], "graft": OKABE_ITO["vermillion"],
                "graft_eb": OKABE_ITO["orange"]}
METHOD_NAME = {"zero_shot": "CLIP zero-shot", "transclip": "TransCLIP", "stata": "StatA", "moon": "MOON",
               "count_shrinkage": "Anchor-only control", "graft": "GRAFT", "graft_eb": "GRAFT + EB"}
SINGLE, DOUBLE = 3.5, 7.16   # inches (CVPR column / text width)


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.2,
        "errorbar.capsize": 1.5, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "axes.grid": False, "figure.dpi": 150})


def panel_label(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def watermark(fig, on=True):
    if on:
        fig.text(0.5, 0.5, "PLACEHOLDER DATA", ha="center", va="center", fontsize=20, color="#D55E00", alpha=0.15, rotation=25, weight="bold")
