"""Render a concise presentation diagram from the supplied bootcamp slides.

The diagram distinguishes the Day 1 work from feature, tuning, and challenger
refinements. Model source links are embedded in the SVG and recorded in README.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as DrawingPath


OUT = Path(__file__).resolve().parent
WIDTH, HEIGHT, DPI = 3200, 1800, 160
BG = "#F6F7FB"
INK = "#202940"
BODY = "#4E5A70"
NAVY = "#20263F"
PINK = "#DC278B"
PURPLE = "#8255C8"
TEAL = "#117C80"
BORDER = "#DFE4EE"

plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
ax.axis("off")


def text(x, y, value, size=30, color=INK, weight="normal", **kwargs):
    """Place crisp editable text using pixel coordinates."""
    return ax.text(x, y, value, fontsize=size * 72 / DPI, color=color,
                   fontweight=weight, va="top", linespacing=1.4, zorder=5, **kwargs)


def box(x, y, w, h, face="white", edge=BORDER, radius=24, shadow=False,
        width=1.2, zorder=2):
    """Draw a panel, optionally with a subtle offset shadow."""
    if shadow:
        ax.add_patch(FancyBboxPatch((x, y + 9), w, h,
                     boxstyle=f"round,pad=0,rounding_size={radius}",
                     facecolor="#E8EBF3", edgecolor="none", zorder=zorder - 0.1))
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0,rounding_size={radius}",
                 facecolor=face, edgecolor=edge, linewidth=width, zorder=zorder))


def arrow(points, color=TEAL, dashed=False, zorder=1):
    """Draw a routed connector."""
    path = DrawingPath(points, [DrawingPath.MOVETO] + [DrawingPath.LINETO] * (len(points) - 1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=20,
                 linewidth=2.6, color=color, zorder=zorder,
                 linestyle=(0, (6, 5)) if dashed else "solid"))


def line(points, color=PURPLE, width=2.3, zorder=4):
    """Draw a pictogram stroke."""
    ax.plot(*zip(*points), color=color, linewidth=width, zorder=zorder,
            solid_capstyle="round")


def pill(x, y, w, label, face, color, size=23):
    """Draw a compact status or metric label."""
    box(x, y, w, 43, face, face, radius=12, zorder=3)
    text(x + w / 2, y + 8, label, size, color, "bold", ha="center")


def step(x, number, label):
    """Label each stage outside its card."""
    ax.add_patch(Circle((x + 18, 517), 18, facecolor=NAVY, zorder=3))
    text(x + 18, 503, number, 22, "white", "bold", ha="center")
    text(x + 53, 502, label, 25, BODY, "bold")


def tables(cx, cy):
    """Draw a small linked-table motif."""
    for x, y, color in [(cx - 142, cy - 70, PINK), (cx + 35, cy - 115, PURPLE),
                        (cx + 35, cy + 5, TEAL)]:
        box(x, y, 115, 83, "white", color, radius=10, zorder=3)
        line([(x + 6, y + 28), (x + 109, y + 28)], color)
        line([(x + 39, y + 33), (x + 39, y + 76)], color, 1.3)
        line([(x + 9, y + 54), (x + 106, y + 54)], color, 1.3)
    line([(cx - 27, cy - 29), (cx + 5, cy - 29), (cx + 5, cy - 74), (cx + 35, cy - 74)], PURPLE)
    line([(cx + 5, cy - 29), (cx + 5, cy + 46), (cx + 35, cy + 46)], TEAL)


def feature_icon(cx, cy, kind):
    """Use a consistent compact icon style for feature families."""
    box(cx - 36, cy - 36, 72, 72, "#EDE5F8", "#EDE5F8", radius=18, zorder=3)
    if kind == "activity":
        for dx, h in [(-19, 19), (0, 38), (19, 29)]:
            line([(cx + dx, cy + 20), (cx + dx, cy + 20 - h)], PURPLE, 5)
    elif kind == "cash":
        arrow([(cx - 22, cy - 11), (cx + 24, cy - 11)], PURPLE, zorder=4)
        arrow([(cx + 22, cy + 14), (cx - 24, cy + 14)], PINK, zorder=4)
    else:
        ax.add_patch(Circle((cx, cy - 12), 10, facecolor="none", edgecolor=PURPLE, linewidth=2, zorder=4))
        line([(cx - 21, cy + 23), (cx - 16, cy + 5), (cx + 16, cy + 5), (cx + 21, cy + 23)], PURPLE)


def check(cx, cy):
    """Mark an achievement from the supplied Day 1 slide."""
    ax.add_patch(Circle((cx, cy), 19, facecolor="#DFF1EA", zorder=4))
    line([(cx - 8, cy), (cx - 2, cy + 6), (cx + 10, cy - 7)], TEAL, 2.2)


# Compact subject line; no solution-architecture title block.
text(70, 53, "Synthetic data for overdraft detection", 49, INK, "bold")
text(3080, 65, "MULTI-TABLE  /  BUILD & REFINE", 25, PURPLE, "bold", ha="right")

# Target card anchors the whole use case.
box(70, 155, 455, 295, NAVY, NAVY, shadow=True)
text(105, 190, "TARGET LABEL", 24, "#D5B5F1", "bold")
text(105, 242, "Overdraft", 51, "white", "bold")
text(105, 313, "Negative balance", 31, "#E0E3EE")
pill(105, 375, 245, "DETECTION TASK", "#353D59", "#FFFFFF", 22)

# A visible tuning loop expands on the work started on Day 1.
box(645, 155, 2435, 295, "#F0EBF8", "#E0D5EF", shadow=True)
text(680, 183, "HYPERPARAMETER TUNING", 28, PURPLE, "bold")
pill(1170, 179, 240, "STARTED DAY 1", "#E2D4F0", PURPLE, 22)
for x in (1490, 2280):
    line([(x, 250), (x, 413)], "#D9CCE8", 1.3)
text(680, 252, "Guidance sweep", 35, INK, "bold")
text(680, 306, "5 strengths · reuse trained models", 28, BODY)
for i in range(5):
    ax.add_patch(Circle((701 + 135 * i, 389), 13,
                 facecolor=["#E2CEE9", "#CEA8DF", "#B987D4", "#9C68CB", "#8255C8"][i], zorder=4))
line([(701, 389), (1241, 389)], "#D7C5E7", 2, zorder=3)
text(1525, 252, "Clustering", 35, INK, "bold")
pill(1737, 251, 90, "NEXT", "#E2D4F0", PURPLE, 21)
text(1525, 312, "Cluster count · parent weighting", 28, BODY)
text(1525, 361, "Clustering method", 28, BODY)
text(2315, 252, "Training", 35, INK, "bold")
pill(2495, 251, 90, "NEXT", "#E2D4F0", PURPLE, 21)
text(2315, 312, "Learning rate · training steps", 28, BODY)
text(2315, 361, "Network size · diffusion steps", 28, BODY)

# The architecture: broad backbone plus the team's areas of refinement.
for x, n, label in [(70, "1", "SOURCE"), (645, "2", "GENERATE"),
                     (1480, "3", "ENRICH"), (2330, "4", "EVALUATE")]:
    step(x, n, label)
box(70, 560, 455, 675, shadow=True)
box(645, 560, 710, 675, NAVY, NAVY, shadow=True)
box(1480, 560, 725, 675, "#FCF9FF", "#DED0ED", shadow=True)
box(2330, 560, 750, 675, shadow=True)

# Source data and one compact Day 1 generation result.
text(105, 601, "Berka data", 39, INK, "bold")
tables(294, 829)
text(105, 986, "Customers · accounts", 29, INK)
text(105, 1036, "Transactions", 29, INK)
pill(105, 1142, 385, "LARGER SAMPLE TESTED", "#FBE8F2", PINK, 23)

# Challenger models are clearly distinguished from the current baseline.
text(680, 601, "Generator candidates", 37, "white", "bold")
box(680, 677, 640, 153, "#343D58", "#424B67", radius=17, zorder=3)
text(705, 696, "BASELINE", 22, "#9CE0D4", "bold")
text(705, 736, "ClavaDDPM", 40, "white", "bold")
text(705, 788, "Relational diffusion", 26, "#D5DAE7")
text(680, 863, "CHALLENGERS · PLANNED", 23, "#D5B5F1", "bold")
box(680, 909, 640, 115, "#2D344F", "#4A4568", radius=17, zorder=3)
text(705, 925, "REaLTabFormer", 34, "white", "bold", url="https://github.com/worldbank/REaLTabFormer")
text(705, 976, "Transformer-based generation", 26, "#D5DAE7")
box(680, 1051, 640, 115, "#2D344F", "#4A4568", radius=17, zorder=3)
text(705, 1067, "RelDiff", 34, "white", "bold", url="https://arxiv.org/abs/2506.00710")
text(705, 1118, "Graph-based diffusion", 26, "#D5DAE7")

# Candidate features around the overdraft label.
text(1515, 601, "Enrich overdraft context", 36, INK, "bold")
pill(1515, 662, 280, "NEXT REFINEMENT", "#EDE3F7", PURPLE, 23)
for cy, kind, title, desc in [
    (776, "activity", "Activity patterns", "Frequency · timing"),
    (929, "cash", "Cash-flow patterns", "Prior debit / credit totals"),
    (1082, "account", "Account context", "Account age · customer profile"),
]:
    feature_icon(1551, cy, kind)
    text(1610, cy - 33, title, 32, INK, "bold")
    text(1610, cy + 19, desc, 27, BODY)
text(1842, 1183, "Past information only", 25, PURPLE, ha="center")

# Make the transfer direction and key evaluation metrics visible.
text(2365, 601, "Evaluate transfer", 38, INK, "bold")
text(2365, 664, "REAL-TRAINED RANDOM FOREST", 24, TEAL, "bold")
pill(2365, 711, 312, "REAL HOLDOUT", "#E5F3F0", TEAL, 24)
pill(2700, 711, 345, "SYNTHETIC RUNS", "#E5F3F0", TEAL, 24)
text(2365, 805, "Detection metrics", 32, INK, "bold")
text(2365, 860, "Average precision · ROC-AUC", 29, TEAL)
text(2365, 908, "Precision · recall", 29, TEAL)
line([(2365, 975), (3045, 975)], BORDER, 1.3)
text(2365, 1010, "Fidelity & relationships", 32, INK, "bold")
text(2365, 1065, "Distributions · overdraft rates", 29, TEAL)
text(2365, 1113, "8 parent–child relationships", 29, TEAL)

# Data flow and shared real reference.
arrow([(525, 862), (645, 862)])
arrow([(1355, 862), (1480, 862)])
text(1417, 810, "synthetic", 22, TEAL, ha="center")
arrow([(2205, 862), (2330, 862)])
arrow([(296, 1235), (296, 1325), (1842, 1325), (1842, 1235)])
text(1070, 1351, "Real reference · shared feature definitions", 27, TEAL, ha="center")
# Comparison informs the next generator settings.
arrow([(2705, 560), (2705, 495), (2900, 495), (2900, 450)], PURPLE, dashed=True)
arrow([(1000, 450), (1000, 560)], PURPLE, dashed=True)
text(2590, 465, "compare & refine", 23, PURPLE, ha="right")

# Evidence from the user's Day 1 slide, kept separate from planned additions.
box(70, 1450, 3010, 203, "white", BORDER, shadow=True)
text(105, 1481, "BUILD DAY 1", 26, PINK, "bold")
text(105, 1528, "What we established", 25, BODY)
for x, title, lines in [
    (520, "Utility baseline", "Real holdout + two synthetic runs"),
    (1375, "Diagnostics added", "Feature & score distributions\\nOverdraft rates by category"),
    (2230, "Transaction graph analyzed", "Mostly internal ↔ external accounts"),
]:
    check(x, 1503)
    text(x + 37, 1480, title, 29, INK, "bold")
    text(x + 37, 1537, lines.replace("\\n", "\n"), 26, BODY)

text(70, 1716, "BEYOND BUILD DAYS", 25, PURPLE, "bold")
text(440, 1714, "Graph-based privacy metrics     /     Bank dataset adaptation     /     Research collaboration", 28, INK)

for extension in ("png", "svg"):
    fig.savefig(OUT / f"solution_architecture.{extension}", dpi=DPI, facecolor=BG)
plt.close(fig)
print(OUT / "solution_architecture.png")
