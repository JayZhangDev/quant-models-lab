"""One shared look for every figure, plus helpers to save figures and tables.

Keeping the style in one place means all charts in ``results/`` look like they belong
to the same report, and changing a colour is a one-line edit.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only; no window pops up, works on servers and CI

import matplotlib.pyplot as plt  # noqa: E402  (must come after choosing the backend)
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from qml.common import RESULTS_DIR  # noqa: E402

# Series colours, used in this fixed order. The order was chosen so neighbouring
# colours stay distinguishable for colour-blind readers.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#6250d6"
RED = "#e34948"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

# "Ink" and chrome: text is near-black, grids and axes are quiet so the data stands out.
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

# One-hue ramp for "how much" (light = small, dark = large).
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "qml_blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
# Two-hue ramp for "which side of zero": blue = negative, grey = nothing, red = positive.
DIVERGING = LinearSegmentedColormap.from_list(
    "qml_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ec835a", "#b22a2a"]
)

DPI = 150


def apply_style() -> None:
    """Set the global matplotlib style used by every experiment.

    Call this once at the top of a script, before creating any figure.
    """
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 10,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 1.8,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "axes.prop_cycle": plt.cycler(color=SERIES),
            "figure.constrained_layout.use": True,
        }
    )


def save_fig(fig: Figure, name: str, results_dir: Path | str = RESULTS_DIR) -> Path:
    """Save a figure as ``results/<name>.png`` and close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    name : str
        File name without extension, e.g. ``"rmt_eigenvalues"``.
    results_dir : Path or str, optional
        Output folder. Defaults to ``results/``.

    Returns
    -------
    Path
        Where the file was written.
    """
    # Closing the figure after saving frees its memory; scripts that make many
    # figures would otherwise keep all of them alive.
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def to_markdown(df: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    """Format a DataFrame as a GitHub-flavoured Markdown table.

    Parameters
    ----------
    df : pd.DataFrame
        Table to format. The index is not included; call ``reset_index()`` first
        if you want it as a column.
    float_format : str, optional
        Format string applied to every float cell.

    Returns
    -------
    str
        The Markdown table, ending with a newline.
    """

    # Written by hand so the project does not need the extra `tabulate` dependency
    # that pandas' own to_markdown() requires.
    def fmt(value: object) -> str:
        return float_format.format(value) if isinstance(value, float) else str(value)

    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, divider, *rows]) + "\n"


def save_table(
    df: pd.DataFrame,
    name: str,
    float_format: str = "{:.4f}",
    results_dir: Path | str = RESULTS_DIR,
) -> Path:
    """Save a table twice: ``<name>.csv`` for machines and ``<name>.md`` for people.

    Parameters
    ----------
    df : pd.DataFrame
        Table to save (index is dropped).
    name : str
        File name without extension.
    float_format : str, optional
        Format string for floats in the Markdown copy. The CSV keeps full precision.
    results_dir : Path or str, optional
        Output folder. Defaults to ``results/``.

    Returns
    -------
    Path
        Path of the CSV file.
    """
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False, lineterminator="\n")
    (out_dir / f"{name}.md").write_text(to_markdown(df, float_format), encoding="utf-8")
    return csv_path
