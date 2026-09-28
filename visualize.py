from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive — required for script/headless use

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from config import FIGURE_DPI, FIGURES_DIR

logger = logging.getLogger(__name__)


FONT_TITLE  = 34
FONT_LABEL  = 28
FONT_TICK   = 24
FONT_LEGEND = 24
FONT_ANNOT  = 20

# Colour palette — high-contrast, colour-blind-friendly (Okabe-Ito derived)
CONDITION_COLOURS = {
    "Baseline-0 (Identity)":   "#8B949E",   # neutral grey
    "Condition-A (Zero-Shot)": "#E69F00",   # amber
    "Condition-B (Few-Shot)":  "#009E73",   # teal
}

CONDITION_LABELS = {
    "Baseline-0 (Identity)":   "Baseline-0\n(Identity)",
    "Condition-A (Zero-Shot)": "Condition-A\n(Zero-Shot)",
    "Condition-B (Few-Shot)":  "Condition-B\n(Few-Shot)",
}


def _poster_theme() -> None:
    """Apply dark, high-contrast poster theme."""
    mpl.rcParams.update({
        "figure.facecolor":  "#0D1117",
        "axes.facecolor":    "#161B22",
        "axes.edgecolor":    "#30363D",
        "axes.labelcolor":   "#E6EDF3",
        "axes.titlecolor":   "#E6EDF3",
        "xtick.color":       "#8B949E",
        "ytick.color":       "#8B949E",
        "text.color":        "#E6EDF3",
        "grid.color":        "#21262D",
        "grid.linewidth":    0.8,
        "legend.facecolor":  "#161B22",
        "legend.edgecolor":  "#30363D",
        "font.family":       "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


# Figure 1 — WER by condition 


def plot_wer_by_condition(
    summary_df: pd.DataFrame,
    output_path: str | Path | None = None,
    dpi: int = FIGURE_DPI,
) -> Path:
    """
    Bar chart comparing WER and BLEU across the 3 experimental tiers.

    Parameters
    ----------
    summary_df : DataFrame with columns: condition, wer, bleu, mean_fluency
    """
    _poster_theme()
    output_path = Path(output_path or Path(FIGURES_DIR) / "fig1_wer_by_condition.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conditions = summary_df["condition"].tolist()
    wer_vals   = (summary_df["wer"].values * 100)   # convert to %
    bleu_vals  = summary_df["bleu"].values
    colours    = [CONDITION_COLOURS.get(c, "#56B4E9") for c in conditions]
    x_labels   = [CONDITION_LABELS.get(c, c) for c in conditions]
    x          = np.arange(len(conditions))
    width      = 0.35

    fig, ax1 = plt.subplots(figsize=(18, 11), dpi=dpi)

    # WER bars (left axis)
    bars_wer = ax1.bar(
        x - width / 2, wer_vals, width,
        color=colours, alpha=0.9, linewidth=0, zorder=3,
        label="WER (%)",
    )

    # BLEU bars (left axis, scaled differently — label them separately)
    bars_bleu = ax1.bar(
        x + width / 2, bleu_vals, width,
        color=[c + "99" for c in ["#8B949E", "#E69F00", "#009E73"]],
        alpha=0.7, linewidth=0, zorder=3, hatch="///",
        label="BLEU Score",
    )

    # Value annotations
    for bar, val in zip(bars_wer, wer_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.1f}%",
            ha="center", va="bottom",
            fontsize=FONT_ANNOT, color="#E6EDF3", fontweight="bold",
        )
    for bar, val in zip(bars_bleu, bleu_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.1f}",
            ha="center", va="bottom",
            fontsize=FONT_ANNOT, color="#E6EDF3",
        )

    # WER reduction arrows between bars
    prev_wer = None
    for i, val in enumerate(wer_vals):
        if prev_wer is not None:
            reduction = prev_wer - val
            if reduction > 0:
                ax1.annotate(
                    "",
                    xy=(x[i] - width / 2, val + 2),
                    xytext=(x[i - 1] - width / 2, prev_wer + 2),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="#FF6B6B",
                        lw=2.5,
                    ),
                )
                mid_x = (x[i - 1] + x[i]) / 2 - width / 2
                mid_y = (prev_wer + val) / 2 + 5
                ax1.text(
                    mid_x, mid_y,
                    f"-{reduction:.1f}%",
                    ha="center", color="#FF6B6B",
                    fontsize=FONT_ANNOT, fontweight="bold",
                )
        prev_wer = val

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=FONT_TICK)
    ax1.set_ylabel("Score (%)", fontsize=FONT_LABEL, labelpad=12)
    ax1.set_xlabel("Experimental Condition", fontsize=FONT_LABEL, labelpad=12)
    ax1.set_title(
        "WER & BLEU Across 3 Experimental Conditions\n"
        "(WER = bars, lower is better | BLEU = hatched, higher is better)",
        fontsize=FONT_TITLE, fontweight="bold", pad=20,
    )
    ax1.set_ylim(0, max(wer_vals) * 1.25)
    ax1.yaxis.grid(True, zorder=0)
    ax1.legend(fontsize=FONT_LEGEND, loc="upper right", framealpha=0.6)

    fig.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Fig 1 saved -> %s  (DPI=%d)", output_path, dpi)
    return output_path



# Figure 2 — Per-region WER heatmap


def plot_region_heatmap(
    region_df: pd.DataFrame,
    metric: str = "wer",
    output_path: str | Path | None = None,
    dpi: int = FIGURE_DPI,
) -> Path:
    """Heatmap of WER (or other metric) per region × condition."""
    _poster_theme()
    output_path = Path(output_path or Path(FIGURES_DIR) / "fig2_region_heatmap.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

   
    short_labels = {
        "Baseline-0 (Identity)":   "Baseline-0",
        "Condition-A (Zero-Shot)": "Zero-Shot",
        "Condition-B (Few-Shot)":  "Few-Shot",
    }
    region_df = region_df.copy()
    short_regions = {
        "mittelfranken": "MFR",
        "niederbayern": "NBY",
        "oberbayern": "OBY",
        "oberfranken": "OFR",
        "oberpfalz": "OPF",
        "schwaben": "SWB",
        "unterfranken": "UFR",
    }
    region_df["region"] = region_df["region"].map(lambda r: short_regions.get(r, r))
    region_df["condition"] = region_df["condition"].map(
        lambda c: short_labels.get(c, c)
    )

    pivot = region_df.pivot(index="region", columns="condition", values=metric)
    # Order columns logically
    col_order = ["Baseline-0", "Zero-Shot", "Few-Shot"]
    pivot = pivot[[c for c in col_order if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(16, 11), dpi=dpi)
    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".3f",
        annot_kws={"size": FONT_ANNOT},
        cmap="RdYlGn_r",        # red=high WER (bad), green=low WER (good)
        linewidths=0.5,
        linecolor="#0D1117",
        cbar_kws={"label": metric.upper()},
        vmin=0,
        vmax=1,
    )
    ax.set_title(
        f"{metric.upper()} by Dialect Region x Condition\n"
        "(lower = better normalisation)",
        fontsize=FONT_TITLE, fontweight="bold", pad=20,
    )
    ax.set_xlabel("Condition", fontsize=FONT_LABEL)
    ax.set_ylabel("Dialect Region", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=FONT_TICK)
    cbar.set_label(metric.upper(), fontsize=FONT_LABEL)

    fig.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Fig 2 saved -> %s  (DPI=%d)", output_path, dpi)
    return output_path



# Figure 3 — Radar: all metrics across 3 conditions


def plot_radar(
    summary_df: pd.DataFrame,
    output_path: str | Path | None = None,
    dpi: int = FIGURE_DPI,
) -> Path:
    """
    Radar chart: BLEU, 1-WER, 1-CER, Fluency, Meaning Pres. for all 3 conditions.
    All axes normalised to [0, 1], higher = better.
    """
    _poster_theme()
    output_path = Path(output_path or Path(FIGURES_DIR) / "fig3_radar.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_labels = ["BLEU", "1-WER", "1-CER", "Fluency", "Meaning\nPres."]
    N = len(metrics_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    colours = ["#8B949E", "#E69F00", "#009E73"]

    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw={"polar": True}, dpi=dpi)
    ax.set_facecolor("#161B22")
    fig.patch.set_facecolor("#0D1117")

    def _normalise(row: pd.Series) -> list[float]:
        bleu_n = row.get("bleu", 0) / 100.0
        wer_n  = 1.0 - min(row.get("wer", 1), 1.0)
        cer_n  = 1.0 - min(row.get("cer", 1), 1.0)
        fl_n   = (row.get("mean_fluency", 0) - 1) / 4.0 if row.get("mean_fluency", 0) > 0 else 0
        mp_n   = (row.get("mean_meaning_preservation", 0) - 1) / 4.0 if row.get("mean_meaning_preservation", 0) > 0 else 0
        return [bleu_n, wer_n, cer_n, fl_n, mp_n]

    for (_, row), colour in zip(summary_df.iterrows(), colours):
        values = _normalise(row) + [_normalise(row)[0]]
        label  = CONDITION_LABELS.get(row["condition"], row["condition"]).replace("\n", " ")
        ax.plot(angles, values, "o-", linewidth=3, color=colour, label=label)
        ax.fill(angles, values, alpha=0.18, color=colour)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_labels, fontsize=FONT_TICK, color="#E6EDF3")
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                       fontsize=FONT_TICK - 4, color="#8B949E")
    ax.grid(color="#21262D", linewidth=0.8)
    ax.spines["polar"].set_color("#30363D")
    ax.set_title(
        "Full Metric Comparison Across All 3 Conditions\n(normalised to 0-1, higher = better)",
        fontsize=FONT_TITLE, fontweight="bold", pad=30, color="#E6EDF3",
    )
    ax.legend(fontsize=FONT_LEGEND, loc="lower right",
              bbox_to_anchor=(1.4, -0.05), framealpha=0.6)

    fig.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Fig 3 saved -> %s  (DPI=%d)", output_path, dpi)
    return output_path



# Figure 4 — WER improvement waterfall 


def plot_wer_improvement(
    summary_df: pd.DataFrame,
    output_path: str | Path | None = None,
    dpi: int = FIGURE_DPI,
) -> Path:
    """
    Horizontal bar chart showing absolute WER reduction step-by-step:
      Baseline-0 -> Condition-A (zero-shot gain)
      Condition-A -> Condition-B (few-shot gain)
    """
    _poster_theme()
    output_path = Path(output_path or Path(FIGURES_DIR) / "fig4_wer_improvement.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wer_map = dict(zip(summary_df["condition"], summary_df["wer"] * 100))
    b0 = wer_map.get("Baseline-0 (Identity)", 0)
    ca = wer_map.get("Condition-A (Zero-Shot)", 0)
    cb = wer_map.get("Condition-B (Few-Shot)", 0)

    steps   = ["Identity\nFloor", "Zero-Shot\nGain", "Few-Shot\nICL Gain", "Few-Shot\nFinal WER"]
    values  = [b0, -(b0 - ca), -(ca - cb), cb]
    colours = ["#8B949E", "#E69F00", "#009E73", "#56B4E9"]

    fig, ax = plt.subplots(figsize=(18, 9), dpi=dpi)
    bars = ax.bar(steps, [abs(v) for v in values], color=colours,
                  alpha=0.88, linewidth=0, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{abs(val):.1f}%",
            ha="center", va="bottom",
            fontsize=FONT_ANNOT, color="#E6EDF3", fontweight="bold",
        )

    ax.set_ylabel("WER (%)", fontsize=FONT_LABEL, labelpad=12)
    ax.set_title(
        "WER Reduction from Identity Floor to Few-Shot LLM\n"
        "(absolute percentage points — key ICL finding)",
        fontsize=FONT_TITLE, fontweight="bold", pad=20,
    )
    ax.set_ylim(0, max(abs(v) for v in values) * 1.25)
    ax.yaxis.grid(True, zorder=0)
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    ax.tick_params(axis="y", labelsize=FONT_TICK)

    # Annotate total reduction
    total_reduction = b0 - cb
    ax.text(
        0.98, 0.95,
        f"Total WER reduction: {total_reduction:.1f}pp\n({total_reduction/b0*100:.0f}% relative)",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=FONT_ANNOT + 2,
        color="#009E73",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#161B22",
                  edgecolor="#009E73", alpha=0.8),
    )

    fig.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Fig 4 saved -> %s  (DPI=%d)", output_path, dpi)
    return output_path



# Demo with synthetic data  (run: python visualize.py)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    summary = pd.DataFrame([
        {"condition": "Baseline-0 (Identity)",   "bleu":  9.8, "wer": 0.571, "cer": 0.241,
         "mean_fluency": 1.4, "mean_meaning_preservation": 2.1},
        {"condition": "Condition-A (Zero-Shot)", "bleu": 28.4, "wer": 0.381, "cer": 0.163,
         "mean_fluency": 4.1, "mean_meaning_preservation": 4.0},
        {"condition": "Condition-B (Few-Shot)",  "bleu": 35.7, "wer": 0.291, "cer": 0.122,
         "mean_fluency": 4.4, "mean_meaning_preservation": 4.3},
    ])

    import numpy as np
    rng = np.random.default_rng(42)
    regions = ["mittelfranken","niederbayern","oberbayern","oberfranken","oberpfalz","schwaben","unterfranken"]
    region_rows = []
    for r in regions:
        region_rows += [
            {"region": r, "condition": "Baseline-0 (Identity)",   "wer": rng.uniform(0.50, 0.65), "cer": rng.uniform(0.20, 0.28), "bleu": rng.uniform(7, 14)},
            {"region": r, "condition": "Condition-A (Zero-Shot)", "wer": rng.uniform(0.33, 0.45), "cer": rng.uniform(0.13, 0.20), "bleu": rng.uniform(22, 34)},
            {"region": r, "condition": "Condition-B (Few-Shot)",  "wer": rng.uniform(0.24, 0.36), "cer": rng.uniform(0.10, 0.17), "bleu": rng.uniform(28, 40)},
        ]
    region_df = pd.DataFrame(region_rows)

    p1 = plot_wer_by_condition(summary)
    p2 = plot_region_heatmap(region_df)
    p3 = plot_radar(summary)
    p4 = plot_wer_improvement(summary)
    print(f"Figures:\n  {p1}\n  {p2}\n  {p3}\n  {p4}")
