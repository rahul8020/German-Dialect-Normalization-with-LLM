"""
evaluate.py — Main evaluation pipeline for the ICL dialect normalisation study.

Pipeline:
  1. Load data (example pool + test set)
  2. Baseline-0 : identity control    (no API calls)
  3. Condition-A : zero-shot Groq     (1 API call per sentence)
  4. Condition-B : few-shot Groq      (1 API call per sentence, richer prompt)
  5. Metrics     : WER / CER / BLEU   (local, no API)
  6. LLM Judge   : Groq scores        (optional, --skip-judge to omit)
  7. Figures     : 4 poster charts

Checkpointing: each condition's sentence-level results are saved to CSV
immediately after generation. If the run is interrupted, re-run with
--load-checkpoints to skip re-generating and jump to metrics/figures.

Usage
-----
    python evaluate.py                          # full pipeline, all 7 regions
    python evaluate.py --regions oberbayern     # single region (quick test)
    python evaluate.py --skip-judge             # skip LLM judge (saves API quota)
    python evaluate.py --load-checkpoints       # reload CSVs, skip generation
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from config import FIGURES_DIR, REGIONS, RESULTS_DIR
from conditions import run_identity, run_zero_shot, run_few_shot
from data_loader import load_all_regions
from metrics import (
    aggregate_judge,
    compute_metrics_from_df,
    compute_per_region,
    llm_judge,
)
from visualize import (
    plot_wer_by_condition,
    plot_region_heatmap,
    plot_radar,
    plot_wer_improvement,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")

CONDITION_FILES = {
    "Baseline-0 (Identity)":   "baseline0.csv",
    "Condition-A (Zero-Shot)": "condition_a.csv",
    "Condition-B (Few-Shot)":  "condition_b.csv",
}


def _ensure_dirs() -> None:
    for d in (RESULTS_DIR, FIGURES_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)


def _save(df: pd.DataFrame, filename: str) -> Path:
    path = Path(RESULTS_DIR) / filename
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Saved -> %s", path)
    return path


def _load_checkpoint(filename: str) -> pd.DataFrame | None:
    path = Path(RESULTS_DIR) / filename
    if path.exists():
        df = pd.read_csv(path, encoding="utf-8")
        logger.info("Loaded checkpoint: %s (%d rows)", path.name, len(df))
        return df
    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(
    regions: list[str] | None = None,
    skip_judge: bool = False,
    load_checkpoints: bool = False,
) -> None:
    _ensure_dirs()
    regions = regions or REGIONS

    # ── Step 1: Load data ──────────────────────────────────────────────────
    logger.info("=== Loading Betthupferl data ===")
    examples_df, test_df = load_all_regions(regions=regions)
    logger.info(
        "Example pool: %d rows | Test set: %d rows", len(examples_df), len(test_df)
    )

    # ── Steps 2–4: Run conditions (or load from checkpoints) ───────────────
    condition_dfs: dict[str, pd.DataFrame] = {}

    # Baseline-0 (no API, always fast)
    key0 = "Baseline-0 (Identity)"
    _ck0 = _load_checkpoint(CONDITION_FILES[key0]) if load_checkpoints else None
    if _ck0 is not None:
        condition_dfs[key0] = _ck0
    else:
        logger.info("=== Baseline-0: Identity Control ===")
        condition_dfs[key0] = run_identity(test_df)
        _save(condition_dfs[key0], CONDITION_FILES[key0])

    # Condition-A: Zero-shot
    keyA = "Condition-A (Zero-Shot)"
    _ckA = _load_checkpoint(CONDITION_FILES[keyA]) if load_checkpoints else None
    if _ckA is not None:
        condition_dfs[keyA] = _ckA
    else:
        logger.info("=== Condition-A: Zero-Shot LLM ===")
        condition_dfs[keyA] = run_zero_shot(test_df)
        _save(condition_dfs[keyA], CONDITION_FILES[keyA])

    # Condition-B: Few-shot
    keyB = "Condition-B (Few-Shot)"
    _ckB = _load_checkpoint(CONDITION_FILES[keyB]) if load_checkpoints else None
    if _ckB is not None:
        condition_dfs[keyB] = _ckB
    else:
        logger.info("=== Condition-B: Few-Shot LLM ===")
        condition_dfs[keyB] = run_few_shot(test_df, examples_df)
        _save(condition_dfs[keyB], CONDITION_FILES[keyB])

    # ── Step 5: Traditional metrics ────────────────────────────────────────
    logger.info("=== Computing traditional metrics ===")
    corpus_rows = []
    region_rows = []

    for name, df in condition_dfs.items():
        m = compute_metrics_from_df(df)
        corpus_rows.append(m)
        logger.info(
            "[%s]  WER=%.4f  CER=%.4f  BLEU=%.2f",
            name, m["wer"], m["cer"], m["bleu"],
        )
        region_rows.append(compute_per_region(df))

    corpus_metrics = pd.DataFrame(corpus_rows)
    region_metrics = pd.concat(region_rows, ignore_index=True)
    _save(corpus_metrics, "corpus_metrics.csv")
    _save(region_metrics, "region_metrics.csv")

    # ── Step 6: LLM-as-a-Judge ────────────────────────────────────────────
    judge_summary_rows = []

    if skip_judge:
        logger.warning("--skip-judge set: LLM judge skipped.")
        for name in condition_dfs:
            judge_summary_rows.append({
                "condition": name,
                "mean_meaning_preservation": 0.0,
                "mean_fluency": 0.0,
            })
    else:
        logger.info("=== LLM-as-a-Judge ===")
        for name, df in condition_dfs.items():
            judge_path = Path(RESULTS_DIR) / f"judge_{name.split('(')[1].rstrip(')').lower().replace('-','_')}.csv"
            if load_checkpoints and judge_path.exists():
                judged = pd.read_csv(judge_path, encoding="utf-8")
                logger.info("Loaded judge checkpoint for %s.", name)
            else:
                logger.info("Judging: %s", name)
                judged = llm_judge(df)
                judged.to_csv(judge_path, index=False, encoding="utf-8")
            judge_summary_rows.append(aggregate_judge(judged))
            logger.info(
                "[Judge][%s]  MeanMP=%.3f  MeanFL=%.3f",
                name,
                judge_summary_rows[-1]["mean_meaning_preservation"],
                judge_summary_rows[-1]["mean_fluency"],
            )

    judge_summary = pd.DataFrame(judge_summary_rows)
    _save(judge_summary, "judge_summary.csv")

    # ── Step 7: Full summary table ─────────────────────────────────────────
    full_summary = corpus_metrics.merge(judge_summary, on="condition", how="left")
    _save(full_summary, "full_summary.csv")

    print("\n" + "=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)
    print(full_summary.to_string(index=False))
    print("=" * 72 + "\n")

    # ── Step 8: Generate poster figures ────────────────────────────────────
    logger.info("=== Generating poster figures ===")
    p1 = plot_wer_by_condition(full_summary)
    p2 = plot_region_heatmap(region_metrics)
    p3 = plot_radar(full_summary)
    p4 = plot_wer_improvement(full_summary)

    print("Figures saved:")
    for p in (p1, p2, p3, p4):
        print(f"  {p}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3-tier ICL evaluation on Betthupferl dialect benchmark."
    )
    parser.add_argument(
        "--regions", nargs="+", default=None,
        help="Region codes to include (default: all 7). E.g. --regions oberbayern schwaben",
    )
    parser.add_argument(
        "--skip-judge", action="store_true",
        help="Skip LLM-as-a-Judge (saves API quota during development).",
    )
    parser.add_argument(
        "--load-checkpoints", action="store_true",
        help="Load pre-generated CSV checkpoints; skip LLM generation calls.",
    )
    args = parser.parse_args()

    main(
        regions=args.regions,
        skip_judge=args.skip_judge,
        load_checkpoints=args.load_checkpoints,
    )
