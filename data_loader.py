"""
data_loader.py — Load Betthupferl parallel text files into a pandas DataFrame.

Returns two splits per region:
  - example_pool  : first N_FEW_SHOT_EXAMPLES rows  (used as few-shot examples)
  - test_set      : next MAX_SENTENCES_PER_REGION rows  (evaluated in all tiers)

"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import DATA_ROOT, MAX_SENTENCES_PER_REGION, N_FEW_SHOT_EXAMPLES, REGIONS

logger = logging.getLogger(__name__)


def _read_lines(path: Path) -> list[str]:
    """Return non-empty, stripped lines from a UTF-8 text file."""
    with path.open(encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def load_region(
    region: str,
    data_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data for a single region, split into example pool and test set.

    Parameters
    ----------
    region   : Region code, e.g. "oberbayern".
    data_dir : Path to pure_text directory. Defaults to config.DATA_ROOT.

    Returns
    -------
    example_pool : DataFrame (N_FEW_SHOT_EXAMPLES rows) for few-shot prompting.
    test_set     : DataFrame (MAX_SENTENCES_PER_REGION rows) for evaluation.

    Both DataFrames have columns: region, dialect, reference.
    """
    data_dir = Path(data_dir or DATA_ROOT)
    dial_path = data_dir / f"{region}_gold_dial.txt"
    deu_path  = data_dir / f"{region}_gold_deu.txt"

    for p in (dial_path, deu_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    dial_lines = _read_lines(dial_path)
    deu_lines  = _read_lines(deu_path)

    n = min(len(dial_lines), len(deu_lines))
    if len(dial_lines) != len(deu_lines):
        logger.warning("Line count mismatch for '%s' — truncating to %d.", region, n)
    dial_lines, deu_lines = dial_lines[:n], deu_lines[:n]

    def _make_df(dial, deu):
        return pd.DataFrame({"region": region, "dialect": dial, "reference": deu})

    split = N_FEW_SHOT_EXAMPLES
    end   = split + MAX_SENTENCES_PER_REGION

    example_pool = _make_df(dial_lines[:split],   deu_lines[:split])
    test_set     = _make_df(dial_lines[split:end], deu_lines[split:end])

    logger.info(
        "Region '%s': %d few-shot examples | %d test sentences.",
        region, len(example_pool), len(test_set),
    )
    return example_pool, test_set


def load_all_regions(
    regions: list[str] | None = None,
    data_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all regions, returning combined example pool and test set DataFrames.

    Returns
    -------
    examples : Combined few-shot pool across all regions.
    test_df  : Combined test set across all regions.
    """
    regions = regions or REGIONS
    all_examples, all_tests = [], []

    for region in regions:
        try:
            examples, tests = load_region(region, data_dir)
            all_examples.append(examples)
            all_tests.append(tests)
        except FileNotFoundError as exc:
            logger.warning("Skipping '%s': %s", region, exc)

    if not all_tests:
        raise RuntimeError(
            f"No data found. Check DATA_ROOT='{data_dir or DATA_ROOT}' "
            "contains Betthupferl files."
        )

    examples_df = pd.concat(all_examples, ignore_index=True)
    test_df     = pd.concat(all_tests,    ignore_index=True)

    logger.info(
        "Total: %d few-shot examples | %d test sentences across %d regions.",
        len(examples_df), len(test_df), len(all_tests),
    )
    return examples_df, test_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    ex, test = load_all_regions()
    print(f"Example pool : {len(ex)} rows")
    print(f"Test set     : {len(test)} rows")
    print("\nTest set sample:")
    print(test.head(3).to_string())
