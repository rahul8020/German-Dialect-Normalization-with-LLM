"""
metrics.py — Evaluation metrics for the ICL dialect normalisation study.

Traditional metrics  (local, no API)
--------------------------------------
  BLEU  : sacrebleu corpus-level BLEU (exp smoothing, lowercase)
  WER   : jiwer Word Error Rate
  CER   : jiwer Character Error Rate

LLM-as-a-Judge  (Groq)
----------------------------------------------
  Scores each hypothesis on:
    - Meaning Preservation  (1–5)
    - Fluency               (1–5)
  Uses JSON-mode output for reliable parsing.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import List

import jiwer
import pandas as pd
from sacrebleu.metrics import BLEU

from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_REQUEST_DELAY,
    JUDGE_TEMPERATURE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Traditional metrics
# ---------------------------------------------------------------------------

def compute_bleu(hypotheses: List[str], references: List[str]) -> float:
    """Corpus-level BLEU (sacrebleu, lowercase, exp smoothing). Range 0–100."""
    bleu = BLEU(lowercase=True, smooth_method="exp")
    return bleu.corpus_score(hypotheses, [references]).score


def compute_wer(hypotheses: List[str], references: List[str]) -> float:
    """Corpus-level Word Error Rate. Range 0–1, lower is better."""
    transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    return float(jiwer.wer(
        references, hypotheses,
        reference_transform=transform,
        hypothesis_transform=transform,
    ))


def compute_cer(hypotheses: List[str], references: List[str]) -> float:
    """Corpus-level Character Error Rate. Range 0–1, lower is better."""
    transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfChars(),
    ])
    return float(jiwer.cer(
        references, hypotheses,
        reference_transform=transform,
        hypothesis_transform=transform,
    ))


def compute_all_traditional(
    hypotheses: List[str],
    references: List[str],
    condition_name: str = "condition",
) -> dict:
    """
    Compute BLEU, WER, CER in one call.

    Returns
    -------
    dict: {condition, bleu, wer, cer}
    """
    return {
        "condition": condition_name,
        "bleu":      round(compute_bleu(hypotheses, references), 2),
        "wer":       round(compute_wer(hypotheses, references), 4),
        "cer":       round(compute_cer(hypotheses, references), 4),
    }


def compute_metrics_from_df(df: pd.DataFrame) -> dict:
    """
    Convenience wrapper: compute metrics from a condition DataFrame.

    DataFrame must have columns: condition, hypothesis, reference.
    """
    hyps = df["hypothesis"].fillna("").tolist()
    refs = df["reference"].tolist()
    name = df["condition"].iloc[0]
    return compute_all_traditional(hyps, refs, condition_name=name)


def compute_per_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute WER, CER, BLEU per region within a condition DataFrame.

    Returns a DataFrame with columns: region, condition, bleu, wer, cer.
    """
    rows = []
    condition = df["condition"].iloc[0]
    for region, grp in df.groupby("region"):
        hyps = grp["hypothesis"].fillna("").tolist()
        refs = grp["reference"].tolist()
        rows.append({
            "region":    region,
            "condition": condition,
            "bleu":      round(compute_bleu(hyps, refs), 2),
            "wer":       round(compute_wer(hyps, refs), 4),
            "cer":       round(compute_cer(hyps, refs), 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LLM-as-a-Judge  (Groq free model: qwen/qwen3.8-27b)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """\
You are an expert evaluator for German language quality.
You will receive:
  1. The original dialect sentence
  2. The human Standard German reference translation
  3. A machine-generated hypothesis

Score the hypothesis on two dimensions from 1 to 5.

Meaning Preservation (1–5):
  5 = Exact same meaning as the reference.
  4 = Mostly correct; one minor detail differs.
  3 = General meaning preserved but one notable inaccuracy.
  2 = Significant meaning loss or distortion.
  1 = Meaning not conveyed at all.

Fluency (1–5):
  5 = Flawless, natural Standard German.
  4 = Mostly fluent; minor issues.
  3 = Comprehensible but with noticeable errors.
  2 = Difficult to read; multiple errors.
  1 = Unintelligible or not Standard German.

Respond ONLY with this JSON (no prose):
{"meaning_preservation": <int>, "fluency": <int>}
"""

_JUDGE_USER = (
    "Dialect source : {dialect}\n"
    "Reference (std): {reference}\n"
    "Hypothesis     : {hypothesis}"
)


def _parse_scores(raw: str) -> dict[str, int]:
    # Strip markdown codeblocks if present
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"```$", "", raw).strip()
    try:
        data = json.loads(raw.strip())
        return {
            "meaning_preservation": int(data.get("meaning_preservation", 0)),
            "fluency":              int(data.get("fluency", 0)),
        }
    except Exception:
        mp = re.search(r'"meaning_preservation"\s*:\s*(\d)', raw)
        fl = re.search(r'"fluency"\s*:\s*(\d)', raw)
        return {
            "meaning_preservation": int(mp.group(1)) if mp else 0,
            "fluency":              int(fl.group(1)) if fl else 0,
        }


def llm_judge(
    df: pd.DataFrame,
    request_delay: float = GROQ_REQUEST_DELAY,
) -> pd.DataFrame:
    """
    Run LLM-as-a-Judge on a condition DataFrame via Groq (qwen/qwen3.8-27b).

    Parameters
    ----------
    df : Must have columns: condition, dialect, reference, hypothesis.

    Returns
    -------
    DataFrame with added columns: meaning_preservation, fluency.
    """
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("Run: pip install groq")

    if not GROQ_API_KEY:
        raise ValueError(
            "Groq API key missing. Set GROQ_API_KEY in config.py.\n"
            "Free key: https://console.groq.com"
        )

    client = Groq(api_key=GROQ_API_KEY)

    mp_scores, fl_scores = [], []
    total = len(df)
    condition = df["condition"].iloc[0]

    for idx, (_, row) in enumerate(df.iterrows()):
        user_msg = _JUDGE_USER.format(
            dialect=row["dialect"],
            reference=row["reference"],
            hypothesis=row.get("hypothesis", ""),
        )
        scores = None
        max_retries = 5
        base_delay = 5.0
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": _JUDGE_SYSTEM},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=JUDGE_TEMPERATURE,
                    max_tokens=64,
                    response_format={"type": "json_object"},
                )
                raw = (resp.choices[0].message.content or "").strip()
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                parsed = _parse_scores(raw)
                if parsed["meaning_preservation"] > 0 or parsed["fluency"] > 0:
                    scores = parsed
                    break
            except Exception as exc:
                err = str(exc)
                if "429" in err or "rate_limit" in err.lower() or "503" in err:
                    wait = base_delay * (2 ** attempt)
                    logger.warning("[Judge][%s] Rate limit on row %d (attempt %d/%d), waiting %.0fs", condition, idx, attempt + 1, max_retries, wait)
                    time.sleep(wait)
                else:
                    logger.warning("[Judge][%s] Error row %d: %s", condition, idx, exc)
                    break
        if not scores:
            scores = {"meaning_preservation": 0, "fluency": 0}

        mp_scores.append(scores.get("meaning_preservation", 0))
        fl_scores.append(scores.get("fluency", 0))

        if (idx + 1) % 10 == 0:
            logger.info("[Judge][%s] %d/%d done.", condition, idx + 1, total)

        if request_delay > 0:
            time.sleep(request_delay)

    result = df.copy()
    result["meaning_preservation"] = mp_scores
    result["fluency"]              = fl_scores
    return result


def aggregate_judge(df: pd.DataFrame) -> dict:
    """Mean judge scores from a judged DataFrame."""
    return {
        "condition":                 df["condition"].iloc[0],
        "mean_meaning_preservation": round(df["meaning_preservation"].mean(), 3),
        "mean_fluency":              round(df["fluency"].mean(), 3),
    }
